import torch
from torch_geometric.data import InMemoryDataset
import os
import re
import math
from data_processor import PharmGraphBuilder

class PDBbindDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # Como as subpastas podem variar (19.000 IDs), não listamos todas aqui rigidamente
        return []

    @property
    def processed_file_names(self):
        return ['pharm_xai_dataset.pt']

    def parse_labels(self):
        """Lê o arquivo INDEX_general_PL.2020R1.lst e converte as afinidades para -log(Kd/Ki)"""
        labels = {}
        index_file = os.path.join(self.root, 'index', 'INDEX_general_PL.2020R1.lst')
        
        if not os.path.exists(index_file):
            print(f"AVISO: Arquivo de rótulos não encontrado em {index_file}")
            return labels
            
        print("Mapeando Afinidades Termodinâmicas experimentais do arquivo INDEX...")
        with open(index_file, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) < 4: continue
                
                pdb_id = parts[0]
                binding_data = parts[3] # Ex: Kd=49uM ou Ki=0.35nM
                
                # Regex para extrair valor e unidade
                match = re.search(r'(=|~|<|>)\s*([\d\.]+)\s*(mM|uM|nM|pM|fM)', binding_data)
                if match:
                    val = float(match.group(2))
                    unit = match.group(3)
                    
                    # Convertendo tudo para Molar (M) absoluto
                    if unit == 'mM': val *= 1e-3
                    elif unit == 'uM': val *= 1e-6
                    elif unit == 'nM': val *= 1e-9
                    elif unit == 'pM': val *= 1e-12
                    elif unit == 'fM': val *= 1e-15
                    
                    if val > 0:
                        # pKd = -log10(Kd)
                        pKd = -math.log10(val)
                        labels[pdb_id] = pKd
        
        print(f"Rótulos (-logK) mapeados para {len(labels)} complexos.")
        return labels

    def process(self):
        print("Iniciando o processamento dos arquivos brutos (.pdb/.sdf) em lote...")
        data_list = []
        builder = PharmGraphBuilder(pocket_radius=6.0)
        
        # 1. Carregar o Dicionário de Afinidades (Ground Truth / Labels)
        affinity_dict = self.parse_labels()

        folders = [f for f in os.listdir(self.raw_dir) if os.path.isdir(os.path.join(self.raw_dir, f))]
        print(f"Encontrados {len(folders)} complexos na pasta raw/. Processando...")

        for folder in folders:
            pdb_id = folder.lower()
            if pdb_id not in affinity_dict:
                continue # Ignoramos complexos sem rótulo experimental
                
            folder_path = os.path.join(self.raw_dir, folder)
            protein_path = os.path.join(folder_path, f"{pdb_id}_protein.pdb")
            
            # PDBbind costuma usar SDF, MOL2 ou PDB
            ligand_sdf = os.path.join(folder_path, f"{pdb_id}_ligand.sdf")
            ligand_mol2 = os.path.join(folder_path, f"{pdb_id}_ligand.mol2")
            ligand_pdb = os.path.join(folder_path, f"ligand.pdb") 
            
            if os.path.exists(ligand_sdf): ligand_path = ligand_sdf
            elif os.path.exists(ligand_mol2): ligand_path = ligand_mol2
            else: ligand_path = ligand_pdb

            if os.path.exists(protein_path) and os.path.exists(ligand_path):
                try:
                    # 2. Constrói a Física do Grafo
                    graph = builder.build_hetero_graph(protein_path, ligand_path)
                    
                    # 3. Adiciona a Resposta Certa (Loss Target)
                    graph.y = torch.tensor([[affinity_dict[pdb_id]]], dtype=torch.float)
                    
                    # Salva o ID (crachá) para podermos fazer o corte Refined vs General depois!
                    graph.pdb_id = pdb_id
                    
                    data_list.append(graph)
                except Exception as e:
                    pass

        print(f"\n{len(data_list)} grafos rotulados com sucesso!")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dataset salvo em: {self.processed_paths[0]}")

if __name__ == "__main__":
    dataset = PDBbindDataset(root='.')
    print("\nResumo:")
    print(f"Tamanho total: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Amostra [0]: {dataset[0]}")
        print(f"Target de Afinidade (data.y) [0]: {dataset[0].y.item():.3f}")
