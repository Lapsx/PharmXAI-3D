import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import radius_graph
from rdkit import Chem
from scipy.spatial import KDTree
import numpy as np

def custom_radius_graph(x, r, loop=False):
    """
    Native PyTorch fallback for radius_graph.
    Bypasses the 'torch_cluster' C++ extension dependency which causes AttributeError: 'NoneType' object.
    """
    dist = torch.cdist(x, x)
    mask = dist <= r
    if not loop:
        mask.fill_diagonal_(False)
    return mask.nonzero(as_tuple=False).t().contiguous()

class PharmGraphBuilder:
    def __init__(self, pocket_radius=6.0, interaction_radius=5.0, covalent_radius=2.0):
        self.pocket_radius = pocket_radius
        self.interaction_radius = interaction_radius
        self.covalent_radius = covalent_radius

    def get_atom_features(self, atom):
        """
        Extrai um vetor rico de propriedades químicas de cada átomo.
        Tamanho do Vetor (Input Features) = 11
        """
        hybridization = atom.GetHybridization()
        hyb_encoded = [
            1.0 if hybridization == Chem.rdchem.HybridizationType.SP else 0.0,
            1.0 if hybridization == Chem.rdchem.HybridizationType.SP2 else 0.0,
            1.0 if hybridization == Chem.rdchem.HybridizationType.SP3 else 0.0,
            1.0 if hybridization == Chem.rdchem.HybridizationType.SP3D else 0.0,
            1.0 if hybridization == Chem.rdchem.HybridizationType.SP3D2 else 0.0,
            1.0 if hybridization == Chem.rdchem.HybridizationType.UNSPECIFIED else 0.0,
        ]
        
        return [
            float(atom.GetAtomicNum()),      # Z
            float(atom.GetMass()),           # Massa atômica
            float(atom.GetDegree()),         # Número de ligações
            1.0 if atom.GetIsAromatic() else 0.0, # Aromaticidade
            float(atom.GetFormalCharge())    # Carga Formal (Eletrostática)
        ] + hyb_encoded

    def parse_ligand(self, ligand_path):
        if ligand_path.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(ligand_path, removeHs=True, sanitize=False)
        else:
            mol = Chem.MolFromMolFile(ligand_path, removeHs=True, sanitize=True)
            
        if mol is None: raise ValueError("Falha ao ler ligante")
        
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        features = [self.get_atom_features(atom) for atom in mol.GetAtoms()]
        
        return np.array(positions), np.array(features)

    def parse_protein(self, protein_path):
        mol = Chem.MolFromPDBFile(protein_path, removeHs=True, sanitize=False)
        if mol is None: raise ValueError("Falha ao ler proteina")
        
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        features = [self.get_atom_features(atom) for atom in mol.GetAtoms()]
        
        return np.array(positions), np.array(features)

    def build_hetero_graph(self, protein_path, ligand_path):
        lig_pos, lig_feat = self.parse_ligand(ligand_path)
        prot_pos, prot_feat = self.parse_protein(protein_path)

        # Filtrar o bolso (Pocket) usando KDTree
        tree_prot = KDTree(prot_pos)
        pocket_indices = tree_prot.query_ball_point(lig_pos, r=self.pocket_radius)
        pocket_indices = np.unique([idx for sublist in pocket_indices for idx in sublist]).astype(int)
        
        if len(pocket_indices) == 0:
            raise ValueError(
                "Nenhum átomo da proteína foi encontrado próximo ao ligante! "
                "Isso significa que as coordenadas do ligante (.sdf) estão fora do bolso do receptor. "
                "Você precisa realizar o Docking Molecular (Ex: AutoDock Vina) ou Alinhamento 3D "
                "para posicionar o fármaco dentro da proteína antes de prever a afinidade."
            )
            
        pocket_pos = prot_pos[pocket_indices]
        pocket_feat = prot_feat[pocket_indices]

        data = HeteroData()

        # ==============================================================
        # 1. NÓS (Atribuindo a química quântica)
        # ==============================================================
        data['ligand'].x = torch.tensor(lig_feat, dtype=torch.float)
        data['ligand'].pos = torch.tensor(lig_pos, dtype=torch.float)

        data['protein'].x = torch.tensor(pocket_feat, dtype=torch.float)
        data['protein'].pos = torch.tensor(pocket_pos, dtype=torch.float)

        # ==============================================================
        # 2. ARESTAS INTRAMOLECULARES (O "Esqueleto" das moléculas)
        # ==============================================================
        # Conecta átomos do ligante entre si (Ligações covalentes)
        lig_edge_idx = custom_radius_graph(data['ligand'].pos, r=self.covalent_radius, loop=False)
        lig_edge_attr = torch.norm(data['ligand'].pos[lig_edge_idx[0]] - data['ligand'].pos[lig_edge_idx[1]], dim=1).view(-1, 1)
        data['ligand', 'covalent', 'ligand'].edge_index = lig_edge_idx
        data['ligand', 'covalent', 'ligand'].edge_attr = lig_edge_attr

        # Conecta átomos da proteína entre si (Backbone e cadeias laterais)
        prot_edge_idx = custom_radius_graph(data['protein'].pos, r=self.covalent_radius, loop=False)
        prot_edge_attr = torch.norm(data['protein'].pos[prot_edge_idx[0]] - data['protein'].pos[prot_edge_idx[1]], dim=1).view(-1, 1)
        data['protein', 'covalent', 'protein'].edge_index = prot_edge_idx
        data['protein', 'covalent', 'protein'].edge_attr = prot_edge_attr

        # ==============================================================
        # 3. ARESTAS INTERMOLECULARES (A Interação Droga-Alvo)
        # ==============================================================
        tree_pocket = KDTree(pocket_pos)
        interaction_pairs = tree_pocket.query_ball_point(lig_pos, r=self.interaction_radius)

        lig_edges = []
        prot_edges = []
        edge_dist = []

        for lig_idx, prot_neighbors in enumerate(interaction_pairs):
            for prot_idx in prot_neighbors:
                lig_edges.append(lig_idx)
                prot_edges.append(prot_idx)
                dist = np.linalg.norm(lig_pos[lig_idx] - pocket_pos[prot_idx])
                edge_dist.append(dist)

        inter_edge_index = torch.tensor([lig_edges, prot_edges], dtype=torch.long)
        inter_edge_attr = torch.tensor(edge_dist, dtype=torch.float).view(-1, 1)

        # Aresta Ligante -> Proteína
        data['ligand', 'interacts', 'protein'].edge_index = inter_edge_index
        data['ligand', 'interacts', 'protein'].edge_attr = inter_edge_attr
        
        # Aresta Proteína -> Ligante (Fundamental para a troca de mensagens bidirecional)
        rev_inter_edge_index = torch.tensor([prot_edges, lig_edges], dtype=torch.long)
        data['protein', 'interacts', 'ligand'].edge_index = rev_inter_edge_index
        data['protein', 'interacts', 'ligand'].edge_attr = inter_edge_attr

        return data
