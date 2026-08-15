import os
import urllib.request
from rdkit import Chem

# =========================================================================
# BANCO DE DADOS ALVO (Benchmark de GPCRs Farmacêuticos)
# =========================================================================
LIBRARY_CONFIG = {
    "Dopamina_D3": {
        "pdb_id": "3PBL",
        "native_code": "ETQ",
        "ligands": ["Haloperidol", "Pramipexole", "Eticlopride", "Dopamine"]
    },
    "Adenosina_A2A": {
        "pdb_id": "4EIY",
        "native_code": "ZMA",
        "ligands": ["Caffeine", "Adenosine", "Regadenoson"]
    },
    "Opioide_Delta": {
        "pdb_id": "4N6H",
        "native_code": "EJ4",
        "ligands": ["Naltrindole", "Enkephalin", "Morphine"]
    },
    "Opioide_Kappa": {
        "pdb_id": "4DJH",
        "native_code": "JDC",
        "ligands": ["Salvinorin A", "Naloxone", "Fentanyl"]
    },
    "Opioide_Mu": {
        "pdb_id": "4DKL",
        "native_code": "BF0",
        "ligands": ["Tramadol", "Hydrocodone", "Acetylmorphine"]
    }
}

def fetch_pdb_and_clean(pdb_id, output_path, native_code):
    print(f"[*] Baixando Receptor {pdb_id} do Banco de Dados Mundial (RCSB PDB)...")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    raw_pdb = output_path.replace(".pdb", "_raw.pdb")
    native_pdb = output_path.replace("_receptor.pdb", "_native_crystal.pdb")
    
    try:
        urllib.request.urlretrieve(url, raw_pdb)
    except Exception as e:
        print(f"    [!] Falha ao baixar PDB {pdb_id}: {e}")
        return False

    print(f"    [+] Extraindo Receptor limpo e Ligante Nativo ({native_code}) do cristal...")
    with open(raw_pdb, 'r') as f_in, open(output_path, 'w') as f_out, open(native_pdb, 'w') as f_nat:
        for line in f_in:
            if line.startswith("ATOM  "):
                f_out.write(line)
            if line.startswith("TER") or line.startswith("END"):
                f_out.write(line)
                
            # Extrair o ligante nativo para servir de 'molde' para o Docking
            if line.startswith("HETATM") and native_code in line:
                f_nat.write(line)
                
    os.remove(raw_pdb)
    return True

def fetch_ligand_sdf(ligand_name, output_dir):
    print(f"[*] Buscando ligante '{ligand_name}' no PubChem...")
    # Usa a API REST do PubChem para pegar a conformação 3D (se disponível) ou 2D (que o OpenMM/RDKit otimiza)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{ligand_name}/SDF?record_type=3d"
    
    out_path = os.path.join(output_dir, f"{ligand_name.lower().replace(' ', '_')}.sdf")
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"    [+] Salvo em: {out_path}")
    except Exception:
        # Tenta pegar a versão 2D caso a 3D falhe (o minimizador do script predict_custom.py resolve o 3D)
        try:
            url_2d = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{ligand_name}/SDF?record_type=2d"
            urllib.request.urlretrieve(url_2d, out_path)
            
            # Converte 2D para 3D rápido usando RDKit
            mol = Chem.MolFromMolFile(out_path)
            if mol:
                mol = Chem.AddHs(mol)
                Chem.AllChem.EmbedMolecule(mol, randomSeed=42)
                Chem.AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
                writer = Chem.SDWriter(out_path)
                writer.write(mol)
                writer.close()
            print(f"    [+] (Versão 2D otimizada para 3D) Salva em: {out_path}")
        except Exception as e:
            print(f"    [!] Falha ao buscar ligante '{ligand_name}': {e}")

def build_library():
    base_dir = "mini_library_gpcr"
    os.makedirs(base_dir, exist_ok=True)
    
    for family, data in LIBRARY_CONFIG.items():
        print(f"\n" + "="*50)
        print(f"🧬 MONTANDO FAMÍLIA: {family}")
        print("="*50)
        
        fam_dir = os.path.join(base_dir, family)
        os.makedirs(fam_dir, exist_ok=True)
        
        # 1. Prepara Receptor e Extrai o Molde
        pdb_path = os.path.join(fam_dir, f"{data['pdb_id']}_receptor.pdb")
        fetch_pdb_and_clean(data['pdb_id'], pdb_path, data['native_code'])
        
        # 2. Prepara Ligantes
        for lig in data['ligands']:
            fetch_ligand_sdf(lig, fam_dir)
            
    print("\n[✔] Mini-Biblioteca concluída! Pronta para testes Cross-Reactivity.")

if __name__ == "__main__":
    build_library()
