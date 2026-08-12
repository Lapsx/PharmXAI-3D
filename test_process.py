import traceback
import os
from dataset import PDBbindDataset
from data_processor import PharmGraphBuilder

builder = PharmGraphBuilder()

# Find one valid pdb_id
raw_dir = "raw"
for folder in os.listdir(raw_dir):
    pdb_id = folder
    ligand_dir = os.path.join(raw_dir, pdb_id)
    if os.path.isdir(ligand_dir):
        protein_path = os.path.join(ligand_dir, f"{pdb_id}_protein.pdb")
        ligand_path = os.path.join(ligand_dir, f"{pdb_id}_ligand.sdf")
        if not os.path.exists(ligand_path):
            ligand_path = os.path.join(ligand_dir, f"{pdb_id}_ligand.mol2")
            if not os.path.exists(ligand_path):
                ligand_path = os.path.join(ligand_dir, f"{pdb_id}_ligand.pdb")
        
        try:
            print(f"Processing {pdb_id}...")
            graph = builder.build_hetero_graph(protein_path, ligand_path)
            print("Success!")
            break
        except Exception as e:
            traceback.print_exc()
            break
