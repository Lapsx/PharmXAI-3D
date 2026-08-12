import traceback
from rdkit import Chem

try:
    # Create a dummy PDB block
    pdb_block = """
ATOM      1  C   UNL A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  O   UNL A   1       1.200   0.000   0.000  1.00  0.00           O
END
"""
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False)
    for atom in mol.GetAtoms():
        Z = atom.GetAtomicNum()
        mass = atom.GetMass()
        degree = atom.GetDegree()
        arom = atom.GetIsAromatic()
        h = atom.GetHybridization()
        print(f"Z={Z}, Mass={mass}, Degree={degree}, Arom={arom}, Hyb={h}")
except Exception as e:
    traceback.print_exc()
