import urllib.request
import os
from data_processor import PharmGraphBuilder

def fetch_and_split_1hsg():
    print("Baixando o complexo 1HSG (HIV Protease + Indinavir)...")
    url = "https://files.rcsb.org/download/1hsg.pdb"
    urllib.request.urlretrieve(url, "1hsg.pdb")

    # Separando Proteína (ATOM) e Ligante Indinavir (MK1)
    with open("1hsg.pdb", "r") as f:
        lines = f.readlines()

    with open("protein.pdb", "w") as fp, open("ligand.pdb", "w") as fl:
        for line in lines:
            if line.startswith("ATOM"):
                fp.write(line)
            elif line.startswith("HETATM") and "MK1" in line:
                fl.write(line)

    print("Complexo separado com sucesso em 'protein.pdb' e 'ligand.pdb'.")

if __name__ == "__main__":
    fetch_and_split_1hsg()
    
    print("\nIniciando PharmGraphBuilder...")
    builder = PharmGraphBuilder(pocket_radius=6.0)
    
    # Construir Grafo
    graph = builder.build_hetero_graph("protein.pdb", "ligand.pdb")
    
    print("\n--- Relatório do Grafo Heterogêneo Bipartido (HeteroData) ---")
    print(graph)
    print(f"\nNúmero de nós do Ligante: {graph['ligand'].num_nodes}")
    print(f"Número de nós do Bolso da Proteína (< 6.0A): {graph['protein'].num_nodes}")
    print(f"Número de Arestas (Interações a < 4.0A): {graph['ligand', 'interacts_with', 'protein'].num_edges}")
    
    # Mostrar algumas distâncias de arestas
    edges = graph['ligand', 'interacts_with', 'protein'].edge_attr
    if len(edges) > 0:
        print(f"\nDistância mínima de interação: {edges.min().item():.2f} Angstroms")
        print(f"Distância média de interação: {edges.mean().item():.2f} Angstroms")
