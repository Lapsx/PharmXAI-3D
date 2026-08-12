import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os

from dataset import PDBbindDataset
from gnn_model import PharmGeometricGNN

def load_refined_ids(filepath="refined_ids.txt"):
    """Lê o arquivo gerado pelo bash script com os IDs oficiais de Validação/Teste."""
    if not os.path.exists(filepath):
        raise FileNotFoundError("O arquivo refined_ids.txt não foi encontrado! Rode o flatten_raw.sh primeiro.")
    
    with open(filepath, 'r') as f:
        return set(line.strip().lower() for line in f if line.strip())

def calculate_lj_potential(edge_attr, edge_index, batch, num_graphs, sigma=3.0):
    """
    Computa a aproximação de potencial estérico (Lennard-Jones proxy) 
    baseado exclusivamente na geometria estática.
    """
    r = torch.clamp(edge_attr, min=0.5)
    term = (sigma / r) ** 6
    lj_energy = term ** 2 - 2 * term
    
    # Mapear cada aresta para seu respectivo grafo no batch
    edge_batch = batch[edge_index[0]].unsqueeze(1)
    
    # Somar energia estérico por grafo
    lj_graph = torch.zeros(num_graphs, 1, device=edge_attr.device)
    lj_graph.scatter_add_(0, edge_batch, lj_energy)
    
    # Normalização robusta para evitar targets infinitos
    return torch.clamp(lj_graph, min=-10.0, max=10.0)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Treinando em: {device}")

    print("Carregando o Banco de Dados...")
    dataset = PDBbindDataset(root='.')
    
    # =========================================================================
    # CORTE HERMÉTICO: DATA LEAKAGE = ZERO
    # =========================================================================
    print("Separando General Set (Treino) e Refined Set (Validação)...")
    refined_set_ids = load_refined_ids()
    
    train_dataset = []
    val_dataset = []
    
    for graph in dataset:
        if not hasattr(graph, 'pdb_id'):
            continue 
            
        if graph.pdb_id in refined_set_ids:
            val_dataset.append(graph)
        else:
            train_dataset.append(graph)
            
    print(f"Total: {len(dataset)} | Treino (General): {len(train_dataset)} | Validação (Refined): {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Inicializar o Modelo (Dropout ativo = 0.2 default)
    model = PharmGeometricGNN(node_dim=64, num_gaussians=32, num_layers=4).to(device)
    
    # Regularização L2: weight_decay penaliza pesos gigantes
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    criterion = nn.MSELoss()
    lambda_physics = 0.1 # Peso da restrição física na Loss global

    epochs = 300
    print(f"Iniciando o treinamento rigoroso por {epochs} épocas...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss_pkd = 0
        total_loss_physics = 0
        
        for data in train_loader:
            data_homo = data.to_homogeneous().to(device)
            target_pkd = data.y.to(device)
            
            # Gerar target de física dinamicamente
            target_steric = calculate_lj_potential(
                data_homo.edge_attr, data_homo.edge_index, data_homo.batch, data.num_graphs
            )
            
            optimizer.zero_grad()
            out_pkd, out_steric = model(data_homo)
            
            loss_pkd = criterion(out_pkd, target_pkd)
            loss_physics = criterion(out_steric, target_steric)
            
            # Loss Multi-Tarefa
            loss = loss_pkd + lambda_physics * loss_physics
            
            loss.backward()
            optimizer.step()
            
            total_loss_pkd += loss_pkd.item() * data.num_graphs
            total_loss_physics += loss_physics.item() * data.num_graphs
            
        avg_train_loss_pkd = total_loss_pkd / len(train_dataset)
        avg_train_loss_phys = total_loss_physics / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss_pkd = 0
        with torch.no_grad():
            for data in val_loader:
                data_homo = data.to_homogeneous().to(device)
                target_pkd = data.y.to(device)
                out_pkd, _ = model(data_homo) # Validação baseia-se apenas no pKd
                val_loss_pkd += criterion(out_pkd, target_pkd).item() * data.num_graphs
                
        avg_val_loss_pkd = val_loss_pkd / len(val_dataset)
        
        # O Scheduler escuta estritamente a Loss de pKd da Validação
        scheduler.step(avg_val_loss_pkd)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Época [{epoch+1:03d}/{epochs:03d}] | Train pKd: {avg_train_loss_pkd:.4f} | Train Phys: {avg_train_loss_phys:.4f} | Val pKd: {avg_val_loss_pkd:.4f} | LR: {current_lr:.6f}")
        
        # =========================================================================
        # EARLY STOPPING E CHECKPOINTING DE MELHOR MODELO
        # =========================================================================
        if avg_val_loss_pkd < best_val_loss:
            best_val_loss = avg_val_loss_pkd
            torch.save(model.state_dict(), "pharm_model_weights_best.pth")
            print(f"  -> Novo melhor modelo salvo! (Val pKd: {best_val_loss:.4f})")
            
    print(f"Treinamento científico finalizado! O melhor modelo alcançou Val pKd: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
