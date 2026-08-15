import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool

class GaussianSmearing(nn.Module):
    """
    Expande a distância escalar Euclidiana em um conjunto contínuo de Funções de Base Radial (RBF).
    Isso é crucial para invariância SE(3), mapeando distâncias físicas para um manifold suave.
    """
    def __init__(self, start=0.0, stop=10.0, num_gaussians=32):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class ContinuousFilterConv(MessagePassing):
    """
    Convolução de Filtro Contínuo (inspirado no SchNet).
    A distância geométrica RBF modula as matrizes de pesos, evitando a concatenação ingênua.
    """
    def __init__(self, node_dim=64, num_gaussians=32):
        super(ContinuousFilterConv, self).__init__(aggr='add')
        
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # O filtro contínuo W atua como um portão (gate) modulado pela distância espacial
        W = self.filter_network(edge_attr)
        return x_j * W

    def update(self, aggr_out, x):
        # Conexão residual estrita (ResNet style) para evitar vanishing gradients
        return x + self.update_mlp(aggr_out)

class NodeEncoder(nn.Module):
    def __init__(self, max_z=100, z_emb_dim=32, phys_feat_dim=10, out_dim=64, dropout=0.2):
        super().__init__()
        self.z_embedding = nn.Embedding(max_z, z_emb_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(z_emb_dim + phys_feat_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_raw):
        # x_raw[:, 0] é o Número Atômico Z (índice categórico)
        # x_raw[:, 1:] são os 10 Tensores Físicos
        z_idx = x_raw[:, 0].long()
        phys_features = x_raw[:, 1:]
        
        z_emb = self.z_embedding(z_idx)
        h_0 = torch.cat([z_emb, phys_features], dim=-1)
        return self.fusion_mlp(h_0)

class PharmGeometricGNN(nn.Module):
    def __init__(self, node_dim=64, num_gaussians=32, num_layers=4, dropout=0.2):
        super(PharmGeometricGNN, self).__init__()
        
        self.node_encoder = NodeEncoder(max_z=100, z_emb_dim=32, phys_feat_dim=10, out_dim=node_dim, dropout=dropout)
        self.rbf_expansion = GaussianSmearing(start=0.0, stop=10.0, num_gaussians=num_gaussians)
        
        self.layers = nn.ModuleList([
            ContinuousFilterConv(node_dim, num_gaussians) for _ in range(num_layers)
        ])
        
        # Predictor Primário: Afinidade de Ligação (pKd)
        self.predictor = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Predictor Auxiliar: Regularização Física (Potencial Estérico)
        self.steric_predictor = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, data_homo):
        x, edge_index, pos, batch = data_homo.x, data_homo.edge_index, data_homo.pos, data_homo.batch
        
        # Computação Dinâmica Diferenciável da Distância Euclidiana (Crucial para Autograd)
        row, col = edge_index
        edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1).view(-1, 1)
        
        # 1. Encoding Inicial Hibrido (Embedding Categórico + Features Químicas)
        x = self.node_encoder(x)
        
        # 2. Expansão Contínua de Distâncias (Invariância Translacional e Rotacional)
        rbf_attr = self.rbf_expansion(edge_attr)
        
        # 3. Propagação Geométrica Modulada
        for layer in self.layers:
            x = layer(x, edge_index, rbf_attr)
            
        # 4. Readout Global
        x_graph = global_mean_pool(x, batch)
        
        # 5. Saídas Multi-Tarefa
        pkd_pred = self.predictor(x_graph)
        steric_pred = self.steric_predictor(x_graph)
        
        return pkd_pred, steric_pred
