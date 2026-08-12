import torch
import numpy as np
import os
from dataset import PDBbindDataset
from gnn_model import PharmGeometricGNN

def explain_pharmacophore(pdb_id="4jwr"):
    """
    Executa o mapeamento de Saliência Física (Input x Gradient) para 
    determinar quais distâncias estruturais governam a predição de afinidade.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Inicializando Explainer na arquitetura: {device}")
    
    # 1. Carregar Modelo Otimizado
    model = PharmGeometricGNN(node_dim=64, num_gaussians=32, num_layers=4).to(device)
    model.load_state_dict(torch.load("pharm_model_weights_best.pth", map_location=device, weights_only=True))
    model.eval()

    # 2. Varredura do Dataset para isolar o complexo alvo
    print(f"Extraindo topologia do complexo {pdb_id}...")
    dataset = PDBbindDataset(root='.')
    graph = next((g for g in dataset if hasattr(g, 'pdb_id') and g.pdb_id == pdb_id), None)
    
    if graph is None:
        raise ValueError(f"ERRO: Complexo {pdb_id} não encontrado no dataset processado.")

    data_homo = graph.to_homogeneous().to(device)
    
    # 3. Habilitar rastreamento de gradiente na Variedade Espacial (Distâncias)
    # Isso permite calcular a derivada parcial: d(pKd) / d(Distância_Atomica)
    data_homo.edge_attr.requires_grad_(True)

    # 4. Propagação Forward
    pkd_pred, _ = model(data_homo)
    print(f"\n[{pdb_id}] Afinidade Prevista (pKd): {pkd_pred.item():.4f} | Experimental Real: {graph.y.item():.4f}")

    # 5. Saliency Mapping via Backpropagation (Input x Gradient)
    model.zero_grad()
    pkd_pred.backward()
    
    # A magnitude do gradiente indica a sensibilidade da rede àquela aresta específica
    edge_importance = data_homo.edge_attr.grad.abs().squeeze().cpu().numpy()
    
    # 6. Mapeamento Contínuo (Saliency Distribuído - Focado no LIGANTE)
    edge_index = data_homo.edge_index.cpu().numpy()
    
    # Extrair limites de indexação (O PyG to_homogeneous ordena alfabeticamente: 'ligand', 'protein')
    # Portanto, os nós do ligante ocupam os índices de 0 até num_ligand_nodes - 1
    num_ligand_nodes = graph['ligand'].num_nodes
    
    # Agregar importância das arestas para os nós
    node_importance = np.zeros(data_homo.num_nodes)
    for idx, val in enumerate(edge_importance):
        u, v = edge_index[0, idx], edge_index[1, idx]
        node_importance[u] += val
        node_importance[v] += val
        
    # ATUALIZAÇÃO ARQUITETURAL: Isolar a Saliency APENAS para o Ligante
    ligand_importance = node_importance[:num_ligand_nodes]
    
    # Calcular percentis matemáticos estritamente sobre o ligante
    p99 = np.percentile(ligand_importance, 99)
    p95 = np.percentile(ligand_importance, 95)
    p85 = np.percentile(ligand_importance, 85)
    p70 = np.percentile(ligand_importance, 70)
    
    # Calcular arestas ultra-críticas (Top 1%) envolvendo o ligante
    edge_threshold = np.percentile(edge_importance, 99)
    critical_edges_idx = np.where(edge_importance >= edge_threshold)[0]
    
    pos = data_homo.pos.cpu().numpy()
    
    # 7. Projeção Vetorial para o Web Viewer (3Dmol.js)
    html_filename = f"pharmacophore_{pdb_id}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ligand Pharmacophore Extractor: {pdb_id}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; background-color: #111; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #info {{ position: absolute; top: 20px; left: 20px; color: #fff; font-family: 'Courier New', Courier, monospace; z-index: 10; background: rgba(0,0,0,0.85); padding: 20px; border: 1px solid #444; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }}
    </style>
</head>
<body>
    <div id="info">
        <b style="color: #0f0; font-size: 1.2em;">PharmXAI-3D: Ligand Pharmacophore</b><br><br>
        <b>Target:</b> {pdb_id}<br>
        <b>Afinidade Prevista:</b> {pkd_pred.item():.2f} pKd<br>
        <b>Afinidade Real:</b> {graph.y.item():.2f} pKd<br><br>
        <hr style="border-color: #444;">
        <b>FARMACOFORO ISOLADO (LIGANTE):</b><br><br>
        <span style="color:magenta;">● Top 1% (Magenta):</span> Grupos Funcionais Vitais<br>
        <span style="color:orange;">● Top 5% (Laranja):</span> Contribuição Eletrostática Forte<br>
        <span style="color:yellow;">● Top 15% (Amarelo):</span> Contribuição Estérica / Secundária<br>
        <span style="color:lime;">● Top 30% (Verde):</span> Tolerância Leve<br>
        <span style="color:magenta;">- Linhas Tracejadas:</span> Vetores Diretos Receptor-Ligante<br>
        <span style="color:cyan;">■ Ciano Base:</span> Scaffold Neutro / Modificável<br>
    </div>
    <div id="container"></div>
    <script>
        $(document).ready(function() {{
            let viewer = $3Dmol.createViewer($('#container'), {{backgroundColor: '#111'}});
            
            // Carrega Proteina
            $.get("raw/{pdb_id}/{pdb_id}_protein.pdb", function(pdb_data) {{
                viewer.addModel(pdb_data, "pdb");
                viewer.setStyle({{model: 0}}, {{cartoon: {{color: 'white', opacity: 0.15}}}});
                
                // Carrega Ligante
                $.get("raw/{pdb_id}/{pdb_id}_ligand.sdf", function(sdf_data) {{
                    viewer.addModel(sdf_data, "sdf");
                    // Arcabouço base em Ciano (Tolerante)
                    viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'cyanCarbon', radius: 0.15}}}});
                    
                    // 1. Renderiza Vetores de Interação Crítica (Arestas Top 1%)
''')
        for idx in critical_edges_idx:
            u, v = edge_index[0, idx], edge_index[1, idx]
            p1, p2 = pos[u], pos[v]
            f.write(f"                    viewer.addCylinder({{start: {{x:{p1[0]:.3f}, y:{p1[1]:.3f}, z:{p1[2]:.3f}}}, end: {{x:{p2[0]:.3f}, y:{p2[1]:.3f}, z:{p2[2]:.3f}}}, radius: 0.05, color: 'magenta', dashed: true}});\n")
            
        f.write('''
                    // 2. Renderiza o Gradiente Contínuo na Nuvem Farmacofórica (Esferas)
''')
        z_array = data_homo.x[:, 0].cpu().numpy()
        element_map = {6: ('C', 'gray'), 7: ('N', 'blue'), 8: ('O', 'red'), 16: ('S', 'yellow'), 15: ('P', 'orange'), 9: ('F', 'green'), 17: ('Cl', 'green')}
        
        # Render Nodes as Gradient Spheres + Elemental Cores (LIGAND ONLY)
        for i, val in enumerate(ligand_importance):
            if val < p70: continue # Skip noise
            
            color = 'lime'
            radius = 0.2
            alpha = 0.4
            
            if val >= p99:
                color = 'magenta'
                radius = 0.5
                alpha = 0.8
            elif val >= p95:
                color = 'orange'
                radius = 0.4
                alpha = 0.7
            elif val >= p85:
                color = 'yellow'
                radius = 0.3
                alpha = 0.6
                
            p = pos[i]
            z = int(z_array[i])
            elem, elem_color = element_map.get(z, ('X', 'white'))
            
            # 1. Renderiza o Átomo Real (Núcleo Sólido) com cor CPK padrão
            f.write(f"                    viewer.addSphere({{center: {{x:{p[0]:.3f}, y:{p[1]:.3f}, z:{p[2]:.3f}}}, radius: 0.12, color: '{elem_color}', alpha: 1.0}});\n")
            
            # 2. Renderiza a Nuvem de Saliency (Esfera Transparente)
            f.write(f"                    viewer.addSphere({{center: {{x:{p[0]:.3f}, y:{p[1]:.3f}, z:{p[2]:.3f}}}, radius: {radius}, color: '{color}', alpha: {alpha}}});\n")

            # 3. Adiciona Rótulo de Texto apenas para os átomos super críticos (Evitar poluição)
            if val >= p95:
                f.write(f"                    viewer.addLabel('{elem}', {{position: {{x:{p[0]:.3f}, y:{p[1]:.3f}, z:{p[2]:.3f}}}, backgroundColor: 'rgba(0,0,0,0.6)', fontColor: 'white', fontSize: 11, backgroundOpacity: 0.6}});\n")

        f.write('''
                    // Adiciona Interatividade de Mouse (Hover) para identificar átomos
                    viewer.setHoverable({}, true, 
                        function(atom, viewer, event, container) {
                            if(!atom.label) {
                                let atomName = atom.elem + (atom.resn ? " (" + atom.resn + " " + (atom.resi || "") + ")" : "");
                                atom.label = viewer.addLabel(atomName, {position: atom, backgroundColor: 'rgba(0,255,0,0.8)', fontColor: 'black', fontSize: 14});
                            }
                        },
                        function(atom, viewer) {
                            if(atom.label) {
                                viewer.removeLabel(atom.label);
                                delete atom.label;
                            }
                        }
                    );
                    
                    viewer.zoomTo({model: 1});
                    viewer.render();
                });
            });
        });
    </script>
</body>
</html>
''')
        
    print(f"\n[SUCESSO] Viewer 3D interativo gerado: {html_filename}")
    print("Para visualizar contornando as restrições locais de arquivo do navegador, inicie um servidor Python rápido:")
    print("1. Execute: python3 -m http.server 8000")
    print(f"2. Abra no navegador: http://localhost:8000/{html_filename}")

if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Extract Ligand Pharmacophore from a PDBbind Complex.")
    # Dopamine D3 Receptor (3pbl) definido como default, mas permite qualquer ID via linha de comando
    parser.add_argument("pdb_id", type=str, nargs='?', default="3pbl", 
                        help="PDB ID do complexo alvo (ex: 5tvn para Serotonina, 3pbl para Dopamina, 4jwr para Adenosina)")
    
    args = parser.parse_args()
    
    try:
        explain_pharmacophore(args.pdb_id.lower())
    except Exception as e:
        print(f"\n[FALHA] {e}")
        print("Dica: Verifique se o ID existe na sua pasta 'raw/' e se foi incluído no dataset Refined.")
        sys.exit(1)
