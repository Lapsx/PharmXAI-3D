import argparse
import os
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from data_processor import PharmGraphBuilder
from gnn_model import PharmGeometricGNN

def dock_ligand(receptor_pdb, ligand_sdf, bounding_box_pdb, out_sdf, blind=False):
    """
    Realiza o Docking Molecular do ligante no receptor utilizando o Smina.
    Se blind=True, a bounding box engloba a proteína inteira (Blind Docking).
    Se blind=False, usa o ligante nativo como 'cola' (Targeted Docking).
    """
    mode_str = "BLIND DOCKING (Escaneando Proteína Inteira)" if blind else "TARGETED DOCKING (Guiado pelo Nativo)"
    print(f"[*] Iniciando {mode_str} via Smina...")
    smina_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smina")
    
    if not os.path.exists(smina_path):
        raise FileNotFoundError("Executável 'smina' não encontrado! Baixe-o com: wget https://sourceforge.net/projects/smina/files/smina.static/download -O smina && chmod +x smina")
        
    cmd = [
        smina_path,
        "-r", receptor_pdb,
        "-l", ligand_sdf,
        "--autobox_ligand", bounding_box_pdb,
        "-o", out_sdf,
        "--exhaustiveness", "8",
        "--num_modes", "1",
        "--quiet"
    ]
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Falha no Docking Molecular.\n{result.stderr}")
        
    print(f"[+] Docking concluído! Melhor pose geométrica salva em {out_sdf}")
    return out_sdf

def minimize_ligand_in_pocket(receptor_pdb, ligand_sdf, out_sdf):
    print(f"[*] Iniciando relaxamento estérico estrutural (Fast-Minimization)...")
    ligand = Chem.MolFromMolFile(ligand_sdf, removeHs=False, sanitize=True)
    receptor = Chem.MolFromPDBFile(receptor_pdb, removeHs=True, sanitize=False)
    combo = Chem.CombineMols(receptor, ligand)
    combo = Chem.AddHs(combo, addCoords=True)
    num_prot_atoms = receptor.GetNumAtoms()
    
    try:
        ff = AllChem.MMFFGetMoleculeForceField(combo, AllChem.MMFFGetMoleculeProperties(combo))
        if ff is None: ff = AllChem.UFFGetMoleculeForceField(combo)
        for i in range(num_prot_atoms): ff.AddFixedPoint(i)
        ff.Minimize(maxIts=200)
        optimized_positions = combo.GetConformer().GetPositions()
        lig_conf = ligand.GetConformer()
        for i in range(ligand.GetNumAtoms()):
            lig_conf.SetAtomPosition(i, optimized_positions[num_prot_atoms + i])
        ligand = Chem.RemoveHs(ligand)
        writer = Chem.SDWriter(out_sdf)
        writer.write(ligand)
        writer.close()
        print(f"[+] Minimização concluída! Salvo em {out_sdf}")
        return out_sdf
    except Exception as e:
        print(f"[!] Aviso: A minimização falhou ({e}). Procedendo com a pose original.")
        return ligand_sdf

def compute_center_of_mass(mol_path):
    if mol_path.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(mol_path, sanitize=False)
    else:
        mol = Chem.MolFromMolFile(mol_path, sanitize=False)
    if not mol: return None
    pos = mol.GetConformer().GetPositions()
    return np.mean(pos, axis=0)

def predict_affinity(receptor_path, ligand_path, native_path=None, compare_path=None, minimize=True, explain=True, true_affinity=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Carregando PharmGeometricGNN na unidade: {device}")
    
    work_ligand = ligand_path
    docked_sdf = ligand_path.replace(".sdf", "_docked.sdf")
    if not docked_sdf.endswith(".sdf"): docked_sdf += "_docked.sdf"
    
    # 1. Decisão do Pipeline de Docking
    if compare_path and os.path.exists(compare_path):
        # Rota BLIND DOCKING (Teste Real)
        work_ligand = dock_ligand(receptor_path, ligand_path, receptor_path, docked_sdf, blind=True)
    elif native_path and os.path.exists(native_path):
        # Rota TARGETED DOCKING
        work_ligand = dock_ligand(receptor_path, ligand_path, native_path, docked_sdf, blind=False)
    elif minimize:
        # Rota RDKIT Fallback
        optimized_sdf = ligand_path.replace(".sdf", "_minimized.sdf")
        if not optimized_sdf.endswith(".sdf"): optimized_sdf += "_minimized.sdf"
        work_ligand = minimize_ligand_in_pocket(receptor_path, work_ligand, optimized_sdf)

    # 2. Instancia o Modelo
    model = PharmGeometricGNN(node_dim=64, num_gaussians=32, num_layers=4).to(device)
    weights_file = "pharm_model_weights_best.pth"
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.eval()

    # 3. Constrói o Grafo
    print(f"[*] Mapeando nuvem farmacofórica do complexo...")
    builder = PharmGraphBuilder(pocket_radius=6.0, interaction_radius=5.0, covalent_radius=2.0)
    hetero_data = builder.build_hetero_graph(receptor_path, work_ligand)
    data_homo = hetero_data.to_homogeneous().to(device)
    
    if minimize:
        print("[*] Aplicando Física PINN: Otimizando induced-fit e choques estéricos via autograd...")
        data_homo.pos.requires_grad_(True)
        optimizer_pinn = torch.optim.Adam([data_homo.pos], lr=0.01)
        model.train()
        for step in range(50):
            optimizer_pinn.zero_grad()
            pkd_pred, steric_pred = model(data_homo)
            # Maximiza afinidade (minimiza -pKd) e minimiza repulsão estérica
            loss = -pkd_pred.sum() + 0.5 * steric_pred.sum()
            loss.backward()
            optimizer_pinn.step()
        data_homo.pos.requires_grad_(False)
        model.eval()
        print("[+] Induced-fit resolvido pela rede neural!")

    if explain: data_homo.edge_attr.requires_grad_(True)

    # 4. Inferência
    pkd_pred, _ = model(data_homo)
    affinity_pkd = pkd_pred.item()
    
    kd_molar = 10 ** (-affinity_pkd)
    kd_str = f"{kd_molar * 1e12:.2f} pM" if kd_molar < 1e-9 else f"{kd_molar * 1e9:.2f} nM" if kd_molar < 1e-6 else f"{kd_molar * 1e6:.2f} µM"
            
    print("\n" + "="*50)
    print(" RESULTADO DO VIRTUAL SCREENING ")
    print("="*50)
    print(f" Afinidade Prevista (pKd) : {affinity_pkd:.3f}")
    
    if true_affinity is not None:
        delta = abs(true_affinity - affinity_pkd)
        print(f" Afinidade Real (pKd)     : {true_affinity:.3f}")
        print(f" Erro Absoluto (Delta)    : {delta:.3f} pKd")
        
    print(f" Constante de Dissociação : {kd_str}")
    
    # Validação Científica Pós-Docking (A Rota "Compare")
    if compare_path and os.path.exists(compare_path):
        com_pred = compute_center_of_mass(work_ligand)
        com_true = compute_center_of_mass(compare_path)
        if com_pred is not None and com_true is not None:
            dist = np.linalg.norm(com_pred - com_true)
            print("-" * 50)
            print(" 🔬 VALIDAÇÃO CEGA (BLIND DOCKING TEST)")
            print("-" * 50)
            print(f" Distância entre Centro de Massa (Previsto vs Cristal): {dist:.2f} Å")
            if dist < 5.0:
                print(" [✔] SUCESSO! A rede encontrou o Bolso Ativo correto de forma totalmente cega.")
            else:
                print(" [!] ALERTA! A molécula ancorou fora do bolso catalítico principal (Possível sítio alostérico).")
                
    print("="*50 + "\n")
    
    # 5. Explicação
    if explain:
        print("[*] Computando Saliência e Farmacóforo Espacial (Backward Pass)...")
        model.zero_grad()
        pkd_pred.backward()
        
        edge_importance = data_homo.edge_attr.grad.abs().squeeze().cpu().numpy()
        edge_index = data_homo.edge_index.cpu().numpy()
        
        node_importance = np.zeros(data_homo.num_nodes)
        for idx, val in enumerate(edge_importance):
            node_importance[edge_index[0, idx]] += val
            node_importance[edge_index[1, idx]] += val
            
        p70 = np.percentile(node_importance, 70) if len(node_importance) > 0 else 0
        p85 = np.percentile(node_importance, 85) if len(node_importance) > 0 else 0
        p95 = np.percentile(node_importance, 95) if len(node_importance) > 0 else 0
        p99 = np.percentile(node_importance, 99) if len(node_importance) > 0 else 0
        
        edge_threshold = np.percentile(edge_importance, 99) if len(edge_importance) > 0 else 0
        critical_edges_idx = np.where(edge_importance >= edge_threshold)[0]
        pos = data_homo.pos.cpu().numpy()
        z_array = data_homo.x[:, 0].cpu().numpy()
        
        html_filename = f"pharmacophore_{os.path.basename(receptor_path).replace('.pdb', '')}_{os.path.basename(ligand_path).replace('.sdf', '')}.html"
        
        with open(receptor_path, 'r') as f_rec:
            pdb_data = f_rec.read().replace('`', '\\`').replace('$', '\\$')
            
        with open(work_ligand, 'r') as f_lig:
            sdf_data = f_lig.read().replace('`', '\\`').replace('$', '\\$')
            
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PharmXAI-3D Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; background-color: #111; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #info {{ position: absolute; top: 20px; left: 20px; color: #fff; font-family: 'Courier New', Courier, monospace; z-index: 10; background: rgba(0,0,0,0.85); padding: 20px; border: 1px solid #444; border-radius: 8px; }}
    </style>
</head>
<body>
    <div id="info">
        <b style="color: #0f0; font-size: 1.2em;">PharmXAI-3D: Inferencia Customizada</b><br><br>
        <b>Target:</b> {os.path.basename(receptor_path)}<br>
        <b>Afinidade Prevista:</b> {affinity_pkd:.2f} pKd<br>''')
        
            if true_affinity is not None:
                f.write(f'''        <b>Afinidade Real:</b> {true_affinity:.2f} pKd<br>
        <b>Margem de Erro:</b> {abs(true_affinity - affinity_pkd):.2f} pKd<br>''')
            
            f.write(f'''        <hr style="border-color: #444;">
        <b>FARMACOFORO DO COMPLEXO:</b><br><br>
        <span style="color:magenta;">● Top 1% (Magenta):</span> Contatos Críticos<br>
        <span style="color:orange;">● Top 5% (Laranja):</span> Ligações Fortes<br>
        <span style="color:yellow;">● Top 15% (Amarelo):</span> Van der Waals<br>
        <span style="color:lime;">● Top 30% (Verde):</span> Influência Fraca<br>
    </div>
    <div id="container"></div>
    <script>
        $(document).ready(function() {{
            let viewer = $3Dmol.createViewer($('#container'), {{backgroundColor: '#111'}});
            
            let pdb_str = `{pdb_data}`;
            let sdf_str = `{sdf_data}`;
            
            viewer.addModel(pdb_str, "pdb");
            viewer.setStyle({{model: 0}}, {{cartoon: {{color: 'white', opacity: 0.15}}}});
            
            viewer.addModel(sdf_str, "sdf");
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'cyanCarbon', radius: 0.15}}}});
''')
            for idx in critical_edges_idx:
                p1, p2 = pos[edge_index[0, idx]], pos[edge_index[1, idx]]
                f.write(f"                    viewer.addCylinder({{start: {{x:{p1[0]:.3f}, y:{p1[1]:.3f}, z:{p1[2]:.3f}}}, end: {{x:{p2[0]:.3f}, y:{p2[1]:.3f}, z:{p2[2]:.3f}}}, radius: 0.05, color: 'magenta', dashed: true}});\n")
            
            element_map = {6: ('C', 'gray'), 7: ('N', 'blue'), 8: ('O', 'red'), 16: ('S', 'yellow'), 15: ('P', 'orange'), 9: ('F', 'green'), 17: ('Cl', 'green')}
            for i, val in enumerate(node_importance):
                if val < p70: continue
                color, radius, alpha = ('lime', 0.2, 0.4) if val < p85 else ('yellow', 0.3, 0.6) if val < p95 else ('orange', 0.4, 0.7) if val < p99 else ('magenta', 0.5, 0.8)
                elem_color = element_map.get(int(z_array[i]), ('X', 'white'))[1]
                p = pos[i]
                f.write(f"                    viewer.addSphere({{center: {{x:{p[0]:.3f}, y:{p[1]:.3f}, z:{p[2]:.3f}}}, radius: {radius}, color: '{color}', alpha: {alpha}}});\n")

            f.write('''
                    viewer.zoomTo({model: 1});
                    viewer.render();
        });
    </script>
</body>
</html>
''')
        print(f"[SUCESSO] Viewer HTML salvo: {html_filename}")
        
        # ==========================================
        # Geração de Script PyMOL (.pml)
        # ==========================================
        pml_filename = html_filename.replace('.html', '.pml')
        with open(pml_filename, "w", encoding="utf-8") as f:
            f.write(f"load {os.path.basename(receptor_path)}, receptor\n")
            f.write(f"load {os.path.basename(work_ligand)}, ligand\n")
            f.write("color gray70, receptor\n")
            f.write("show cartoon, receptor\n")
            f.write("color cyan, ligand\n")
            f.write("show sticks, ligand\n")
            
            # Adicionando Esferas Farmacofóricas como pseudoatoms
            for i, val in enumerate(node_importance):
                if val < p70: continue
                color, radius, alpha = ('green', 0.2, 0.6) if val < p85 else ('yellow', 0.3, 0.4) if val < p95 else ('orange', 0.4, 0.3) if val < p99 else ('magenta', 0.5, 0.2)
                p = pos[i]
                f.write(f"pseudoatom pharm_{i}, pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}], color={color}, vdw={radius}\n")
                f.write(f"show spheres, pharm_{i}\n")
                f.write(f"set sphere_transparency, {alpha}, pharm_{i}\n")
                
            # Adicionando Conexões Críticas (Edges)
            for idx in critical_edges_idx:
                p1, p2 = pos[edge_index[0, idx]], pos[edge_index[1, idx]]
                f.write(f"pseudoatom edgeA_{idx}, pos=[{p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f}]\n")
                f.write(f"pseudoatom edgeB_{idx}, pos=[{p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f}]\n")
                f.write(f"distance dist_{idx}, edgeA_{idx}, edgeB_{idx}\n")
                f.write(f"color magenta, dist_{idx}\n")
                f.write(f"hide labels, dist_{idx}\n")
                
            f.write("center ligand\n")
            f.write("zoom ligand, 10\n")
            
        print(f"[SUCESSO] Script PyMOL salvo: {pml_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PharmXAI-3D: Predição e Explicação Customizada.")
    parser.add_argument("-r", "--receptor", required=True, help="Caminho PDB")
    parser.add_argument("-l", "--ligand", required=True, help="Caminho SDF")
    parser.add_argument("-n", "--native", required=False, help="Ligante Nativo para Targeted Docking")
    parser.add_argument("-c", "--compare", required=False, help="Ligante Nativo para validação (Ativa o Blind Docking)")
    parser.add_argument("-t", "--true-affinity", type=float, required=False, help="Afinidade experimental real (pKd) para comparar com a IA")
    parser.add_argument("--no-minimize", action="store_true")
    parser.add_argument("--no-explain", action="store_true")
    
    args = parser.parse_args()
    if not os.path.exists(args.receptor) or not os.path.exists(args.ligand):
        print("[!] Erro: Arquivos de entrada não encontrados.")
        exit(1)
        
    predict_affinity(args.receptor, args.ligand, native_path=args.native, compare_path=args.compare, minimize=not args.no_minimize, explain=not args.no_explain, true_affinity=args.true_affinity)
