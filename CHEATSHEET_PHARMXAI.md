# PharmXAI-3D: Guia Prático e Tabela de Comandos

Este documento serve como a sua *cheatsheet* (folha de dicas) para operar todo o pipeline de triagem virtual de fármacos e visualização na sua arquitetura de inteligência artificial.

## 🧬 Pipeline de Avaliação e Teste

Abaixo está o fluxo completo: desde a preparação da biblioteca até a visualização estrutural do complexo Receptor-Ligante.

| Etapa | Comando | O que ele faz |
| :--- | :--- | :--- |
| **1. Gerar Biblioteca** | `python build_mini_library.py` | Acessa os bancos mundiais (RCSB PDB e PubChem). Faz o download das proteínas brutas, aplica o filtro topológico (remove águas e impurezas) e converte os ligantes alvo para matrizes 3D em `.sdf`. |
| **2. Inferência (Alvo Nativo)** | `python predict_custom.py -r mini_library_gpcr/Opioide_Delta/4N6H_receptor.pdb -l mini_library_gpcr/Opioide_Delta/morphine.sdf` | Executa a predição da sua IA. O script força uma rápida minimização estérico-energética do ligante no bolso e prevê a Afinidade (pKd e Faixa Molar). |
| **3. Inferência (Selectivity)** | `python predict_custom.py -r mini_library_gpcr/Opioide_Delta/4N6H_receptor.pdb -l mini_library_gpcr/Adenosina_A2A/caffeine.sdf` | Submete uma molécula cruzada. O modelo deve prever baixa afinidade e demonstrar seletividade (provando que a rede não alucina o mesmo valor para tudo). |
| **4. Explicação (Saliency)** | `python explain_complex.py 3pbl` | Faz a engenharia reversa do modelo no dataset original. Gera a nuvem Farmacofórica (quais átomos a IA "olhou" para tomar a decisão) e cria um arquivo interativo em `.html`. |

---

## 🔬 Como Visualizar os Resultados Estruturais?

Quando você roda o `predict_custom.py`, o script relaxa o seu fármaco para caber geometricamente na proteína, e salva um arquivo de saída chamado: **`{nome_do_ligante}_minimized.sdf`** na mesma pasta.

Para visualizar o "encaixe" físico validado, você não precisa de códigos complexos, basta abrir ambos os arquivos simultaneamente no seu software de biologia molecular favorito (como o **PyMOL**, **ChimeraX** ou web-viewers).

### Opção A: Usando o PyMOL (Visualização Profissional Local)
1. Abra o PyMOL no seu computador.
2. Arraste para dentro da tela o arquivo do receptor (Ex: `3PBL_receptor.pdb`).
3. Arraste para dentro da tela o arquivo que a rede otimizou (Ex: `haloperidol_minimized.sdf`).
4. **Comandos úteis no PyMOL**:
   * Digite `show sticks, haloperidol_minimized`
   * Digite `color cyan, haloperidol_minimized`
   * Digite `hide lines, 3PBL_receptor` e depois `show cartoon, 3PBL_receptor`

### Opção B: Visualização Online Rápida
Se não tiver o PyMOL instalado ou quiser mostrar rapidamente para sua equipe:
1. Acesse: **[3Dmol.js Viewer](http://3dmol.csb.pitt.edu/viewer.html)**
2. Faça o upload do arquivo do receptor e do `_minimized.sdf`.
3. Pronto! O visualizador construirá o complexo na sua tela.

> **Nota:** Se posteriormente você desejar que o `predict_custom.py` exporte mapas em 3D de relevância atômica nativos (como o `.html` das esferas coloridas que a função `explain` possui), podemos acoplar o módulo `autograd` nele também!
