# PharmXAI-3D: Explainable SciML for Protein-Ligand Binding Affinity

PharmXAI-3D is an advanced Scientific Machine Learning (SciML) pipeline engineered to predict absolute protein-ligand binding affinities ($pK_d$) directly from raw 3D crystallographic coordinates, utilizing SE(3)-equivariant continuous filter convolutions and physics-targeted multi-task regularization.

## Phase 1: Topological Data Extraction & PDBbind Ingestion
- **Pocket Isolation (6Å Cutoff):** To prevent catastrophic memory bottlenecks and noise injection, the receptor extraction pipeline is strictly truncated to a 6Å spatial radius around the ligand, filtering out non-pharmacophoric backbone atoms.
- **Physical Prior Extraction:** A dense, 11-dimensional scalar vector is deterministically computed for every atom prior to GPU serialization. This includes categorical arrays (One-Hot Hybridization [SP, SP2, SP3...]) and raw physical properties (Atomic Number Z, Mass, Covalent Degree, Aromaticity, and Formal Charge).
- **Dependency Hardening:** Overrode the broken C++ `torch_cluster.radius_graph` dependency within Anaconda environments by compiling a mathematically equivalent Euclidean distance matrix calculator using native pure-PyTorch operations (`torch.cdist`).
- **Data Splitting:** Executed hermetic graph segregation. The PDBbind *General Set* acts exclusively as the training distribution, while the rigorous *Refined Set* acts as the untouched validation boundary to prevent data leakage.

## Phase 2: SE(3) Continuous Filter Architecture
- **Categorical Decoupling (`NodeEncoder`):** The raw nuclear identity ($Z$) is stripped and fed into a dynamic `nn.Embedding(100, 32)` layer on the GPU. The remaining 10 deterministic physical priors are concatenated and pushed through a continuous `SiLU` manifold, converging into a robust 64D node state.
- **Continuous Geometric Convolutions (`GaussianSmearing`):** Abandoned naive scalar distances. The network maps Euclidean spatial bonds into a 32-bin Radial Basis Function (RBF). 
- **Message Passing:** Deployed `ContinuousFilterConv` (SchNet topology), where the continuous RBF expansion dynamically gates the message passing between molecular nodes, inherently preserving rotational and translational (SE(3)) invariance.

## Phase 3: SciML Regularization & Convergence
- **Overcoming Catastrophic Overfitting:** Initial architectures memorized the training set (Train Loss: 0.03 / Val Loss: 4.15). The network was fundamentally overhauled using dual-vector regularization.
- **Stochastic Regularization:** Injected strict 20% Dropout pathways throughout the embedding and readout MLPs, combined with an L2 Weight Decay penalty (`1e-4`) in the Adam optimizer to suppress unbounded gradient explosion.
- **Physics-Targeted Auxiliary Loss:** Converted the architecture into a Multi-Task learner. The network predicts $pK_d$ while an auxiliary head simultaneously predicts a global Lennard-Jones steric potential proxy. The true steric proxy is deterministically computed on the CPU using classic $r^{-12}$ and $r^{-6}$ repulsion/attraction geometry. This forces the latent manifold to physically understand spatial crowding, drastically increasing generalizability.
- **Early Stopping Checkpointing:** The learning rate scheduler (`ReduceLROnPlateau`) dynamically monitors the validation state. The pipeline is programmed to halt and serialize the `pharm_model_weights_best.pth` the moment the generalization limit is breached.

## Phase 4: Saliency Mapping & XAI 
- **Input x Gradient Extraction:** Bypassed black-box evaluations by forcing gradients through the static spatial distances (`edge_attr.requires_grad = True`). 
- **Pharmacophore Identification:** Computes the partial derivative $\frac{\partial (pK_d)}{\partial (Dist)}$, quantifying exactly how much the binding affinity shifts upon infinitesimal spatial perturbance of a specific atomic bond.
- **Continuous Gradient Rendering:** Replaced binary thresholds with a continuous mathematical mapping. Node saliency is bucketed into statistical percentiles (Top 1% Magenta, Top 5% Orange, etc.), rendering physical atoms as glowing volumetric spheres scaled by their electrostatic/steric sensitivity.
- **Dependency Hardening (Web Viewer):** Abandoned PyMOL and Anaconda C++ shared library linkages (`libhdf5.so`) in favor of a standalone, web-based interactive HTML viewer powered by `3Dmol.js`.

### XAI Modalities (Dual-Tool Execution)
To answer distinct scientific questions, the XAI layer is bifurcated into two independent extraction tools:

1. **`explain.py` (Pure Ligand Pharmacophore Extractor):**
   - **Goal:** Lead Optimization and Virtual Screening.
   - **Mechanism:** Isolates the mathematical gradient strictly to the ligand's topology. By completely filtering out the receptor's nodes, it highlights exactly which functional groups on the ligand are "untouchable" (Magenta) versus "modifiable scaffold" (Cyan).

2. **`explain_complex.py` (Orthosteric Micro-environment Mapping):**
   - **Goal:** Mechanistic Biology and Pocket Mapping.
   - **Mechanism:** Evaluates the bipartite graph holistically. Unveils the receptor-ligand interaction vectors, revealing the exact receptor atoms (e.g., polarized carbonyls, H-bond donors) that mathematically lock the ligand into the orthosteric binding site.

### CLI Usage (Command Line Interface)
Both extraction tools are engineered as flexible CLI applications, allowing dynamic sub-system targeting on the fly. They feature interactive **Atom Hovering** (displaying Element, Residue, and Chain IDs dynamically).

```bash
# Execute with the default Dopamine D3 Receptor (3pbl)
python3 explain.py

# Inject a specific PDB ID dynamically (e.g., Serotonin 5-HT2B bound to Ergotamine)
python3 explain.py 5tvn

# Map the holistic complex (Receptor + Ligand) for a specific target
python3 explain_complex.py 5tvn
```
*Note: The targeted PDB ID must exist within your local `raw/` directory and be registered in your refined dataset index.*
