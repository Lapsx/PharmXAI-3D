# 3D-XAI Pharmacophore Platform (PharmXAI-3D)

## Project Overview

A generalizable Scientific Machine Learning (SciML) pipeline that ingests any receptor-ligand structural dataset (e.g., PDBbind, ChEMBL) to predict binding affinity metrics (Ki, Kd, IC50) while automatically extracting the core structural **Pharmacophore** and identifying green-light variable **Bioisostere** substitution zones.

## Core Architecture

- **Model:** 3D Rotationally/Translationally Invariant Graph Neural Networks (SchNet or DimeNet) using continuous 3D coordinate message passing.
- **Graph Structure:** Heterogeneous Bipartite Graph (HeteroData in PyG) isolating receptor residue nodes within a strict 0.6 nm radius of the ligand.
- **Physics Regularization:** Structural penalties injected into the Loss layer (Lennard-Jones atomic exclusions and covalent bond torsion strain via RDKit).
- **Explainable AI (XAI):** GNNExplainer and Captum gradients mapped to an RGB scale (Red=Pharmacophore, Green=Substitution-allowed) on the 3D graph, rendered interactively via py3Dmol/Streamlit-3dmol.
- **Validation:** Scaffold Split (Leave-One-Cluster-Out) to rigorously validate pharmacophore deduction on unseen chemical scaffolds based purely on protein pocket geometry.

## Goals
- [x] Data Pipeline (Data Ingestion): Build python script that reads the protein file and a ligand, cuts the protein receptor pocket (0.6nm cutoff) and converts it to an PyTorch Geometric HeteroData objetct.
- [] The GNN architecture: Build the geometric neural network itself. 
- [] Trainign loop and physics: Trains the model to predict binding affinity, adding physics based penalties (Lennard-Jones) in the Loss function so that the learned geometries are physical consistent.
- [] XAI layer: Apply GNNExplainer to label the molecule, revealing what parts consists of the pharmacophore itself and what parts can be modified, and render that in 3D.

