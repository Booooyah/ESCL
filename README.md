# Empirical Spectral Cohomology Learning (ESCL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains the official Python implementation for the manuscript: **"Robust Non-parametric Inference of Manifold Topology via Empirical Spectral Cohomology Learning."**

## 📖 Overview

Inferring macroscopic topological invariants (e.g., orientability) from finite, high-dimensional empirical data is notoriously difficult due to "metric leakage"—the continuous accumulation of local rotational errors that induces false topological hallucinations. 

The **Empirical Spectral Cohomology Learning (ESCL)** framework elevates the continuous $O(d)$ gauge to a discrete $\mathbb{Z}_2$ orientation sheaf. This discrete quantization acts as an algebraic shock absorber, establishing a strict topological mass gap. By integrating a discrete Connection Cheeger Inequality with the BBP phase transition from Random Matrix Theory, ESCL constructs a dimension-invariant 2D Topological Phase Space, providing exact asymptotic size control and optimal statistical power against severe geometric gasification.

## 📂 Repository Structure
```text
.
├── src/
│   └── models.py                   # Core ESCL & VDM algorithms, Phase Space orchestrator
├── utils/
│   └── visualization.py            # Publication-ready plotting & rendering engine
├── simexp/                         # Monte Carlo Simulation Experiments (Section 4)
│   ├── simexp1_torus_24d.py        # Negative Control (H0)
│   ├── simexp2_klein_24d.py        # Positive Control (H1)
│   ├── simexp3_asymmetric_bias.py  # Bias-variance & gap collapse analysis
│   ├── simexp4_scatter.py          # 2D Topological Phase Space visualization
│   ├── simexp5_noise_robustness.py # Orthogonal noise sweep
│   ├── simexp6_ablation_cutoff.py  # Factorial ablation (Surgery vs. Quantization)
│   ├── simexp7_k_sensitivity_torus.py # K-sensitivity analysis (H0)
│   ├── simexp7_k_sensitivity_klein.py # K-sensitivity analysis (H1)
│   └── simexp8_double_klein_4d.py  # High-rank O(4) bundles & N x N vs Nd x Nd profiling
├── realexp/                        # Real-World Empirical Applications (Section 5)
│   ├── exp_utils.py                # Biological preprocessing & spike-binning utilities
│   ├── realexp1_cyclooctane.py     # Molecular phase space & singularity isolation
│   ├── realexp2_natural_image.py   # Unsupervised topological inverse mapping
│   ├── realexp3_grid_cells.py      # Rat grid cells (Nature 2022) under biological noise
│   ├── realexp4_MH.py              # EuRoC MAV trajectory under continuous sensor drift
│   └── plot_fig4_umap.py           # Generates 1x3 composite figure with UMAP
├── data/                           # Directory for empirical datasets (See instructions below)
├── Results/                        # Auto-generated cache for .npz computational results
└── Figs/                           # Auto-generated directory for publication plots (.pdf)

## ⚙️ Installation & Dependencies
This codebase is engineered to run on standard multi-core workstations without requiring GPU acceleration. The underlying thread concurrency is locked to prevent memory explosion during large-scale sparse matrix eigensolving.

Bash
# Clone the repository
git clone [https://github.com/anonymous/ESCL.git](https://github.com/anonymous/ESCL.git)
cd ESCL

# Install required dependencies
pip install numpy scipy scikit-learn pandas matplotlib numba umap-learn h5py

## 📊 Data Availability (Empirical Benchmarks)
To comply with licensing and repository size limits, the raw empirical datasets used in realexp1 to realexp4 are not bundled within this repository. Reviewers and researchers wishing to reproduce the real-world analyses must download the public datasets and place them into the data/ directory.

Cyclooctane Conformational Space (realexp1):

Download pointsCycloOctane.mat from the standard TDA benchmarks (Brown et al., 2008).

Path: data/pointsCycloOctane.mat

Rat Grid Cells (realexp3):

Download the Toroidal Topology Grid Cell Dataset provided by Gardner et al. (Nature, 2022).

Extract the contents directly into the data/ folder.

EuRoC MAV SO(3) Trajectory (realexp4):

Download the MH_01_easy (or equivalent) ground-truth CSV from the EuRoC MAV Dataset.

Rename and place at: data/MH_data.csv

(Note: The Natural Image Patches dataset in realexp2 is generated natively via scipy.datasets.face and requires no external download).

## 🚀 Running the Experiments
All scripts are configured for one-click execution. They will automatically perform the data-driven optimal bandwidth search, execute the MCMC null-bath and node-subsampling routines, save raw data to Results/, and export publication-ready PDFs to Figs/.

Reproducing Simulations (Section 4)
Run any script within the simexp directory from the repository root:

Bash
python simexp/simexp4_scatter.py
python simexp/simexp5_noise_robustness.py
Reproducing Real-World Analyses (Section 5)
Ensure the required datasets are located in data/, then execute:

Bash
python realexp/realexp1_cyclooctane.py
python realexp/realexp2_natural_image.py

## 🎨 Fast Rendering Mode
If you have already executed the computational engines and generated the .npz files in Results/, you can bypass the heavy eigensolver computations and instantly re-render the PDF figures using the --plot-only flag:

Bash
python simexp/simexp4_scatter.py --plot-only

## 📝 Citation & Academic Integrity
If you utilize the ESCL framework in your research, please cite our manuscript.

Note on exp_utils.py: The biological spike-binning and spatial preprocessing routines located in realexp/exp_utils.py are structurally ported from the official MATLAB codebase of Gardner et al. (2022). All scientific credits for these specific preprocessing heuristics belong to the original authors.

📄 License
This project is licensed under the MIT License.
