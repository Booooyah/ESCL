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
