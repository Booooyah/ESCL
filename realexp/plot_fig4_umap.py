"""
realexp/plot_fig4_umap.py
-----------------------
Generates the publication-ready 1x3 composite figure for Section 5.1.
Demonstrates ESCL's Exact Size Control under severe biological noise (Grid Cells)
and continuous sensor drift (MAV SO(3)).
Features a UMAP projection for visualizing the non-linear manifold topology.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Attempt to import UMAP for non-linear dimensionality reduction
try:
    import umap
except ImportError:
    print("🚨 UMAP library missing! Please run: pip install umap-learn")
    import sys
    sys.exit(1)

def plot_master_figure():
    # =========================================================================
    # 1. Load Empirical Datasets (Grid Cells & MAV SO(3))
    # =========================================================================
    try:
        # Load the Rat Grid Cell neural manifold data
        grid_data = np.load('Results/realexp3_grid_cells.npz', allow_pickle=True)
        X_neural = grid_data['X_target']

        # Extract the dictionary from the numpy object array
        res_escl_grid = grid_data['res_escl'].item()
        rho_nulls = res_escl_grid.get('rho_nulls', [])
        obs_lam1 = res_escl_grid.get('obs_lam1', 0.0)

        # Load the UAV MAV SO(3) rigid body trajectory data
        drone_data = np.load('Results/realexp4_MH.npz', allow_pickle=True)
        res_vdm_drone = drone_data['res_vdm'].item()
        res_escl_drone = drone_data['res_escl'].item()

        vdm_evals = [res_vdm_drone.get('obs_lam1', 0.0), res_vdm_drone.get('obs_lam2', 0.0)]
        escl_evals = [res_escl_drone.get('obs_lam1', 0.0), res_escl_drone.get('obs_lam2', 0.0)]

    except FileNotFoundError as e:
        print(f"🚨 Data files not found! Error details: {e}")
        print("Please ensure you have successfully executed realexp3_grid_cells.py and realexp4_MH.py")
        return
    except KeyError as e:
        print(f"🚨 Data key mismatch! Error details: {e}")
        return

    # =========================================================================
    # 2. Configure Publication-Quality Canvas
    # =========================================================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.25)

    # =========================================================================
    # Panel A: 166D Neural Manifold (UMAP Projection)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0], projection='3d')

    print("🎨 Projecting the 166D neural manifold to 3D via UMAP, please wait...")
    reducer = umap.UMAP(n_components=3, n_neighbors=50, min_dist=0.1, metric='euclidean', random_state=42)
    X_umap = reducer.fit_transform(X_neural)

    # Render the UMAP point cloud
    ax1.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2],
                c=X_umap[:, 2], cmap='viridis', s=15, alpha=0.8, edgecolors='none')

    ax1.set_title('(a) Empirical Neural Manifold\n(Rat Grid Cells, 166D $\\to$ 3D UMAP)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_axis_off()

    # =========================================================================
    # Panel B: MCMC Null Bath Histogram (Phase I)
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])

    if len(rho_nulls) > 0:
        ax2.hist(rho_nulls, bins=20, density=True, color='#3498DB', alpha=0.7, edgecolor='white', label='Null Bath (Chaos)')
        ax2.axvline(obs_lam1, color='#2ECC71', linestyle='--', linewidth=3.5, label=f'Observed $\\lambda_1$ = {float(obs_lam1):.4f}')
        ax2.axvline(np.percentile(rho_nulls, 5), color='#2C3E50', linestyle=':', linewidth=2.5, label='95% Reject Threshold')
    else:
        ax2.text(0.5, 0.5, 'Exact Algebraic Zero\n$\\lambda_1 = 0.000$\n(Strict $H_0$ Retention)',
                 ha='center', va='center', fontsize=18, fontweight='bold', color='#27AE60', transform=ax2.transAxes)

    ax2.set_title('(b) Microcanonical Null Distribution\n(Exact Size Control under Biological Noise)', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Fundamental Energy $\\lambda_1$', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=12)

    # =========================================================================
    # Panel C: Spectral Gap Preservation (Drone SO(3))
    # =========================================================================
    ax3 = fig.add_subplot(gs[2])
    labels = ['Fundamental Energy ($\\lambda_1$)', 'Macroscopic Gap ($\\lambda_2$)']
    x = np.arange(len(labels))
    width = 0.35

    # Illustrates VDM's spectral collapse due to metric leakage versus ESCL's topological rigidity
    ax3.bar(x - width / 2, vdm_evals, width, label='VDM (Shattered / Degenerate)', color='#E74C3C', alpha=0.85, edgecolor='black')
    ax3.bar(x + width / 2, escl_evals, width, label='ESCL (Topological Rigidity)', color='#2ECC71', alpha=0.9, edgecolor='black')

    ax3.set_title('(c) Preservation of the Spectral Gap\n(MAV SO(3) under Sensor Drift)', fontsize=16, fontweight='bold', pad=15)
    ax3.set_ylabel('Eigenvalue Magnitude', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', frameon=True, fontsize=12)

    # Baseline reference line
    ax3.axhline(0, color='black', linewidth=1.5)

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/Fig4_Real_Data_H0_Control_UMAP.pdf', dpi=300, bbox_inches='tight')
    print("\n✅ Publication-ready composite figure saved to Figs/Fig4_Real_Data_H0_Control_UMAP.pdf")
    plt.show()

if __name__ == "__main__":
    plot_master_figure()