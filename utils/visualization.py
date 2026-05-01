"""
utils/visualization.py
----------------------
Publication-ready visualization engine for the ESCL framework.
Handles high-dimensional PCA projections, Z-ordered singularity heatmaps,
and rigorous statistical curve rendering for academic journals.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =============================================================================
# Global Publication-Ready Formatting (JASA / NeurIPS Style)
# =============================================================================
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})


# =============================================================================
# Topology Singularity Heatmaps
# =============================================================================

def _prepare_3d_render_data(X, F_field):
    """Internal helper to handle PCA projection and validity masks."""
    if F_field is None:
        raise ValueError("F_field is None. Ensure ESCL tester returned the scalar field.")

    # 1. Dimensionality Reduction
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X_render = pca.fit_transform(X)
    elif X.shape[1] == 3:
        X_render = X
    elif X.shape[1] == 2:
        X_render = np.hstack([X, np.zeros((X.shape[0], 1))])
    else:
        raise ValueError("Data must have at least 2 dimensions.")

    # 2. Validity Masking
    valid_mask = F_field > 0
    if not np.any(valid_mask):
        valid_mask = np.ones(len(F_field), dtype=bool)

    X_valid = X_render[valid_mask]
    F_valid = F_field[valid_mask]

    # 3. Z-order Fix: Sort by F_field so extreme singularities are drawn last (on top)
    sort_idx = np.argsort(F_valid)
    return X_valid[sort_idx], F_valid[sort_idx]


def plot_singularity_map(X, F_field, title="Topological Singularity Map", cmap='magma_r',
                         point_size=12, elev=30, azim=45, save_name='cyclooctane_heatmap.pdf'):
    """
    Renders the point cloud colored by the Frame Distortion Index (F_i) field.
    Provides a continuous heatmap floating on the manifold.
    """
    X_sorted, F_sorted = _prepare_3d_render_data(X, F_field)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        X_sorted[:, 0], X_sorted[:, 1], X_sorted[:, 2],
        c=F_sorted, cmap=cmap, s=point_size, alpha=0.90, edgecolors='none'
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label(r'Frame Distortion Index $\mathcal{F}_i$', fontsize=14, fontweight='bold')

    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    # Trim margins
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.dist = 8

    os.makedirs('Figs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'Figs/{save_name}', dpi=300, bbox_inches='tight')
    plt.close()


def plot_singularity_map_extreme(X, F_field, title="Topological Singularity Map",
                                 elev=20, azim=-45, save_name='cyclooctane_heatmap_extreme.pdf'):
    """
    Renders the point cloud with a strict bimodal style:
    Ghostly grey bulk vs. Extreme red singularities (Top 5%).
    """
    X_sorted, F_sorted = _prepare_3d_render_data(X, F_field)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Identify the Top 5% extreme cross-caps
    threshold = np.percentile(F_sorted, 95)
    extreme_mask = F_sorted > threshold

    # Plot Background Bulk (Ghostly Grey)
    ax.scatter(X_sorted[:, 0], X_sorted[:, 1], X_sorted[:, 2],
               c='#999999', s=5, alpha=0.4, edgecolors='none')

    # Plot Extreme Singularities (Pure Red)
    ax.scatter(X_sorted[extreme_mask, 0], X_sorted[extreme_mask, 1], X_sorted[extreme_mask, 2],
               c='red', s=30, alpha=1.0, edgecolors='none')

    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.dist = 8

    os.makedirs('Figs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'Figs/{save_name}', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


# =============================================================================
# Statistical Curve Plotting
# =============================================================================

def plot_statistical_curves(torus_data, klein_data, save_path="Figs/simexp1_2_curves.pdf"):
    """
    Generates a 1x2 publication-ready plot comparing ESCL and VDM.
    Panel A: Fiedler Ratio vs Ambient Noise
    Panel B: Rejection Rates (Power & Type I Error)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    c_escl, c_vdm = '#2ca02c', '#d62728'

    # ---------------------------------------------------------
    # Panel A: Fiedler Ratio vs. Noise
    # ---------------------------------------------------------
    ax1 = axes[0]

    # Torus (H0)
    ax1.plot(torus_data['noise'], torus_data['escl_rho'], marker='o', linestyle='-', color=c_escl,
             label='ESCL (Torus - $H_0$)')
    ax1.fill_between(torus_data['noise'], torus_data['escl_ci_lower'], torus_data['escl_ci_upper'], color=c_escl,
                     alpha=0.2)
    ax1.plot(torus_data['noise'], torus_data['vdm_rho'], marker='o', linestyle='-', color=c_vdm,
             label='VDM (Torus - $H_0$)')
    ax1.fill_between(torus_data['noise'], torus_data['vdm_ci_lower'], torus_data['vdm_ci_upper'], color=c_vdm,
                     alpha=0.2)

    # Klein Bottle (H1)
    ax1.plot(klein_data['noise'], klein_data['escl_rho'], marker='s', linestyle='--', color=c_escl,
             label='ESCL (Klein - $H_1$)')
    ax1.fill_between(klein_data['noise'], klein_data['escl_ci_lower'], klein_data['escl_ci_upper'], color=c_escl,
                     alpha=0.2)
    ax1.plot(klein_data['noise'], klein_data['vdm_rho'], marker='s', linestyle='--', color=c_vdm,
             label='VDM (Klein - $H_1$)')
    ax1.fill_between(klein_data['noise'], klein_data['vdm_ci_lower'], klein_data['vdm_ci_upper'], color=c_vdm,
                     alpha=0.2)

    ax1.axhline(y=0.20, color='black', linestyle='-.', linewidth=2, label='Macro Gap Threshold (0.20)')
    ax1.set_xlabel(r'Ambient Gaussian Noise ($\sigma$)')
    ax1.set_ylabel(r'Fiedler Ratio ($\rho = \lambda_1 / \lambda_2$)')
    ax1.set_title('(a) Topological Mass Gap vs. Metric Leakage')
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, alpha=0.4)

    # ---------------------------------------------------------
    # Panel B: Rejection Rate (Power & Type I Error)
    # ---------------------------------------------------------
    ax2 = axes[1]

    ax2.plot(torus_data['noise'], torus_data['escl_reject_rate'], marker='o', linestyle='-', color=c_escl,
             label='ESCL Type I Error')
    ax2.plot(torus_data['noise'], torus_data['vdm_reject_rate'], marker='o', linestyle='-', color=c_vdm,
             label='VDM Type I Error')
    ax2.plot(klein_data['noise'], klein_data['escl_reject_rate'], marker='s', linestyle='--', color=c_escl,
             label='ESCL Power')
    ax2.plot(klein_data['noise'], klein_data['vdm_reject_rate'], marker='s', linestyle='--', color=c_vdm,
             label='VDM Power')

    ax2.axhline(y=0.05, color='black', linestyle=':', linewidth=2, label=r'Nominal $\alpha = 0.05$')
    ax2.set_xlabel(r'Ambient Gaussian Noise ($\sigma$)')
    ax2.set_ylabel('Rejection Rate')
    ax2.set_title('(b) Statistical Power and Type I Error Control')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.4)

    # ---------------------------------------------------------
    # Global Legend Integration
    # ---------------------------------------------------------
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2[-1:], labels1 + labels2[-1:],
               loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, framealpha=0.95)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close()