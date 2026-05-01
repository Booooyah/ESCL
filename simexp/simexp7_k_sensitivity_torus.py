"""
simexp/simexp7_k_sensitivity_torus.py
------------------------------------------
Demonstrating the Bias-Variance Trade-off in tangent space estimation.
VDM fails at small K (noise variance) and large K (curvature bias).
ESCL maintains absolute rigidity across a massive range of K.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

# Lock underlying threads to prevent memory explosion during numpy/scipy multiprocessing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Dynamically append the project root directory to the system path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TopologyGraphBuilder, ESCL_H1_Analyzer, VDM_H1_Analyzer, auto_select_cutoff


def generate_flat_torus(N=2000, noise=0.10, seed=42):
    """
    Generates a 24D embedded Flat Torus with fixed moderate ambient noise.
    Serves as the negative control (H0) to evaluate false positive rates across K.
    """
    np.random.seed(seed)
    u = np.random.rand(N) * 2 * np.pi
    v = np.random.rand(N) * 2 * np.pi
    x1, x2 = np.cos(u), np.sin(u)
    x3, x4 = np.cos(v), np.sin(v)

    X_4d = np.vstack([x1, x2, x3, x4]).T
    X_base = np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')

    # Inject fixed moderate ambient Gaussian noise
    X_noisy = X_base + np.random.normal(0, noise, size=X_base.shape)

    # Apply a random 24D orthogonal rotation
    return np.dot(X_noisy, ortho_group.rvs(24, random_state=seed))


def run_k_sensitivity_experiment():
    k_values = np.arange(10, 105, 5)  # Sweep K from 10 to 100
    N_points = 2000
    fixed_noise = 0.10
    d_prior = 2

    results = {'escl_rho': [], 'vdm_rho': []}

    print("\n" + "=" * 100)
    print(f" Running K-Sensitivity on Torus (Fixed Noise sigma={fixed_noise})")
    print("=" * 100)

    X = generate_flat_torus(N=N_points, noise=fixed_noise)

    for K in k_values:
        builder = TopologyGraphBuilder(intrinsic_dim=d_prior, k_neighbors=K, mutual_knn=False, n_jobs=-1)
        E, edges, comps = builder.build(X)

        if not comps:
            print(f"K = {K:3d} | Graph Shattered!")
            results['escl_rho'].append(np.nan)
            results['vdm_rho'].append(np.nan)
            continue

        main_comp = max(comps, key=len)

        # 1. Dynamic topological surgery threshold
        best_cutoff = auto_select_cutoff(E, edges, target_ratio=0.90)

        # 2. Run ESCL (Extract algebraic eigenvalues only, bypass Bootstrap)
        # Evaluates quantization rigidity against noise variance and curvature bias.
        escl = ESCL_H1_Analyzer(beta=5.0, auto_beta=True, cutoff=best_cutoff)
        lam1_e, lam2_e, _, _, _ = escl.analyze(E, edges, main_comp, X)
        rho_e = (lam1_e / lam2_e) if (lam2_e and lam2_e > 1e-5) else np.nan

        # 3. Run VDM (Extract algebraic eigenvalues only)
        # Evaluates susceptibility to curvature-induced metric leakage.
        vdm = VDM_H1_Analyzer(epsilon_scale=1.0)
        lam1_v, lam2_v, _, _, _ = vdm.analyze(E, edges, main_comp, X)
        rho_v = (lam1_v / lam2_v) if (lam2_v and lam2_v > 1e-5) else np.nan

        results['escl_rho'].append(rho_e)
        results['vdm_rho'].append(rho_v)

        print(f"K = {K:3d} | ESCL rho: {rho_e:.4f} | VDM rho: {rho_v:.4f}")

    # Save experimental data to disk
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/simexp6_k_sensitivity_torus.npz', k_values=k_values, **results)
    print("\n💾 Complete simulation data saved to Results/simexp6_k_sensitivity_torus.npz")

    plot_k_sensitivity(k_values, results, N_points)


def plot_k_sensitivity(k_values, results, N):
    """Renders the bias-variance trade-off curve across varying neighborhood sizes."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))

    # Dynamic macroscopic threshold based on Cheeger bounds
    macro_threshold = 1.5 / np.log(N)

    # Plot VDM curve
    ax.plot(k_values, results['vdm_rho'], marker='^', markersize=10, linewidth=3,
            color='#E74C3C', linestyle='--', label=r'VDM (Continuous O(d) Gauge)')

    # Plot ESCL curve
    ax.plot(k_values, results['escl_rho'], marker='o', markersize=10, linewidth=3.5,
            color='#2ECC71', linestyle='-', label=r'ESCL (Discrete Z_2 Gauge)')

    # Macroscopic threshold line
    ax.axhline(y=macro_threshold, color='black', linestyle='-.', linewidth=2,
               label=f'Macro Gap Threshold ({macro_threshold:.2f})')

    # Annotate extreme regions (Background shading)
    ax.axvspan(10, 25, facecolor='#FDEBD0', alpha=0.5, zorder=0)
    ax.text(17.5, 1.15, 'High Variance\n(Noise Dominated)', color='#D35400',
            fontsize=12, fontweight='bold', ha='center', va='center', rotation=90)

    ax.axvspan(65, 105, facecolor='#E8DAEF', alpha=0.5, zorder=0)
    ax.text(85, 1.15, 'High Bias\n(Curvature Dominated)', color='#8E44AD',
            fontsize=12, fontweight='bold', ha='center', va='center')

    # Highlight the danger zone (Red background)
    ax.axhspan(macro_threshold, 1.35, facecolor='#FDEDEC', alpha=0.3, zorder=0)

    # Formatting and labels
    ax.set_xlabel(r'Number of Nearest Neighbors ($K$)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Fiedler Ratio ($\rho = \lambda_1 / \lambda_2$)', fontsize=15, fontweight='bold')
    ax.set_title(r'Bias-Variance Trade-off ($H_0$: Flat Torus)', fontsize=17, fontweight='bold', pad=15)

    ax.set_ylim(-0.05, 1.35)
    ax.set_xlim(10, max(k_values))
    ax.tick_params(axis='both', which='major', labelsize=13)

    # Consolidate and position the legend outside the main plot area
    plt.legend(loc='upper right', ncol=1, framealpha=0.95, fontsize=12, edgecolor='black')

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/simexp6_k_sensitivity_torus.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print(" Plot successfully saved to 'Figs/simexp6_k_sensitivity_torus.pdf'")
    plt.show()


# Support offline plotting directly from cached .npz files
def reload_and_plot():
    try:
        data = np.load('Results/simexp6_k_sensitivity_torus.npz')
        k_values = data['k_values']
        results = {k: data[k] for k in data.files if k != 'k_values'}
        plot_k_sensitivity(k_values, results, N=2000)
    except FileNotFoundError:
        print(" Cached data not found. Please execute the computational engine first.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        reload_and_plot()
    else:
        t0 = time.time()
        run_k_sensitivity_experiment()
        print(f"\n Torus K-Sensitivity experiment completed successfully in {time.time() - t0:.1f}s")