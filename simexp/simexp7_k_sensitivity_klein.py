"""
simexp/simexp7_k_sensitivity_klein.py
------------------------------------------
Proving that ESCL maintains high statistical power (True Positives)
across a wide range of K, effectively resisting metric leakage.
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


def generate_klein_bottle(N=2000, noise=0.10, seed=42):
    """
    Generates a 24D embedded Klein bottle.
    Utilizes rejection sampling to guarantee an absolutely uniform density
    measure across the surface, eliminating sampling artifacts.
    """
    np.random.seed(seed)
    R, r = 4.0, 1.0
    max_J = r * (R + r)
    u_list, v_list = [], []

    while len(u_list) < N:
        batch = N - len(u_list)
        u_cand = np.random.rand(batch) * 2 * np.pi
        v_cand = np.random.rand(batch) * 2 * np.pi
        w_cand = np.random.rand(batch) * max_J

        # Jacobian determinant for the area element of the Klein bottle
        J_val = r * np.sqrt((R + r * np.cos(v_cand)) ** 2 + (r ** 2 / 4.0) * np.sin(v_cand) ** 2)
        valid = w_cand <= J_val
        u_list.extend(u_cand[valid])
        v_list.extend(v_cand[valid])

    u, v = np.array(u_list[:N]), np.array(v_list[:N])

    # Analytic 4D embedding without self-intersections
    x1 = (R + r * np.cos(v)) * np.cos(u)
    x2 = (R + r * np.cos(v)) * np.sin(u)
    x3 = r * np.sin(v) * np.cos(u / 2)
    x4 = r * np.sin(v) * np.sin(u / 2)

    X_4d = np.vstack([x1, x2, x3, x4]).T
    X_base = np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')

    # Inject isotropic ambient Gaussian noise
    X_noisy = X_base + np.random.normal(0, noise, size=X_base.shape)

    # Apply a random 24D orthogonal rotation
    return np.dot(X_noisy, ortho_group.rvs(24, random_state=seed))


def run_k_sensitivity_klein():
    k_values = np.arange(15, 105, 5) # Sweep K from 15 to 100
    N_points = 2000
    fixed_noise = 0.10
    d_prior = 2

    results = {'escl_rho': [], 'vdm_rho': []}

    print("\n" + "=" * 100)
    print(f"🎛 Running K-Sensitivity on Klein Bottle (Fixed Noise sigma={fixed_noise})")
    print("=" * 100)

    X = generate_klein_bottle(N=N_points, noise=fixed_noise)

    for K in k_values:
        builder = TopologyGraphBuilder(intrinsic_dim=d_prior, k_neighbors=K, mutual_knn=False, n_jobs=-1)
        E, edges, comps = builder.build(X)

        if not comps:
            print(f"K = {K:3d} | Graph Shattered!")
            results['escl_rho'].append(np.nan)
            results['vdm_rho'].append(np.nan)
            continue

        main_comp = max(comps, key=len)
        best_cutoff = auto_select_cutoff(E, edges, target_ratio=0.90)

        # 1. Run ESCL (Evaluate quantization rigidity and topological mass gap)
        escl = ESCL_H1_Analyzer(beta=5.0, auto_beta=True, cutoff=best_cutoff)
        lam1_e, lam2_e, _, _, _ = escl.analyze(E, edges, main_comp, X)
        rho_e = (lam1_e / lam2_e) if (lam2_e and lam2_e > 1e-6) else np.nan

        # 2. Run VDM (Evaluate susceptibility to curvature-induced metric leakage)
        vdm = VDM_H1_Analyzer(epsilon_scale=1.0)
        lam1_v, lam2_v, _, _, _ = vdm.analyze(E, edges, main_comp, X)
        rho_v = (lam1_v / lam2_v) if (lam2_v and lam2_v > 1e-6) else np.nan

        results['escl_rho'].append(rho_e)
        results['vdm_rho'].append(rho_v)

        print(f"K = {K:3d} | ESCL rho: {rho_e:.4f} | VDM rho: {rho_v:.4f}")

    # Save experimental data to disk
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/simexp7_k_sensitivity_klein.npz', k_values=k_values, **results)
    print("\n Complete simulation data saved to Results/simexp7_k_sensitivity_klein.npz")

    plot_k_sensitivity_klein(k_values, results, N_points)


def plot_k_sensitivity_klein(k_values, results, N):
    """Renders the stability of topological discovery (Power) across varying K."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))

    # Dynamic macroscopic threshold based on Cheeger bounds
    macro_threshold = 1.5 / np.log(N)

    # Plot VDM curve
    ax.plot(k_values, results['vdm_rho'], marker='D', markersize=10, linewidth=3,
            color='#C0392B', linestyle='--', label=r'VDM (Continuous O(d) Gauge)')

    # Plot ESCL curve
    ax.plot(k_values, results['escl_rho'], marker='s', markersize=10, linewidth=3.5,
            color='#27AE60', linestyle='-', label=r'ESCL (Discrete Z_2 Gauge)')

    # Macroscopic gap threshold line
    ax.axhline(y=macro_threshold, color='black', linestyle='-.', linewidth=2,
               label=f'Macro Gap Threshold ({macro_threshold:.2f})')

    # Shade the True Positive Zone (Robust Discovery region)
    ax.axhspan(macro_threshold, 1.35, facecolor='#E8F8F5', alpha=0.5, zorder=0)
    ax.text(18, 0.85, 'True Positive Zone\n(Robust Discovery)', color='#27AE60',
            fontsize=14, fontweight='bold', alpha=0.8)

    # Formatting and labels
    ax.set_xlabel(r'Number of Nearest Neighbors ($K$)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Fiedler Ratio ($\rho = \lambda_1 / \lambda_2$)', fontsize=15, fontweight='bold')
    ax.set_title(r'Stability of Topological Discovery ($H_1$: Klein Bottle)', fontsize=17, fontweight='bold', pad=15)

    ax.set_ylim(-0.05, 1.35)
    ax.set_xlim(min(k_values), max(k_values))
    ax.tick_params(axis='both', which='major', labelsize=13)

    # Consolidate and position the legend outside the main plot area
    plt.legend(loc='lower right', ncol=1, framealpha=0.95, fontsize=12, edgecolor='black')

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/simexp7_k_sensitivity_klein.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print(" Plot successfully saved to 'Figs/simexp7_k_sensitivity_klein.pdf'")
    plt.show()


# Support offline plotting directly from cached .npz files
def reload_and_plot():
    try:
        data = np.load('Results/simexp7_k_sensitivity_klein.npz')
        k_values = data['k_values']
        results = {k: data[k] for k in data.files if k != 'k_values'}
        plot_k_sensitivity_klein(k_values, results, N=2000)
    except FileNotFoundError:
        print(" Cached data not found. Please execute the computational engine first.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        reload_and_plot()
    else:
        t0 = time.time()
        run_k_sensitivity_klein()
        print(f"\n Experiment 7 (Klein K-Sensitivity) completed successfully in {time.time() - t0:.1f}s")