"""
simexp/simexp5_noise_robustness.py
---------------------------------------
This script executes the "Orthogonal Sweep" over ambient noise (sigma) while
dynamically optimizing the local bandwidth (K) to balance the bias-variance trade-off.
Generates the publication-ready Factorial Design plot comparing ESCL vs. VDM.
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

from src.models import (
    TwoStageBootstrapTester,
    ESCL_H1_Analyzer,
    VDM_H1_Analyzer,
    auto_select_optimal_k
)

if not hasattr(sys.modules['__main__'], '__spec__'):
    sys.modules['__main__'].__spec__ = None


# ==========================================
# 1. Rigorous Manifold Generators (Uniform Sampling)
# ==========================================
def generate_torus(N=3000):
    """Generates a 4D Clifford Torus embedded in 24D with a strictly uniform volume measure."""
    np.random.seed(42)
    u = np.random.rand(N) * 2 * np.pi
    v = np.random.rand(N) * 2 * np.pi
    X_4d = np.vstack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)]).T
    return np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')


def generate_klein_bottle(N=3000):
    """
    Generates a 24D embedded Klein bottle.
    Rejection sampling ensures absolute uniform density across the non-orientable surface.
    """
    np.random.seed(42)
    R, r = 4.0, 1.0
    max_J = r * (R + r)
    u_list, v_list = [], []
    while len(u_list) < N:
        batch = N - len(u_list)
        u_cand = np.random.rand(batch) * 2 * np.pi
        v_cand = np.random.rand(batch) * 2 * np.pi
        w_cand = np.random.rand(batch) * max_J
        J_val = r * np.sqrt((R + r * np.cos(v_cand)) ** 2 + (r ** 2 / 4.0) * np.sin(v_cand) ** 2)
        mask = w_cand <= J_val
        u_list.extend(u_cand[mask])
        v_list.extend(v_cand[mask])

    u, v = np.array(u_list[:N]), np.array(v_list[:N])
    x1 = (R + r * np.cos(v)) * np.cos(u)
    x2 = (R + r * np.cos(v)) * np.sin(u)
    x3 = r * np.sin(v) * np.cos(u / 2)
    x4 = r * np.sin(v) * np.sin(u / 2)

    X_4d = np.vstack([x1, x2, x3, x4]).T
    return np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')


def apply_noise_and_rotate(X_base, noise_scale, seed):
    """
    Injects isotropic ambient Gaussian noise and applies a random orthogonal
    transformation to simulate high-dimensional measurement perturbation.
    """
    np.random.seed(seed)
    N, D = X_base.shape
    X_noisy = X_base + np.random.normal(0, noise_scale, size=(N, D)) if noise_scale > 0 else X_base
    return np.dot(X_noisy, ortho_group.rvs(D, random_state=seed))


def get_fiedler_ratio(res):
    """Safely extracts the Algebraic Fiedler Ratio to prevent division by zero."""
    if not res: return np.nan
    lam1 = res[0].get('obs_lam1', 0.0)
    lam2 = res[0].get('obs_lam2', 0.0)
    return lam1 / lam2 if (lam2 and lam2 > 1e-7) else np.nan


# ==========================================
# 2. Core Experimental Engine (Orthogonal Noise Sweep)
# ==========================================
def run_noise_robustness_experiment():
    d_prior = 2
    N_points = 3000
    # Linearly sweep the ambient noise variance from 0.0 to 0.35
    noise_levels = np.linspace(0.0, 0.35, 8)

    # Initialize data dictionary for persistent storage
    results = {
        'ESCL_Torus': [], 'VDM_Torus': [],
        'ESCL_Klein': [], 'VDM_Klein': []
    }

    print("\n" + "=" * 100)
    print("🚀 Running Topological Mass Gap & Noise Robustness (Section 5.1)")
    print("=" * 100)

    X_base_torus = generate_torus(N=N_points)
    X_base_klein = generate_klein_bottle(N=N_points)

    for i, noise in enumerate(noise_levels):
        print(f"\n[{'=' * 20} Noise Level: sigma = {noise:.2f} {'=' * 20}]")
        seed = 100 + i
        X_torus = apply_noise_and_rotate(X_base_torus, noise, seed)
        X_klein = apply_noise_and_rotate(X_base_klein, noise, seed + 50)

        # ----------------------------------------------------
        # 1. Evaluate Orientable Torus (Negative Control H0)
        # ----------------------------------------------------
        print("  [Auto-Radar] Detecting optimal intrinsic scale for Torus...")
        k_opt_torus = auto_select_optimal_k(X_torus, k_range=np.arange(15, 80, 5), intrinsic_dim=d_prior,
                                            mutual_knn=False, n_jobs=-1)
        print(f"  -> Torus: Locked K = {k_opt_torus}")

        tester_escl_t = TwoStageBootstrapTester(
            builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_torus, 'mutual_knn': False, 'n_jobs': 4},
            analyzer_class=ESCL_H1_Analyzer,
            analyzer_kwargs={'beta': 15.0, 'auto_beta': True, 'cutoff': 'auto'},
            b_bootstrap=50, max_workers=5
        )
        tester_vdm_t = TwoStageBootstrapTester(
            builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_torus, 'mutual_knn': False, 'n_jobs': 4},
            analyzer_class=VDM_H1_Analyzer,
            analyzer_kwargs={'epsilon_scale': 1.0},
            b_bootstrap=10, max_workers=2  # VDM only requires point estimation; reducing bootstrap load to accelerate
        )

        _, _, res_escl_t = tester_escl_t.test(X_torus)
        _, _, res_vdm_t = tester_vdm_t.test(X_torus)

        results['ESCL_Torus'].append(get_fiedler_ratio(res_escl_t))
        results['VDM_Torus'].append(get_fiedler_ratio(res_vdm_t))

        # ----------------------------------------------------
        # 2. Evaluate Non-orientable Klein Bottle (Positive Control H1)
        # ----------------------------------------------------
        print("  [Auto-Radar] Detecting optimal intrinsic scale for Klein...")
        k_opt_klein = auto_select_optimal_k(X_klein, k_range=np.arange(15, 80, 5), intrinsic_dim=d_prior,
                                            mutual_knn=False, n_jobs=-1)
        print(f"  -> Klein: Locked K = {k_opt_klein}")

        tester_escl_k = TwoStageBootstrapTester(
            builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_klein, 'mutual_knn': False, 'n_jobs': 4},
            analyzer_class=ESCL_H1_Analyzer,
            analyzer_kwargs={'beta': 15.0, 'auto_beta': True, 'cutoff': 'auto'},
            b_bootstrap=50, max_workers=5
        )
        tester_vdm_k = TwoStageBootstrapTester(
            builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_klein, 'mutual_knn': False, 'n_jobs': 4},
            analyzer_class=VDM_H1_Analyzer,
            analyzer_kwargs={'epsilon_scale': 1.0},
            b_bootstrap=10, max_workers=2
        )

        _, _, res_escl_k = tester_escl_k.test(X_klein)
        _, _, res_vdm_k = tester_vdm_k.test(X_klein)

        results['ESCL_Klein'].append(get_fiedler_ratio(res_escl_k))
        results['VDM_Klein'].append(get_fiedler_ratio(res_vdm_k))

        print(
            f"  [Summary] sigma={noise:.2f} | ESCL (H0:{results['ESCL_Torus'][-1]:.3f}, H1:{results['ESCL_Klein'][-1]:.3f}) | VDM (H0:{results['VDM_Torus'][-1]:.3f}, H1:{results['VDM_Klein'][-1]:.3f})")

    # Save data to disk and invoke the visual renderer
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/simexp5_noise_robustness.npz', noises=noise_levels, **results)
    print("\n✅ Data generation complete! Rendering publication-ready plots...")
    plot_noise_robustness(noise_levels, results)


# ==========================================
# 3. Publication-Ready Visual Renderer
# ==========================================
def plot_noise_robustness(noise_levels, results):
    """
    Renders the continuous evolution of the Fiedler ratio across varying ambient noise.
    Demonstrates the exact size control of the ESCL mass gap versus VDM metric leakage.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))

    # VDM (Continuous O(d) Gauge)
    ax.plot(noise_levels, results['VDM_Torus'], color='#E74C3C', linestyle='--', linewidth=3,
            marker='X', markersize=9, label=r'VDM ($H_0$: Torus - Trivial)')
    ax.plot(noise_levels, results['VDM_Klein'], color='#C0392B', linestyle=':', linewidth=3,
            marker='D', markersize=9, label=r'VDM ($H_1$: Klein - Twisted)')

    # ESCL (Discrete Z2 Gauge)
    ax.plot(noise_levels, results['ESCL_Torus'], color='#2ECC71', linestyle='-', linewidth=3.5,
            marker='o', markersize=10, label=r'ESCL ($H_0$: Torus - Trivial)')
    ax.plot(noise_levels, results['ESCL_Klein'], color='#27AE60', linestyle='-.', linewidth=3.5,
            marker='s', markersize=10, label=r'ESCL ($H_1$: Klein - Twisted)')

    # Dynamic Macro Gap Threshold (Section 2.4)
    ax.axhline(y=0.20, color='black', linestyle='-.', linewidth=2, alpha=0.8, label='Macro Gap Threshold (0.20)')

    # Highlight the topological hallucination danger zone
    ax.axhspan(0.20, 1.05, facecolor='#FDEDEC', alpha=0.3, zorder=0)
    ax.text(noise_levels[1], 0.85, 'Topological Hallucination Zone\n(False Positives)',
            color='#C0392B', fontsize=14, fontweight='bold', alpha=0.7)

    # Formatting and labels
    ax.set_xlabel(r'Ambient Gaussian Noise ($\sigma$)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Fiedler Ratio ($\hat{\rho}_n = \hat{\lambda}_1 / \hat{\lambda}_2$)', fontsize=15, fontweight='bold')
    ax.set_title('Noise Robustness & The Topological Mass Gap', fontsize=18, fontweight='bold', pad=15)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(noise_levels[0], noise_levels[-1])
    ax.tick_params(axis='both', which='major', labelsize=13)

    plt.ylim(-0.05, 1.45)

    # Position the legend in the upper right, distributed across two columns
    plt.legend(loc='upper right', ncol=2, framealpha=0.95, fontsize=12, edgecolor='black')

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/simexp5_noise_robustness.pdf', dpi=300, bbox_inches='tight')
    print(" Plot successfully saved to Figs/simexp5_noise_robustness.pdf")
    plt.show()


# Support plotting directly from cached .npz files without recalculation
def reload_and_plot():
    try:
        data = np.load('Results/simexp5_noise_robustness.npz')
        noise_levels = data['noises']
        results = {k: data[k] for k in data.files if k != 'noises'}
        plot_noise_robustness(noise_levels, results)
    except FileNotFoundError:
        print(" Cached data not found. Please execute the computational engine first.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        reload_and_plot()
    else:
        os.makedirs('Results', exist_ok=True)
        t0 = time.time()
        run_noise_robustness_experiment()
        print(f" Experiment 5 (Noise Robustness) completed successfully in {time.time() - t0:.1f}s")