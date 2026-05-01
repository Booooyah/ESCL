"""
simexp/simexp4_scatter.py
------------------------------------
Topological Phase Space Scatter Plot
Visualizing the continuous evolution of manifolds under extreme high-dimensional chaos.
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
    auto_select_optimal_k
)

if not hasattr(sys.modules['__main__'], '__spec__'):
    sys.modules['__main__'].__spec__ = None


# ==========================================
# 1. Rigorous Manifold Generators
# ==========================================
def generate_torus(N=3000):
    """
    Generates a 4D Clifford Torus.
    A flat 4D embedding is strictly required to guarantee absolute geometric symmetry,
    preventing asymmetric self-intersections (short-circuits) under severe ambient noise.
    """
    np.random.seed(42)
    u = np.random.rand(N) * 2 * np.pi
    v = np.random.rand(N) * 2 * np.pi
    X_4d = np.vstack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)]).T
    return np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')


def generate_klein_bottle(N=3000):
    """
    Generates a 24D embedded Klein bottle.
    Rejection sampling ensures a strictly uniform volume measure to prevent density artifacts.
    """
    np.random.seed(42)
    R, r = 4.0, 1.0
    max_J = r * (R + r)
    u_list, v_list = [], []
    while len(u_list) < N:
        batch_size = N - len(u_list)
        u_cand = np.random.rand(batch_size) * 2 * np.pi
        v_cand = np.random.rand(batch_size) * 2 * np.pi
        w_cand = np.random.rand(batch_size) * max_J
        J_val = r * np.sqrt((R + r * np.cos(v_cand)) ** 2 + (r ** 2 / 4.0) * np.sin(v_cand) ** 2)
        valid_mask = w_cand <= J_val
        u_list.extend(u_cand[valid_mask])
        v_list.extend(v_cand[valid_mask])

    u = np.array(u_list[:N])
    v = np.array(v_list[:N])

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


# ==========================================
# 2. Core Experimental Engine
# ==========================================
def run_scatter_experiment():
    FAST_MODE = False  # Set to True for a quick preview, False for the full publication sweep.

    if FAST_MODE:
        print("⚠️ [WARNING] Running in FAST_MODE (Preview)")
        noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_trials = 3
        b_boot_escl = 20
        k_search_range = np.arange(40, 70, 15)
    else:
        print("✅ [INFO] Running FULL Scatter Experiment")
        # Sweep ambient noise up to 0.80 to ensure observation of the theoretical gasification limit
        noise_levels = np.linspace(0.10, 0.80, 8)
        n_trials = 8
        b_boot_escl = 50
        k_search_range = np.arange(30, 90, 10)

    N_points = 3000
    d_prior = 2

    raw_scatter_data = {'noise': [], 'manifold': [], 'chaos': [], 'fiedler': [], 'p_noise': []}

    print("\n" + "=" * 100)
    print("🌌 Scanning Topological Phase Space: Torus vs. Klein Bottle")
    print("=" * 100)

    X_base_torus = generate_torus(N_points)
    X_base_klein = generate_klein_bottle(N_points)

    for noise in noise_levels:
        print(f"\n[{'=' * 30} Noise Level: sigma = {noise:.2f} {'=' * 30}]")

        # Probe the optimal intrinsic scale for the current noise level
        dummy_X_t = apply_noise_and_rotate(X_base_torus, noise, seed=999)
        k_opt_t = auto_select_optimal_k(dummy_X_t, k_range=k_search_range, intrinsic_dim=d_prior, n_jobs=-1)

        dummy_X_k = apply_noise_and_rotate(X_base_klein, noise, seed=888)
        k_opt_k = auto_select_optimal_k(dummy_X_k, k_range=k_search_range, intrinsic_dim=d_prior, n_jobs=-1)

        print(f"  -> Locked K: Torus={k_opt_t}, Klein={k_opt_k}")

        # Initialize the dual-protection ESCL framework (Dynamic Cutoff + MCMC)
        tester_escl_t = TwoStageBootstrapTester(
            builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_t, 'n_jobs': 4},
            analyzer_class=ESCL_H1_Analyzer,
            analyzer_kwargs={'beta': 5.0, 'auto_beta': True, 'cutoff': 'auto'},
            b_bootstrap=b_boot_escl, max_workers=5, alpha_mcmc=0.05, solver_tol=1e-5
        )

        tester_escl_k = TwoStageBootstrapTester(
            builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_k, 'n_jobs': 4},
            analyzer_class=ESCL_H1_Analyzer,
            analyzer_kwargs={'beta': 5.0, 'auto_beta': True, 'cutoff': 'auto'},
            b_bootstrap=b_boot_escl, max_workers=5, alpha_mcmc=0.05, solver_tol=1e-5
        )

        t0 = time.time()
        for trial in range(n_trials):
            X_t = apply_noise_and_rotate(X_base_torus, noise, int(noise * 1000) + trial)
            X_k = apply_noise_and_rotate(X_base_klein, noise, int(noise * 1000) + trial + 5000)

            _, _, res_escl_t = tester_escl_t.test(X_t)
            _, _, res_escl_k = tester_escl_k.test(X_k)

            raw_scatter_data['noise'].extend([noise, noise])
            raw_scatter_data['manifold'].extend(['Torus', 'Klein'])
            raw_scatter_data['chaos'].extend([res_escl_t[0].get('chaos_ci_median', 1.0), res_escl_k[0].get('chaos_ci_median', 1.0)])
            raw_scatter_data['fiedler'].extend([res_escl_t[0].get('fiedler_ci_lower', 0.0), res_escl_k[0].get('fiedler_ci_lower', 0.0)])
            raw_scatter_data['p_noise'].extend([res_escl_t[0].get('p_noise', 1.0), res_escl_k[0].get('p_noise', 1.0)])

            sys.stdout.write(f"\r  Trial {trial + 1}/{n_trials} done.")
            sys.stdout.flush()

        print(f" (Elapsed: {time.time() - t0:.1f}s)")

    # Save experimental data to disk
    os.makedirs('Results', exist_ok=True)
    np.savez('Results/simexp4_scatter.npz', **raw_scatter_data)
    print("\n Data generation complete! Rendering publication plot...")
    plot_phase_space_scatter()


# ==========================================
# 3. Phase Space Scatter Plot Renderer
# ==========================================
def plot_phase_space_scatter():
    """
    Renders the 2D Topological Phase Space.
    Maps the Dimension-Invariant Chaos CI (X-axis) against the Algebraic Fiedler CI (Y-axis).
    """
    try:
        data = np.load('Results/simexp4_scatter.npz')
        noises, manifolds, chaos, fiedler, p_noise = data['noise'], data['manifold'], data['chaos'], data['fiedler'], data['p_noise']
    except FileNotFoundError: return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Expand canvas limits to explicitly visualize degenerate failures at (1.0, 0.0)
    x_min, x_max = -0.01, 1.05
    y_min, y_max = -0.02, 1.05
    chaos_thresh, fiedler_thresh = 0.25, 0.50

    # Fill phase regions corresponding to the dimension-invariant decision tree (Section 2.4)
    ax.axhspan(fiedler_thresh, y_max, facecolor='#27AE60', alpha=0.15, label='Twisted (Macroscopic)')
    ax.fill_between([chaos_thresh, x_max], y_min, fiedler_thresh, facecolor='#E74C3C', alpha=0.10, label='Degenerate (Fail-Safe)')
    ax.fill_between([x_min, chaos_thresh], y_min, fiedler_thresh, facecolor='#F1C40F', alpha=0.10, label='Trivial / Localized')

    # Apply minor jitter to prevent marker occlusion
    jit_x = lambda arr: arr + np.random.normal(0, 0.005, size=len(arr))
    jit_y = lambda arr: arr + np.random.normal(0, 0.005, size=len(arr))

    # Isolate manifolds intercepted by the Phase I MCMC Null test or the structural cutoff
    # Note: Degenerate states natively return p_noise = 1.0
    idx_fail = (p_noise > 0.05)
    idx_t_pass = (manifolds == 'Torus') & ~idx_fail
    idx_k_pass = (manifolds == 'Klein') & ~idx_fail

    sc_t = ax.scatter(jit_x(chaos[idx_t_pass]), jit_y(fiedler[idx_t_pass]), c=noises[idx_t_pass], cmap='Blues', marker='o', edgecolor='k', s=90, alpha=0.7, label='Sim: Torus ($H_0$)')
    sc_k = ax.scatter(jit_x(chaos[idx_k_pass]), jit_y(fiedler[idx_k_pass]), c=noises[idx_k_pass], cmap='Purples', marker='X', edgecolor='k', s=110, alpha=0.7, label='Sim: Klein ($H_1$)')

    # Plot the degenerate fail-safe clusters
    if np.any(idx_fail):
        ax.scatter(jit_x(chaos[idx_fail]), jit_y(fiedler[idx_fail]), c='#FF0000', marker='o', edgecolor='black', linewidth=1.5, s=120, alpha=0.85, label='Phase I / Cutoff Rejected', zorder=8)

    # Overlay empirical datasets (Real-world scientific benchmarks)
    empirical_data = [
        ("Emp: UAV (1D Trajectory)", 0.000, 0.000, 0.000, 0.000, 0.000, 'P', '#00FFFF'),
        ("Emp: Cyclooctane", 0.006, 0.171, 0.001, 0.001, 0.049, 's', '#F1C40F'),
        ("Emp: Grid Cells ($H_0$)", 0.042, 0.377, 0.003, 0.003, 0.051, '^', '#E67E22'),
        ("Emp: Natural Images ($H_1$)", 0.570, 0.751, 0.075, 0.089, 0.240, '*', '#FF1493')
    ]
    for name, x, y, x_err_l, x_err_u, y_err_u, marker, color in empirical_data:
        ax.errorbar(x, y, xerr=[[x_err_l], [x_err_u]], yerr=[[0], [y_err_u]], fmt='none', ecolor=color, elinewidth=1.5, alpha=0.7, capsize=3, zorder=9)
        ax.scatter(x, y, s=250, marker=marker, facecolor=color, edgecolor='white', linewidth=1.5, zorder=10, label=name)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=min(noises), vmax=max(noises))), ax=ax, pad=0.02)
    cbar.set_label(r'Ambient Gaussian Noise ($\sigma$)', fontsize=13, fontweight='bold')

    ax.axvline(x=chaos_thresh, color='red', linestyle='--', linewidth=2.5, alpha=0.9)
    ax.plot([x_min, x_max], [fiedler_thresh, fiedler_thresh], color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)

    # Formatting and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Thermodynamic Chaos CI (X-axis)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Frustration Ratio CI (Y-axis)', fontsize=14, fontweight='bold')
    ax.set_title('Topological Phase Space with Dual Fail-Safe (Cutoff + MCMC)', fontsize=16, fontweight='bold', pad=15)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', bbox_to_anchor=(0.98, 0.12), fontsize=11, framealpha=0.95, edgecolor='black')

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/simexp4_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only': plot_phase_space_scatter()
    else: run_scatter_experiment()