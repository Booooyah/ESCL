"""
simexp/simexp8_double_klein_4d.py
--------------------------------------
High-Dimensional Positive Control: The 24D Double Klein Bottle (K^2 x K^2).
Intrinsic Dimension d = 4.
Purpose:
1. Prove ESCL's Z2 mass gap holds for higher-dimensional O(d) bundles.
2. Demonstrate the computational supremacy of ESCL (N x N) over VDM (Nd x Nd).
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


def generate_double_klein_bottle(N=5000):
    """
    Generates a Double Klein Bottle (K^2 x K^2) with intrinsic dimension d=4.
    The manifold is initially embedded in 8D via a Cartesian product of two 4D embeddings,
    and then padded and rotated into a 24D ambient space.
    """
    np.random.seed(42)
    R, r = 4.0, 1.0
    max_J = r * (R + r)

    def sample_klein_4d(n_points):
        u_list, v_list = [], []
        while len(u_list) < n_points:
            batch = n_points - len(u_list)
            u_cand = np.random.rand(batch) * 2 * np.pi
            v_cand = np.random.rand(batch) * 2 * np.pi
            w_cand = np.random.rand(batch) * max_J
            # Apply rejection sampling based on the Jacobian determinant
            J_val = r * np.sqrt((R + r * np.cos(v_cand)) ** 2 + (r ** 2 / 4.0) * np.sin(v_cand) ** 2)
            valid = w_cand <= J_val
            u_list.extend(u_cand[valid])
            v_list.extend(v_cand[valid])

        u, v = np.array(u_list[:n_points]), np.array(v_list[:n_points])
        x1 = (R + r * np.cos(v)) * np.cos(u)
        x2 = (R + r * np.cos(v)) * np.sin(u)
        x3 = r * np.sin(v) * np.cos(u / 2)
        x4 = r * np.sin(v) * np.sin(u / 2)
        return np.vstack([x1, x2, x3, x4]).T

    # Generate two independent Klein bottles and concatenate to form the 8D embedding
    K1 = sample_klein_4d(N)
    K2 = sample_klein_4d(N)
    X_8d = np.hstack([K1, K2])

    # Pad with zeros to embed into the 24D ambient space
    X_base = np.pad(X_8d, ((0, 0), (0, 16)), mode='constant')
    return X_base


def apply_environment_noise(X_base, noise_scale, random_seed):
    """
    Injects isotropic ambient Gaussian noise and applies a random orthogonal
    transformation to simulate high-dimensional measurement perturbation.
    """
    np.random.seed(random_seed)
    N, D = X_base.shape
    X_noisy = X_base.copy()
    if noise_scale > 0:
        X_noisy += np.random.normal(0, noise_scale, size=(N, D))
    return np.dot(X_noisy, ortho_group.rvs(D, random_state=random_seed))


def run_double_klein_test():
    d_prior = 4  # Core challenge: The intrinsic dimension is elevated to 4
    N_points = 5000  # The 4D space is highly sparse, requiring an increased sample size to maintain connectivity
    noise_levels = np.linspace(0.10, 0.40, 4)

    results = {'escl_rho': [], 'vdm_rho': [], 'escl_time': [], 'vdm_time': []}

    print("\n" + "=" * 110)
    print(f"🌌 [High-Dim Generalization] 24D Double Klein Bottle (d={d_prior}, N={N_points})")
    print("   Goal 1: Prove topological rigidity holds for O(4) bundles.")
    print("   Goal 2: Demonstrate N x N vs Nd x Nd computational supremacy.")
    print("=" * 110)

    X_base = generate_double_klein_bottle(N=N_points)

    for i, noise in enumerate(noise_levels):
        print(f"\n[{'=' * 30} Ambient Gaussian Noise: sigma={noise:<4.2f} {'=' * 30}]")
        X_target = apply_environment_noise(X_base, noise, random_seed=400 + i)

        # Higher-dimensional manifolds demand a larger bandwidth (K) to ensure local connectivity
        t_auto = time.time()
        k_opt = auto_select_optimal_k(X_target, k_range=np.arange(40, 120, 15), intrinsic_dim=d_prior, n_jobs=-1)
        print(f"  [Auto-Radar] Locked K={k_opt} (Elapsed: {time.time() - t_auto:.1f}s)")

        # ----------------------------------------------------
        # ESCL Analysis Pipeline
        # ----------------------------------------------------
        tester_escl = TwoStageBootstrapTester(
            builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt, 'mutual_knn': False, 'n_jobs': 4},
            analyzer_class=ESCL_H1_Analyzer,
            analyzer_kwargs={'beta': 5.0, 'auto_beta': True, 'cutoff': 'auto'},
            b_bootstrap=10, max_workers=2  # Focused on evaluating the algebraic gap; reduced sampling
        )

        t0 = time.time()
        _, _, res_escl = tester_escl.test(X_target)
        t_e = time.time() - t0
        rho_e = (res_escl[0].get('obs_lam1', 0) / res_escl[0].get('obs_lam2', 1)) if res_escl else np.nan

        # ----------------------------------------------------
        # VDM Analysis Pipeline
        # ----------------------------------------------------
        tester_vdm = TwoStageBootstrapTester(
            builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt, 'mutual_knn': False, 'n_jobs': 4},
            analyzer_class=VDM_H1_Analyzer,
            analyzer_kwargs={'epsilon_scale': 1.0},
            b_bootstrap=10, max_workers=2
        )

        t1 = time.time()
        _, _, res_vdm = tester_vdm.test(X_target)
        t_v = time.time() - t1
        rho_v = (res_vdm[0].get('obs_lam1', 0) / res_vdm[0].get('obs_lam2', 1)) if res_vdm else np.nan

        results['escl_rho'].append(rho_e)
        results['vdm_rho'].append(rho_v)
        results['escl_time'].append(t_e)
        results['vdm_time'].append(t_v)

        print(f"  -> ESCL (N x N)   | rho = {rho_e:.4f} | Time = {t_e:.1f}s")
        print(f"  -> VDM  (Nd x Nd) | rho = {rho_v:.4f} | Time = {t_v:.1f}s")

    # Save experimental data to disk
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/simexp8_double_klein_4d.npz', noises=noise_levels, **results)
    print("\n💾 Complete simulation data saved to Results/simexp8_double_klein_4d.npz")

    plot_double_klein(noise_levels, results)


def plot_double_klein(noise_levels, results):
    """Renders the dual-panel comparison of topological rigidity and computational scaling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Topological Gap (Robustness on Higher-Rank Bundles)
    ax1.plot(noise_levels, results['vdm_rho'], marker='D', markersize=9, linewidth=2.5,
             color='#C0392B', linestyle='--', label=r'VDM ($Nd \times Nd$)')
    ax1.plot(noise_levels, results['escl_rho'], marker='s', markersize=9, linewidth=3.5,
             color='#27AE60', linestyle='-', label=r'ESCL ($N \times N$)')
    ax1.axhline(y=0.20, color='black', linestyle=':', linewidth=2, label='Macro Gap Threshold')

    ax1.set_xlabel(r'Ambient Gaussian Noise ($\sigma$)', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'Fiedler Ratio ($\rho = \lambda_1 / \lambda_2$)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Robustness on Intrinsic 4D Manifold', fontsize=15, fontweight='bold', pad=12)
    ax1.set_ylim(-0.05, 1.25)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Panel B: Computational Complexity (Mitigating Matrix Dimension Explosion)
    width = 0.02
    ax2.bar(noise_levels - width/2, results['escl_time'], width, label='ESCL Solver Time', color='#2ECC71', edgecolor='black')
    ax2.bar(noise_levels + width/2, results['vdm_time'], width, label='VDM Solver Time', color='#E74C3C', edgecolor='black')

    ax2.set_xlabel(r'Ambient Gaussian Noise ($\sigma$)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Execution Time (Seconds)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Matrix Dimension Explosion ($N=5000, d=4$)', fontsize=15, fontweight='bold', pad=12)
    ax2.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/simexp8_double_klein_4d.pdf', dpi=300, bbox_inches='tight')
    print("✅ Supplementary plot successfully saved to 'Figs/simexp8_double_klein_4d.pdf'")
    plt.show()

# Support offline plotting directly from cached .npz files
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        try:
            data = np.load('Results/simexp8_double_klein_4d.npz')
            plot_double_klein(data['noises'], {k: data[k] for k in data.files if k != 'noises'})
        except Exception as e:
            print(f"🚨 Plotting failed: {e}")
    else:
        run_double_klein_test()