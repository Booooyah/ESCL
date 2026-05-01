"""
simexp/simexp3_asymmetric_bias.py
--------------------------------------
Demonstrating that VDM (Continuous Gauge) suffers from Metric Leakage (Spectral Gap Collapse),
while ESCL (Discrete Gauge) maintains perfect algebraic rigidity.
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

def generate_manifold(manifold_type, N=2000, noise=0.0):
    """
    Generates pure 24D embedded manifolds and injects ambient Gaussian noise.
    """
    np.random.seed(42)

    if manifold_type == 'torus':
        u = np.random.rand(N) * 2 * np.pi
        v = np.random.rand(N) * 2 * np.pi
        x1, x2 = np.cos(u), np.sin(u)
        x3, x4 = np.cos(v), np.sin(v)
    else:  # klein
        # Use rejection sampling to guarantee a uniform volume measure on the Klein bottle
        R, r = 4.0, 1.0
        max_J = r * (R + r)
        u_list, v_list = [], []
        while len(u_list) < N:
            batch = N - len(u_list)
            u_cand = np.random.rand(batch) * 2 * np.pi
            v_cand = np.random.rand(batch) * 2 * np.pi
            w_cand = np.random.rand(batch) * max_J
            J_val = r * np.sqrt((R + r * np.cos(v_cand))**2 + (r**2 / 4.0) * np.sin(v_cand)**2)
            valid = w_cand <= J_val
            u_list.extend(u_cand[valid])
            v_list.extend(v_cand[valid])
        u, v = np.array(u_list[:N]), np.array(v_list[:N])

        x1 = (R + r * np.cos(v)) * np.cos(u)
        x2 = (R + r * np.cos(v)) * np.sin(u)
        x3 = r * np.sin(v) * np.cos(u / 2)
        x4 = r * np.sin(v) * np.sin(u / 2)

    X_4d = np.vstack([x1, x2, x3, x4]).T
    X_base = np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')

    # Inject ambient Gaussian noise
    np.random.seed(100 + int(noise * 100))
    X_noisy = X_base + np.random.normal(0, noise, size=X_base.shape) if noise > 0 else X_base

    # Apply a random 24D orthogonal rotation
    return np.dot(X_noisy, ortho_group.rvs(24, random_state=42))


def run_bias_experiment():
    noise_levels = np.linspace(0.0, 0.40, 9)  # Uniformly sample 9 points from 0.0 to 0.40
    manifolds = {'Torus (Trivial)': 'torus', 'Klein (Twisted)': 'klein'}

    # Flatten dictionary for convenient .npz storage
    results = {
        'escl_torus': [], 'escl_klein': [],
        'vdm_torus': [], 'vdm_klein': []
    }

    N_points = 2000
    K_neighbors = 40  #  Core premise: Fix the neighborhood size K to ensure an identical graph structure, isolating the algebraic spectral gap comparison.
    d_prior = 2

    print("\n" + "=" * 100)
    print(" Running Spectral Gap Collapse Experiment (Asymmetric Bias)")
    print("=" * 100)

    for name, m_type in manifolds.items():
        print(f"\n--- Processing Manifold: {name} ---")
        for noise in noise_levels:
            X = generate_manifold(m_type, N=N_points, noise=noise)

            # 1. Construct the unified graph skeleton
            builder = TopologyGraphBuilder(intrinsic_dim=d_prior, k_neighbors=K_neighbors, n_jobs=-1)
            E, edges, comps = builder.build(X, use_cknn=False)

            if not comps:
                print(f"Noise {noise:.2f}: Graph Shattered!")
                results[f'escl_{m_type}'].append(np.nan)
                results[f'vdm_{m_type}'].append(np.nan)
                continue

            main_comp = max(comps, key=len)

            best_cutoff = auto_select_cutoff(E, edges, target_ratio=0.90)

            # 2. Run ESCL (Extract algebraic eigenvalues only, bypass Bootstrap)
            escl = ESCL_H1_Analyzer(beta=5.0, auto_beta=True, cutoff=best_cutoff)
            lam1_e, lam2_e, _, _, _ = escl.analyze(E, edges, main_comp, X)
            rho_e = (lam1_e / lam2_e) if (lam2_e and lam2_e > 1e-5) else np.nan

            # 3. Run VDM (Extract algebraic eigenvalues only, bypass Bootstrap)
            vdm = VDM_H1_Analyzer(epsilon_scale=1.0)
            lam1_v, lam2_v, _, _, _ = vdm.analyze(E, edges, main_comp, X)
            rho_v = (lam1_v / lam2_v) if (lam2_v and lam2_v > 1e-5) else np.nan

            results[f'escl_{m_type}'].append(rho_e)
            results[f'vdm_{m_type}'].append(rho_v)

            print(f"sigma = {noise:.2f} | ESCL rho: {rho_e:.4f} | VDM rho: {rho_v:.4f}")

    #  Save experimental data to disk
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/simexp3_asymmetric_bias.npz', noises=noise_levels, **results)
    print("\n Complete simulation data saved to Results/simexp3_asymmetric_bias.npz")

    # Render publication-ready plot
    plot_results(noise_levels, results, N_points)


def plot_results(noise_levels, results, N):
    """Publication-ready line plot (NeurIPS style)."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    # Dynamic macroscopic threshold based on Cheeger bounds
    macro_threshold = 1.5 / np.log(N)

    # Plot ESCL curves
    ax.plot(noise_levels, results['escl_torus'], marker='o', markersize=8, linewidth=2.5,
            color='#2ECC71', label='ESCL (Torus - Trivial)', zorder=4)
    ax.plot(noise_levels, results['escl_klein'], marker='s', markersize=8, linewidth=2.5,
            color='#27AE60', linestyle='--', label='ESCL (Klein - Twisted)', zorder=4)

    # Plot VDM curves
    ax.plot(noise_levels, results['vdm_torus'], marker='^', markersize=8, linewidth=2.5,
            color='#E74C3C', label='VDM (Torus - Trivial)', zorder=3)
    ax.plot(noise_levels, results['vdm_klein'], marker='D', markersize=8, linewidth=2.5,
            color='#C0392B', linestyle='--', label='VDM (Klein - Twisted)', zorder=3)

    # Plot macroscopic phase transition threshold line
    ax.axhline(y=macro_threshold, color='black', linestyle=':', linewidth=2,
               label=f'Macro Gap Threshold ({macro_threshold:.2f})', zorder=2)

    # Fill background regions (Trivial vs. Twisted)
    ax.axhspan(0, macro_threshold, facecolor='#E8F8F5', alpha=0.5, zorder=1)
    ax.axhspan(macro_threshold, 1.0, facecolor='#FDEDEC', alpha=0.5, zorder=1)

    # Formatting and labels
    ax.set_xlabel(r'Ambient Gaussian Noise ($\sigma$)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Fiedler Ratio ($\rho = \lambda_1 / \lambda_2$)', fontsize=14, fontweight='bold')
    ax.set_title('The Topological Mass Gap vs. Metric Leakage', fontsize=16, fontweight='bold', pad=15)

    ax.set_xlim(0.0, max(noise_levels))
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Extend Y-axis upper limit to 1.35 for legend space
    plt.ylim(-0.05, 1.35)

    # Place legend in the upper right, configured in 2 columns to save vertical space
    plt.legend(loc='upper right', ncol=2, framealpha=0.95, fontsize=12)

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/simexp3_asymmetric_bias.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(" Plot successfully saved to 'Figs/simexp3_asymmetric_bias.pdf'")
    plt.show()

if __name__ == "__main__":
    t0 = time.time()
    run_bias_experiment()
    print(f"\n Experiment 3 completed! Total elapsed time: {time.time() - t0:.1f}s")