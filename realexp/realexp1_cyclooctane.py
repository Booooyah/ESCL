"""
realexp/realexp1_cyclooctane.py
-----------------------------------
Real-World Molecular Phase Space (Cyclooctane)
Detecting orientability (w1=0) and isolating singular geometric cross-caps
via the ESCL topological frustration field (F_i).
"""

import os
import gc
import sys
import time
import numpy as np
import scipy.io as sio

# Lock underlying threads to prevent memory explosion during numpy/scipy multiprocessing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Dynamically append the project root directory to the system path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    auto_select_optimal_k,
    ESCL_H1_Analyzer,
    VDM_H1_Analyzer,
    TwoStageBootstrapTester,
    format_topological_verdict
)

# Import refactored publication-ready frame distortion renderers
from utils.visualization import plot_singularity_map, plot_singularity_map_extreme


def run_cyclooctane_experiment():
    print("\n" + "=" * 135)
    print(" 🔬 Exp 1: The Cyclooctane Phase Space (w1 Twist & Singularity Map)")
    print("=" * 135)

    # ---------------------------------------------------------
    # 1. Load Empirical Data
    # ---------------------------------------------------------
    file_path = 'data/pointsCycloOctane.mat'  # Ensure this path points to the local empirical dataset
    try:
        data = sio.loadmat(file_path)
        X_real = data['pointsCycloOctane']
        # Ensure data shape is consistently (N, D)
        if X_real.shape[0] < X_real.shape[1]:
            X_real = X_real.T
        print(f"✅ Empirical data loaded successfully: Shape {X_real.shape} (N={X_real.shape[0]})")
    except Exception as e:
        print(f"🚨 Failed to load data from {file_path}. Error: {e}")
        return

    # ---------------------------------------------------------
    # 2. Experimental Setup
    # ---------------------------------------------------------
    d_est = 2  # The intrinsic dimension of the cyclooctane conformational space is 2
    # Empirical molecular conformation maintains a high SNR; no artificial noise injection is required
    noise_levels = [0.00]

    final_X_test = None
    final_res_escl = None
    k_opt_zero_noise = None

    for noise in noise_levels:
        print(f"\n[{'=' * 40}  Ambient Gaussian Noise: sigma={noise:<4.2f} {'=' * 40}]")

        # Inject ambient noise strictly if sigma > 0
        np.random.seed(42)
        X_test = X_real + np.random.normal(0, noise, size=X_real.shape) if noise > 0 else X_real.copy()

        # ---------------------------------------------------------
        # 3. Auto-Radar: Intrinsic Scale Detection
        # ---------------------------------------------------------
        print(f"  [Auto-Radar] Scanning optimal intrinsic scale for Cheeger bottleneck...")
        t_auto_start = time.time()

        # Cyclooctane contains extremely narrow intersecting singularities (cross-caps),
        # requiring a wider receptive field to maintain global graph connectivity.
        k_opt_dynamic = auto_select_optimal_k(
            X_test, k_range=np.arange(10, 100, 10), intrinsic_dim=d_est, mutual_knn=False, n_jobs=-1
        )
        t_auto = time.time() - t_auto_start
        print(f"  [Auto-Config] Locked Optimal K={k_opt_dynamic} (Elapsed: {t_auto:.2f}s)")
        print("-" * 135)

        if noise == 0.00:
            k_opt_zero_noise = k_opt_dynamic

        # Evaluate exclusively at the optimal K to demonstrate the framework's zero-shot capability
        k_values = [k_opt_dynamic]

        header = f"{'Algorithm':<15} | {'Obs lambda1':<8} | {'Gap lambda2':<8} | {'P_noise':<7} | {'Fiedler CI (Ratio)':<22} | {'Chaos CI (Therm)':<18} | {'Verdict':<28} | {'Time(s)'}"
        print(header)
        print("-" * 135)

        for K in k_values:
            # ---------------------------------------------------------
            # 4. Topological Inference (ESCL vs VDM)
            # ---------------------------------------------------------
            tester_escl = TwoStageBootstrapTester(
                builder_kwargs={'intrinsic_dim': d_est, 'k_neighbors': K, 'mutual_knn': False, 'n_jobs': 4},
                analyzer_class=ESCL_H1_Analyzer,
                analyzer_kwargs={'beta': 5.0, 'auto_beta': True, 'cutoff': 'auto'},  # Automatically prune high-noise edges
                b_bootstrap=100, max_workers=5, alpha_mcmc=0.05, solver_tol=1e-8
            )

            tester_vdm = TwoStageBootstrapTester(
                builder_kwargs={'intrinsic_dim': d_est, 'k_neighbors': K, 'mutual_knn': False, 'n_jobs': 4},
                analyzer_class=VDM_H1_Analyzer,
                analyzer_kwargs={'epsilon_scale': 1.0},
                b_bootstrap=30, max_workers=2, alpha_mcmc=0.05, solver_tol=1e-8  # VDM acts as a baseline; reduce resampling to accelerate execution
            )

            # --- ESCL Pipeline ---
            t0 = time.time()
            _, _, res_escl = tester_escl.test(X_test)
            t_escl = time.time() - t0
            print(format_topological_verdict(f"K={K:<2} (ESCL)", res_escl, t_escl))

            # --- VDM Pipeline ---
            t1 = time.time()
            _, _, res_vdm = tester_vdm.test(X_test)
            t_vdm = time.time() - t1
            print(format_topological_verdict(f"K={K:<2} (VDM)", res_vdm, t_vdm))
            print("-" * 135)

            if K == k_opt_dynamic and noise == 0.00:
                final_X_test = X_test
                final_res_escl = res_escl[0]

            gc.collect()

    os.makedirs('Results', exist_ok=True)
    np.savez('Results/realexp1_cyclooctane.npz', res_escl=final_res_escl, X_test=final_X_test)

    # ---------------------------------------------------------
    # 5. Visual Rendering: The Topological Singularity Map
    # ---------------------------------------------------------
    if final_res_escl and final_X_test is not None:
        print("\n [Visualization] Extracting Frame Distortion Field & Rendering Singularity Map...")

        f1_val = final_res_escl.get('f1_field', None)
        indices = final_res_escl.get('main_comp_indices', None)

        if f1_val is not None and indices is not None:
            # Map the localized frustration back to the global coordinate system
            N_total = final_X_test.shape[0]
            F_field_full = np.zeros(N_total)

            # Compute the local topological frustration magnitude F_i = |f1|^2 and normalize
            f1_squared = np.abs(f1_val) ** 2
            F_field_full[indices] = f1_squared / (np.max(f1_squared) + 1e-12)

            # Render the base and extreme singularity heatmaps (via utils/visualization.py)
            plot_singularity_map(
                X=final_X_test,
                F_field=F_field_full,
                title=f"Cyclooctane Singularity Map (Auto-K={k_opt_zero_noise})"
            )

            plot_singularity_map_extreme(
                X=final_X_test,
                F_field=F_field_full,
                title=f"Cyclooctane Singularity Extreme (Auto-K={k_opt_zero_noise})"
            )
        else:
            print(" Error: Spectral field 'f1_field' not found in ESCL results.")


if __name__ == "__main__":
    t_global = time.time()
    run_cyclooctane_experiment()
    print(f"\n Experiment 1 (Cyclooctane) completed successfully in {time.time() - t_global:.1f}s")