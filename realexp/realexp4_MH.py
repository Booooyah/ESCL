"""
realexp/realexp4_MH.py
--------------------------
Real-World MAV SO(3) Manifold (EuRoC Dataset)
Negative Control (H0): Validating orientability in 9D drone trajectories.
Demonstrating ESCL's exact size control and spectral gap preservation
against continuous sensor drift, where VDM suffers from spectral collapse.
"""

import os
import gc
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

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
    auto_select_optimal_k,
    format_topological_verdict
)

def load_euroc_so3_manifold(csv_path, max_samples=2500):
    print("\n" + "=" * 135)
    print(f" [Data Preparation] Parsing Real-World Robotics SO(3) Manifold")
    print("=" * 135)

    # Note to Reviewers: Please download the EuRoC MAV dataset (e.g., MH_01_easy ground truth)
    # and place the CSV file into the local 'data/' directory before executing this script.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f" File not found: {csv_path}. "
            "Please ensure the EuRoC MAV dataset CSV is placed in the data/ folder."
        )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Extract quaternion orientations representing the MAV state space
    target_cols = ['q_RS_w', 'q_RS_x', 'q_RS_y', 'q_RS_z']
    actual_cols = [col for target in target_cols for col in df.columns if target in col]

    df_quats = df[actual_cols].dropna().values

    # Scipy expects quaternions in scalar-last (x, y, z, w) format
    quats_xyzw = np.column_stack((df_quats[:, 1], df_quats[:, 2], df_quats[:, 3], df_quats[:, 0]))

    # Uniformly subsample the trajectory to maintain computational tractability
    if len(quats_xyzw) > max_samples:
        indices = np.linspace(0, len(quats_xyzw) - 1, max_samples, dtype=int)
        quats_xyzw = quats_xyzw[indices]

    # Convert quaternions to flattened 3x3 rotation matrices, embedding the SO(3) manifold into 9D space
    rots = Rotation.from_quat(quats_xyzw)
    X_9d = rots.as_matrix().reshape(-1, 9)

    print(f"  -> Real MAV SO(3) poses extracted: Shape {X_9d.shape} (9D Ambient Space)")
    return X_9d

def run_euroc_phase_diagram():
    csv_file_path = "data/MH_data.csv"  # Relative path to the local EuRoC ground truth CSV
    try:
        X_so3 = load_euroc_so3_manifold(csv_file_path, max_samples=2500)
    except Exception as e:
        print(e)
        return

    # The Special Orthogonal group SO(3) is a 3-dimensional manifold
    d_prior = 3

    print("\n" + "=" * 135)
    print(" [Phase 2] Topological Inference on MAV SO(3) Trajectory (H0: Trivial Bundle)")
    print("=" * 135)

    print(f"  [Auto-Radar] Scanning optimal intrinsic scale...")
    t_auto_start = time.time()

    # Enforce mutual K-NN to prevent severe temporal short-circuits in the sequential trajectory
    k_opt_dynamic = auto_select_optimal_k(
        X_so3, k_range=np.arange(20, 120, 10), intrinsic_dim=d_prior, mutual_knn=True, n_jobs=-1
    )
    print(f"  [Auto-Config] Selected Optimal K={k_opt_dynamic} (Elapsed: {time.time() - t_auto_start:.2f}s)")
    print("-" * 135)

    header = f"{'Algorithm':<15} | {'Obs lambda1':<8} | {'Gap lambda2':<8} | {'P_noise':<7} | {'Fiedler CI (Ratio)':<22} | {'Chaos CI (Therm)':<18} | {'Verdict':<28} | {'Time(s)'}"
    print(header)
    print("-" * 135)

    # ---------------------------------------------------------
    # Topological Inference (ESCL vs VDM)
    # ---------------------------------------------------------
    tester_escl = TwoStageBootstrapTester(
        builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_dynamic, 'mutual_knn': True, 'n_jobs': 4},
        analyzer_class=ESCL_H1_Analyzer,
        analyzer_kwargs={'beta': 5.0, 'auto_beta': True, 'cutoff': 'auto'},
        b_bootstrap=100, max_workers=5, alpha_mcmc=0.05, solver_tol=1e-5
    )

    tester_vdm = TwoStageBootstrapTester(
        builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_dynamic, 'mutual_knn': True, 'n_jobs': 4},
        analyzer_class=VDM_H1_Analyzer,
        analyzer_kwargs={'epsilon_scale': 1.0},
        b_bootstrap=30, max_workers=2, alpha_mcmc=0.05, solver_tol=1e-5
    )

    # --- ESCL Pipeline ---
    t0 = time.time()
    _, _, res_escl = tester_escl.test(X_so3)
    print(format_topological_verdict(f"K={k_opt_dynamic:<2} (ESCL)", res_escl, time.time() - t0))

    # --- VDM Pipeline ---
    # Note: VDM is highly susceptible to spectral gap collapse (lambda_2 -> 0) here due to metric leakage
    t1 = time.time()
    _, _, res_vdm = tester_vdm.test(X_so3)
    print(format_topological_verdict(f"K={k_opt_dynamic:<2} (VDM)", res_vdm, time.time() - t1))
    print("-" * 135)

    # Save empirical data to disk for visualization rendering
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/realexp4_MH.npz',
                        res_escl=res_escl[0] if res_escl else None,
                        res_vdm=res_vdm[0] if res_vdm else None,
                        X_target=X_so3)
    print("\n Complete empirical data saved to Results/realexp4_MH.npz")

if __name__ == "__main__":
    t_global = time.time()
    run_euroc_phase_diagram()
    print(f"\n Real-world UAV trajectory topology inference completed successfully in {time.time() - t_global:.1f}s")