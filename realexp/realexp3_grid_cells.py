"""
realexp/realexp3_grid_cells.py
----------------------------------
High-Dimensional Neural Phase Space (Grid Cells, Nature 2022)
Negative Control: Validating orientable topology (T^2) in 166D noisy biological data.
Demonstrating ESCL's Exact Size Control under severe spiking heteroscedasticity.
"""

import os
import gc
import sys
import time
import numpy as np
from sklearn.decomposition import PCA

# Lock underlying threads to prevent memory explosion during numpy/scipy multiprocessing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Dynamically append the project root directory to the system path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import official preprocessing and spike-binning functions from Gardner et al.
from realexp.exp_utils import get_spikes

from src.models import (
    TwoStageBootstrapTester,
    ESCL_H1_Analyzer,
    VDM_H1_Analyzer,
    auto_select_optimal_k,
    format_topological_verdict
)

def load_empirical_nature_data(max_samples=6000):
    print("\n" + "=" * 135)
    print(f"🐁 [Data Preparation] Parsing Empirical Rat Grid Cells (Nature 2022 Dataset)")
    print("=" * 135)

    # Note to Reviewers: Please download the empirical dataset from Gardner et al. (Nature 2022)
    # and place the raw files into the local /data/ directory before executing this script.
    try:
        spikes_bin, xx, yy, aa, tt = get_spikes(
            rat_name='R', mod_name='1', day_name='day1', sess_name='OF',
            bType=None, bSmooth=True, bBinned=True, bSpeed=True,
            folder='data/Toroidal_topology_grid_cell_data/'
        )
    except Exception as e:
        print(f"🚨 Failed to load empirical grid cell data. Error: {e}")
        print("Please ensure the Nature 2022 raw data is extracted into the 'data/' folder.")
        raise

    # Square-root transformation to stabilize the variance of Poisson spiking noise
    X_neural = np.sqrt(spikes_bin)

    # Spherical normalization: mean-centering and projection onto the unit hypersphere
    X_neural = X_neural - np.mean(X_neural, axis=0)
    X_neural = X_neural / (np.linalg.norm(X_neural, axis=1, keepdims=True) + 1e-8)

    # Project to 6D via PCA to extract the primary macroscopic firing structure
    # and mitigate isotropic biological background noise.
    pca = PCA(n_components=6)
    X_neural = pca.fit_transform(X_neural)

    # Subsample to maintain computational tractability while preserving global topology
    if X_neural.shape[0] > max_samples:
        np.random.seed(42)
        indices = np.random.choice(X_neural.shape[0], max_samples, replace=False)
        X_neural = X_neural[indices]

    print(f"  -> Empirical Neural Manifold: Shape {X_neural.shape} (Projected 6D State Space)")
    return X_neural

def run_grid_cell_phase_diagram():
    try:
        X_noisy = load_empirical_nature_data(max_samples=6000)
    except Exception as e:
        return

    d_prior = 2
    print("\n" + "=" * 135)
    print(" [Phase 2] Topological Inference on the Neural Manifold (H0: Trivial Torus)")
    print("=" * 135)

    print(f"  [Auto-Radar] Scanning optimal intrinsic scale...")
    t_auto_start = time.time()

    # Biological manifolds exhibit highly non-uniform spatial sampling.
    # We enforce mutual K-NN to symmetrically prune sparse, artifactual connections.
    k_opt_dynamic = auto_select_optimal_k(
        X_noisy, k_range=np.arange(60, 160, 10), intrinsic_dim=d_prior, mutual_knn=True, n_jobs=-1
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
        b_bootstrap=100, max_workers=5, alpha_mcmc=0.05, solver_tol=1e-8
    )

    tester_vdm = TwoStageBootstrapTester(
        builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_dynamic, 'mutual_knn': True, 'n_jobs': 4},
        analyzer_class=VDM_H1_Analyzer,
        analyzer_kwargs={'epsilon_scale': 1.0},
        b_bootstrap=30, max_workers=2, alpha_mcmc=0.05, solver_tol=1e-8
    )

    # --- ESCL Pipeline ---
    t0 = time.time()
    _, _, res_escl = tester_escl.test(X_noisy)
    print(format_topological_verdict(f"K={k_opt_dynamic:<2} (ESCL)", res_escl, time.time() - t0))

    # --- VDM Pipeline ---
    t1 = time.time()
    _, _, res_vdm = tester_vdm.test(X_noisy)
    print(format_topological_verdict(f"K={k_opt_dynamic:<2} (VDM)", res_vdm, time.time() - t1))
    print("-" * 135)

    # Save empirical data to disk for visualization rendering
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/realexp3_grid_cells.npz',
                        res_escl=res_escl[0], res_vdm=res_vdm[0], X_target=X_noisy)
    print("\n Complete empirical data saved to Results/realexp3_grid_cells.npz")

if __name__ == "__main__":
    t_global = time.time()
    run_grid_cell_phase_diagram()
    print(f"\n Rat grid cell topology inference completed successfully in {time.time() - t_global:.1f}s")