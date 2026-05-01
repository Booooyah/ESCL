"""
simexp/simexp1_torus_24d.py
--------------------------------
Negative Control Baseline: The 24D Uniform Torus.
Testing ESCL vs VDM on a perfectly orientable (trivial) topology.
"""
import os
import gc
import sys
import time
import numpy as np
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
    auto_select_optimal_k,
    format_topological_verdict
)

if not hasattr(sys.modules['__main__'], '__spec__'):
    sys.modules['__main__'].__spec__ = None


def generate_torus(N=3000):
    """
    Generates a 4D flat Clifford torus.
    Uniform sampling in the (u, v) parameter space ensures a uniform volume measure.
    """
    np.random.seed(42)
    u = np.random.rand(N) * 2 * np.pi
    v = np.random.rand(N) * 2 * np.pi
    X_4d = np.vstack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)]).T

    # Pad with zeros to embed the 4D torus into a 24D ambient space.
    return np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')


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

    random_rotation = ortho_group.rvs(D, random_state=random_seed)
    X_final = np.dot(X_noisy, random_rotation)
    return X_final


def run_torus_test():
    d_prior = 2
    N_points = 3000
    noise_levels = [0.0, 0.20, 0.40, 0.60]

    print("\n" + "=" * 155)
    print("🍩 [Negative Control Baseline] ESCL vs. VDM on 24D Flat Torus")
    print("   Expectation: Trivial (w1=0) - The manifold is perfectly orientable.")
    print("=" * 155)

    print("\n[Data Preparation] Generating fixed master template and noise dictionary...")
    X_base = generate_torus(N=N_points)

    datasets = {}
    for i, noise in enumerate(noise_levels):
        datasets[noise] = apply_environment_noise(X_base, noise, random_seed=300 + i)

    # Initialize a dictionary to cache results for final .npz persistent storage.
    saved_results = {'noises': noise_levels, 'escl': {}, 'vdm': {}}

    for noise in noise_levels:
        print(f"\n[{'=' * 50} Freezing Dataset at Noise: sigma={noise:<4.2f} {'=' * 50}]")
        X_target = datasets[noise]

        saved_results['escl'][noise] = {}
        saved_results['vdm'][noise] = {}

        print(f"  [Auto-Radar] Scanning optimal intrinsic scale for sigma={noise:.2f}...")
        t_auto_start = time.time()

        # Determine the thermodynamic threshold for graph connectivity.
        k_opt_dynamic = auto_select_optimal_k(X_target, k_range=np.arange(10, 100, 5), intrinsic_dim=d_prior, n_jobs=-1)
        t_auto = time.time() - t_auto_start
        print(f"  [Auto-Config] Selected Optimal K={k_opt_dynamic} for noise sigma={noise:.2f} (Elapsed: {t_auto:.2f}s)")
        print("-" * 155)

        # k_values = [
        #     max(10, k_opt_dynamic - 10),
        #     k_opt_dynamic,
        #     k_opt_dynamic + 10,
        #     k_opt_dynamic + 20,
        #     k_opt_dynamic + 30
        # ]
        # k_values = sorted(list(set(k_values)))

        k_values = [k_opt_dynamic]

        header = f"{'K Value (Algo)':<15} | {'Obs λ1':<8} | {'Gap λ2':<8} | {'P_noise':<7} | {'Fiedler CI (Ratio)':<20} | {'Chaos CI (Therm)':<18} | {'Verdict':<28} | {'Time(s)'}"
        print(header)
        print("-" * 155)

        for K in k_values:
            tester_escl = TwoStageBootstrapTester(
                builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': K, 'mutual_knn': False, 'n_jobs': 4},
                analyzer_class=ESCL_H1_Analyzer,
                analyzer_kwargs={'beta': 5.0, 'auto_beta': True, 'cutoff': 'auto'},
                b_bootstrap=100, max_workers=5, alpha_mcmc=0.05, solver_tol=1e-8
            )

            tester_vdm = TwoStageBootstrapTester(
                builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': K, 'mutual_knn': False, 'n_jobs': 4},
                analyzer_class=VDM_H1_Analyzer,
                analyzer_kwargs={'epsilon_scale': 1.0},
                b_bootstrap=30, max_workers=2, alpha_mcmc=0.05, solver_tol=1e-8
            )

            # --- ESCL Analysis Pipeline ---
            t0 = time.time()
            _, _, res_escl = tester_escl.test(X_target)
            t_escl = time.time() - t0

            escl_verdict_str = format_topological_verdict(
                algo_name=f"K={K:<2} (ESCL)",
                res_dict=res_escl,
                exec_time=t_escl,
                expected_topology='trivial',
                intrinsic_dim=d_prior
            )
            print(escl_verdict_str)
            saved_results['escl'][noise][K] = res_escl[0] if res_escl else None

            # --- VDM Analysis Pipeline ---
            t1 = time.time()
            _, _, res_vdm = tester_vdm.test(X_target)
            t_vdm = time.time() - t1

            vdm_verdict_str = format_topological_verdict(
                algo_name=f"K={K:<2} (VDM)",
                res_dict=res_vdm,
                exec_time=t_vdm,
                expected_topology='trivial',
                intrinsic_dim=d_prior
            )
            print(vdm_verdict_str)
            saved_results['vdm'][noise][K] = res_vdm[0] if res_vdm else None

            print("-" * 155)
            gc.collect()

    # Save experimental data to disk
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/simexp1_torus_24d.npz', **saved_results)
    print("\n Complete simulation data saved to Results/simexp1_torus_24d.npz")


if __name__ == "__main__":
    t_global = time.time()
    run_torus_test()
    print(f"\n Torus negative control test completed! Total elapsed time: {time.time() - t_global:.1f}s")