"""
simexp/simexp2_klein_24d.py
------------------------------------
Positive Control Baseline: The 24D Uniform Klein Bottle.
Testing ESCL vs VDM on a macroscopically non-orientable topological singularity.
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


def generate_base_klein_bottle(N=3000):
    """
    Generates a pure 24D embedded Klein bottle.
    Utilizes rejection sampling to guarantee an absolutely uniform volume
    measure across the 4D manifold surface, eliminating density artifacts.
    """
    np.random.seed(42)
    R = 4.0
    r = 1.0

    # Maximum of the Jacobian determinant J(u,v) (occurs when cos(v) = 1, sin(v) = 0)
    max_J = r * (R + r)

    u_list, v_list = [], []

    # Rejection sampling loop
    while len(u_list) < N:
        batch_size = N - len(u_list)
        u_cand = np.random.rand(batch_size) * 2 * np.pi
        v_cand = np.random.rand(batch_size) * 2 * np.pi
        w_cand = np.random.rand(batch_size) * max_J

        # Calculate the true area weight for the current candidate point
        J_val = r * np.sqrt((R + r * np.cos(v_cand))**2 + (r**2 / 4.0) * np.sin(v_cand)**2)

        # Accept samples falling under the probability threshold
        valid_mask = w_cand <= J_val
        u_list.extend(u_cand[valid_mask])
        v_list.extend(v_cand[valid_mask])

    u = np.array(u_list[:N])
    v = np.array(v_list[:N])

    # Non-self-intersecting analytic embedding of the 4D Klein bottle
    x1 = (R + r * np.cos(v)) * np.cos(u)
    x2 = (R + r * np.cos(v)) * np.sin(u)
    x3 = r * np.sin(v) * np.cos(u / 2)
    x4 = r * np.sin(v) * np.sin(u / 2)

    X_4d = np.vstack([x1, x2, x3, x4]).T

    # Pad with zeros to embed into a 24D ambient space
    X_base = np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')
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

    random_rotation = ortho_group.rvs(D, random_state=random_seed)
    X_final = np.dot(X_noisy, random_rotation)
    return X_final


def run_klein_bottle_test():
    d_prior = 2
    N_points = 3000
    noise_levels = [0.00, 0.20, 0.40, 0.60]

    print("\n" + "=" * 155)
    print("🌌 [Positive Control Baseline] ESCL vs. VDM on 24D Klein Bottle")
    print("   Expectation: Twisted (w1=1) - The manifold contains an intrinsic Moebius tear.")
    print("=" * 155)

    print("\n[Data Preparation] Generating fixed master template and noise dictionary...")
    X_base = generate_base_klein_bottle(N=N_points)

    datasets = {}
    for i, noise in enumerate(noise_levels):
        datasets[noise] = apply_environment_noise(X_base, noise, random_seed=300 + i)

    # Initialize a dictionary to cache results for final .npz persistent storage
    saved_results = {'noises': noise_levels, 'escl': {}, 'vdm': {}}

    for noise in noise_levels:
        print(f"\n[{'=' * 50} 🌪️ Freezing Dataset at Noise: sigma={noise:<4.2f} {'=' * 50}]")
        X_target = datasets[noise]

        saved_results['escl'][noise] = {}
        saved_results['vdm'][noise] = {}

        print(f"  [Auto-Radar] Scanning optimal intrinsic scale for sigma={noise:.2f}...")
        t_auto_start = time.time()
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
                expected_topology='twisted',  # Expectation corresponds to a macroscopic cohomological obstruction (H1)
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
                expected_topology='twisted',
                intrinsic_dim=d_prior
            )
            print(vdm_verdict_str)
            saved_results['vdm'][noise][K] = res_vdm[0] if res_vdm else None

            print("-" * 155)
            gc.collect()

    # Save experimental data to disk
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/simexp2_klein_24d.npz', **saved_results)
    print("\n Complete simulation data saved to Results/simexp2_klein_24d.npz")


if __name__ == "__main__":
    t_global = time.time()
    run_klein_bottle_test()
    print(f"\n Klein Bottle positive control test completed! Total elapsed time: {time.time() - t_global:.1f}s")