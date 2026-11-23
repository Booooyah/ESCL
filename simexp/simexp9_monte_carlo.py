"""
simexp/simexp9_monte_carlo.py
----------------------------------
Rigorous Monte Carlo Statistical Evaluation (JASA/AOAS Standard).
Calculates Empirical Size (Type I Error) on the Torus (H0)
and Empirical Power on the Klein Bottle (H1) over independent trials.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
import concurrent.futures

# 锁定底层计算库的多线程，防止多进程(ProcessPool)嵌套时导致的 CPU/内存死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    TwoStageBootstrapTester,
    ESCL_H1_Analyzer,
    VDM_H1_Analyzer,
    auto_select_optimal_k
)


# =====================================================================
# 1. Manifold Generators (Strictly Uniform Measure)
# =====================================================================
def generate_flat_torus(N=1500):
    np.random.seed(42)
    u = np.random.rand(N) * 2 * np.pi
    v = np.random.rand(N) * 2 * np.pi
    X_4d = np.vstack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)]).T
    return np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')


def generate_klein_bottle(N=1500):
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
        valid = w_cand <= J_val
        u_list.extend(u_cand[valid])
        v_list.extend(v_cand[valid])
    u, v = np.array(u_list[:N]), np.array(v_list[:N])

    x1 = (R + r * np.cos(v)) * np.cos(u)
    x2 = (R + r * np.cos(v)) * np.sin(u)
    x3 = r * np.sin(v) * np.cos(u / 2)
    x4 = r * np.sin(v) * np.sin(u / 2)
    X_4d = np.vstack([x1, x2, x3, x4]).T
    return np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')


def apply_noise_and_rotate(X_base, noise, seed):
    np.random.seed(seed)
    X_noisy = X_base + np.random.normal(0, noise, size=X_base.shape) if noise > 0 else X_base
    return np.dot(X_noisy, ortho_group.rvs(X_base.shape[1], random_state=seed))


# =====================================================================
# 2. Worker Function (For Multiprocessing)
# =====================================================================
def mc_worker(args):
    trial_idx, noise, k_opt_t, k_opt_k, N_points, d_prior, B_BOOTSTRAP = args

    # 独立的随机种子，确保无相关性
    seed_t = int(noise * 1000) + trial_idx * 7
    seed_k = int(noise * 1000) + trial_idx * 7 + 10000

    X_base_t = generate_flat_torus(N_points)
    X_base_k = generate_klein_bottle(N_points)

    X_t = apply_noise_and_rotate(X_base_t, noise, seed_t)
    X_k = apply_noise_and_rotate(X_base_k, noise, seed_k)

    # 强制子线程内部不再开启多线程，避免死锁
    tester_escl_t = TwoStageBootstrapTester(
        builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_t, 'n_jobs': 1},
        analyzer_class=ESCL_H1_Analyzer, analyzer_kwargs={'beta': 5.0, 'auto_beta': True, 'cutoff': 'auto'},
        b_bootstrap=B_BOOTSTRAP, max_workers=1
    )
    tester_vdm_t = TwoStageBootstrapTester(
        builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_t, 'n_jobs': 1},
        analyzer_class=VDM_H1_Analyzer, analyzer_kwargs={'epsilon_scale': 1.0},
        b_bootstrap=60, max_workers=1
    )
    tester_escl_k = TwoStageBootstrapTester(
        builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_k, 'n_jobs': 1},
        analyzer_class=ESCL_H1_Analyzer, analyzer_kwargs={'beta': 5.0, 'auto_beta': True, 'cutoff': 'auto'},
        b_bootstrap=B_BOOTSTRAP, max_workers=1
    )
    tester_vdm_k = TwoStageBootstrapTester(
        builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_k, 'n_jobs': 1},
        analyzer_class=VDM_H1_Analyzer, analyzer_kwargs={'epsilon_scale': 1.0},
        b_bootstrap=60, max_workers=1
    )

    _, _, res_e_t = tester_escl_t.test(X_t)
    _, _, res_v_t = tester_vdm_t.test(X_t)
    _, _, res_e_k = tester_escl_k.test(X_k)
    _, _, res_v_k = tester_vdm_k.test(X_k)

    stat_e_t = res_e_t[0].get('status', '') if res_e_t else ''
    stat_v_t = res_v_t[0].get('status', '') if res_v_t else ''
    stat_e_k = res_e_k[0].get('status', '') if res_e_k else ''
    stat_v_k = res_v_k[0].get('status', '') if res_v_k else ''

    # 输出结果字典，顺便打包最核心的代数统计量 (为画敏感性分析图存数据)
    return {
        'escl_t_twist': 1 if "Twisted" in stat_e_t else 0,
        'vdm_t_twist': 1 if "Twisted" in stat_v_t else 0,
        'escl_k_twist': 1 if "Twisted" in stat_e_k else 0,
        'vdm_k_twist': 1 if "Twisted" in stat_v_k else 0,
        'escl_t_inc': 1 if "Degenerate" in stat_e_t else 0,
        'vdm_t_inc': 1 if "Degenerate" in stat_v_t else 0,

        'L_rho_t': res_e_t[0].get('fiedler_ci_lower', 0) if res_e_t else 0,
        'C_m_t': res_e_t[0].get('chaos_ci_median', 1) if res_e_t else 1,
        'L_rho_k': res_e_k[0].get('fiedler_ci_lower', 0) if res_e_k else 0,
        'C_m_k': res_e_k[0].get('chaos_ci_median', 1) if res_e_k else 1,
    }


# =====================================================================
# 3. Main Orchestrator
# =====================================================================
def run_monte_carlo():
    FAST_TEST = False  # <--- 测试时保持 True，正式跑论文数据时改为 False

    if FAST_TEST:
        print("⚠️ [WARNING] Running in FAST_TEST Mode (Quick Preview)")
        N_TRIALS = 12  # 仅测 12 次
        B_BOOTSTRAP = 20  # 降低 MCMC 采样
        N_POINTS = 1000  # 缩小点云规模
        noise_levels = [0.0, 0.40]
    else:
        print("🚀 [INFO] Running FULL Publication Mode (Go grab a coffee...)")
        N_TRIALS = 500  # JASA/AOAS 黄金标准：500次
        B_BOOTSTRAP = 60
        N_POINTS = 1500
        noise_levels = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    d_prior = 2
    max_cpu = max(1, os.cpu_count() - 1)

    mc_results = {
        'Noise': [], 'ESCL_Size': [], 'VDM_Size': [], 'ESCL_Power': [], 'VDM_Power': []
    }
    raw_logs = []

    print("\n" + "=" * 90)
    print(f"🎲 Monte Carlo Statistical Evaluation (N={N_TRIALS} trials per noise level)")
    print("=" * 90)

    for noise in noise_levels:
        print(f"\n[ Noise Level: sigma={noise:.2f} | Executing {N_TRIALS} Replications ]")

        # Step 1: 使用伪样本确定最优 K
        dummy_t = apply_noise_and_rotate(generate_flat_torus(N_POINTS), noise, 999)
        dummy_k = apply_noise_and_rotate(generate_klein_bottle(N_POINTS), noise, 888)

        k_opt_t = auto_select_optimal_k(dummy_t, k_range=np.arange(20, 70, 10), intrinsic_dim=d_prior, n_jobs=-1)
        k_opt_k = auto_select_optimal_k(dummy_k, k_range=np.arange(20, 70, 10), intrinsic_dim=d_prior, n_jobs=-1)
        print(f"  -> Frozen K locked for current noise: Torus(K={k_opt_t}), Klein(K={k_opt_k})")

        # Step 2: 多进程并发执行 MC 核心推断
        tasks = [(i, noise, k_opt_t, k_opt_k, N_POINTS, d_prior, B_BOOTSTRAP) for i in range(N_TRIALS)]
        agg = {'escl_t_tw': 0, 'vdm_t_tw': 0, 'escl_k_tw': 0, 'vdm_k_tw': 0}

        t_start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpu) as executor:
            completed = 0
            for out in executor.map(mc_worker, tasks):
                if out:
                    agg['escl_t_tw'] += out['escl_t_twist']
                    agg['vdm_t_tw'] += out['vdm_t_twist']
                    agg['escl_k_tw'] += out['escl_k_twist']
                    agg['vdm_k_tw'] += out['vdm_k_twist']

                    raw_logs.append({
                        'noise': noise, 'trial': completed,
                        'L_rho_t': out['L_rho_t'], 'C_m_t': out['C_m_t'],
                        'L_rho_k': out['L_rho_k'], 'C_m_k': out['C_m_k'],
                    })

                completed += 1
                if completed % max(1, (N_TRIALS // 20)) == 0 or completed == N_TRIALS:
                    sys.stdout.write(f"\r  -> Progress: {completed}/{N_TRIALS} trials completed.")
                    sys.stdout.flush()

        print(f"\n  -> Elapsed: {time.time() - t_start:.1f}s")

        mc_results['Noise'].append(noise)
        mc_results['ESCL_Size'].append(agg['escl_t_tw'] / N_TRIALS)
        mc_results['VDM_Size'].append(agg['vdm_t_tw'] / N_TRIALS)
        mc_results['ESCL_Power'].append(agg['escl_k_tw'] / N_TRIALS)
        mc_results['VDM_Power'].append(agg['vdm_k_tw'] / N_TRIALS)

    # =====================================================================
    # 3. Print Final Publication Table
    # =====================================================================
    print("\n" + "=" * 80)
    print("📈 FINAL MONTE CARLO RESULTS (Rejection Frequencies)")
    print("=" * 80)

    df = pd.DataFrame(mc_results)
    for col in ['ESCL_Size', 'VDM_Size', 'ESCL_Power', 'VDM_Power']:
        df[col] = df[col].apply(lambda x: f"{x * 100:.1f}%")

    print(df.to_string(index=False))
    print("=" * 80)

    os.makedirs('Results', exist_ok=True)
    df.to_csv('Results/simexp9_monte_carlo_table.csv', index=False)
    pd.DataFrame(raw_logs).to_csv('Results/simexp9_raw_logs.csv', index=False)

    print("\n✅ Results saved to Results/simexp9_monte_carlo_table.csv")
    print("✅ Raw logs saved to Results/simexp9_raw_logs.csv (Ready for Sensitivity Analysis!)")


if __name__ == "__main__":
    t_global = time.time()
    run_monte_carlo()
    print(f"\n🎉 Total Monte Carlo execution time: {(time.time() - t_global) / 60:.1f} minutes")