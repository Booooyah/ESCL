"""
simexp/simexp6_ablation_cutoff.py
---------------------------------------
Executes a 2x2 factorial design proving that dynamic cutoff only fixes macroscopic
short-circuits, while Z2 quantization is strictly required to cure micro-metric leakage.
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

# Lock underlying threads to prevent memory explosion during numpy/scipy multiprocessing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Dynamically append the project root directory to the system path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TopologyGraphBuilder, ESCL_H1_Analyzer, auto_select_cutoff

# =====================================================================
# Specialized VDM Baseline: Mitigating eigenvalue splitting induced by
# symmetry breaking in random geometric graphs (Strict Control Group).
# =====================================================================
class Fixed_VDM_H1_Analyzer:
    def __init__(self, epsilon_scale=1.0):
        self.epsilon_scale = epsilon_scale

    def analyze(self, E, valid_edges, comp_indices, X=None):
        size_c = len(comp_indices)
        if size_c < 3: return np.nan, np.nan, None, None, None

        d = E.shape[2]
        global_to_local = {g: l for l, g in enumerate(comp_indices)}
        comp_set = set(comp_indices)

        squared_dists = [np.sum((X[i] - X[j]) ** 2) for i, j in valid_edges if i in comp_set and j in comp_set]
        if not squared_dists: return np.nan, np.nan, None, None, None
        epsilon = np.median(squared_dists) * self.epsilon_scale + 1e-10

        row_idx, col_idx, values_A = [], [], []
        degrees = np.zeros(size_c)

        for i, j in valid_edges:
            if i in comp_set and j in comp_set:
                u, v = global_to_local[i], global_to_local[j]
                w_ij = np.exp(-np.sum((X[i] - X[j]) ** 2) / epsilon)

                C_ij = E[i].T @ E[j]
                U, _, Vt = np.linalg.svd(C_ij)

                # Continuous orthogonal gauge assignment
                O_ij = U @ Vt
                O_ji = O_ij.T

                for r in range(d):
                    for c in range(d):
                        row_idx.append(u * d + r); col_idx.append(v * d + c); values_A.append(w_ij * O_ij[r, c])
                        row_idx.append(v * d + r); col_idx.append(u * d + c); values_A.append(w_ij * O_ji[r, c])

                degrees[u] += w_ij
                degrees[v] += w_ij

        A_conn = sparse.coo_matrix((values_A, (row_idx, col_idx)), shape=(size_c * d, size_c * d)).tocsr()
        D_diag = np.repeat(degrees, d)
        D_inv_sqrt = np.zeros_like(D_diag)
        mask = D_diag > 1e-12
        D_inv_sqrt[mask] = 1.0 / np.sqrt(D_diag[mask])
        D_inv_sqrt_mat = sparse.diags(D_inv_sqrt)

        L_conn_norm = sparse.eye(size_c * d) - D_inv_sqrt_mat @ A_conn @ D_inv_sqrt_mat
        L_conn_sym = (L_conn_norm + L_conn_norm.T) / 2.0

        try:
            evals, _ = eigsh(L_conn_sym, k=d + 1, which='SA', tol=1e-2, maxiter=200)
            evals = np.sort(evals)

            # Core Fix: Average the first d eigenvalues to offset random symmetry breaking,
            # ensuring a rigorous representation of the fundamental energy.
            obs_lam1 = max(0.0, float(np.mean(evals[:d])))
            obs_lam2 = max(0.0, float(evals[d]))
            return obs_lam1, obs_lam2, 0.0, None, None
        except Exception:
            return np.nan, np.nan, 0.0, None, None

# =====================================================================

def generate_flat_torus(N=1500, noise=0.0, seed=42):
    """Generates a 24D embedded Flat Torus for the orientable null hypothesis (H0)."""
    np.random.seed(seed)
    u = np.random.rand(N) * 2 * np.pi
    v = np.random.rand(N) * 2 * np.pi
    x1, x2 = np.cos(u), np.sin(u)
    x3, x4 = np.cos(v), np.sin(v)

    X_4d = np.vstack([x1, x2, x3, x4]).T
    X_base = np.pad(X_4d, ((0, 0), (0, 20)), mode='constant')
    X_noisy = X_base + np.random.normal(0, noise, size=X_base.shape) if noise > 0 else X_base
    return np.dot(X_noisy, ortho_group.rvs(24, random_state=seed))

def get_lcc(edges, n):
    """Extracts the indices belonging to the Largest Connected Component (LCC)."""
    if not edges: return []
    row = [e[0] for e in edges] + [e[1] for e in edges]
    col = [e[1] for e in edges] + [e[0] for e in edges]
    val = np.ones(len(row))
    A = sparse.coo_matrix((val, (row, col)), shape=(n, n))
    _, labels = connected_components(A, directed=False)
    counts = np.bincount(labels)
    return np.where(labels == np.argmax(counts))[0].tolist()

def run_ablation_experiment():
    noise_levels = np.linspace(0.0, 0.25, 6)
    N_points = 1500
    # Utilizing a large neighborhood parameter to resist 24D noise inflation
    K_neighbors = 40
    d_prior = 2

    results = {'Standard_VDM': [], 'VDM_Cutoff': [], 'ESCL_no_cutoff': [], 'ESCL_auto': []}

    print("\n" + "=" * 100)
    print("🔬 Running Ablation Study: Surgery vs. Quantization (Section 5.2)")
    print("=" * 100)

    for noise in noise_levels:
        print(f"\n[{'='*20} Ambient Gaussian Noise: sigma = {noise:.2f} {'='*20}]")
        X = generate_flat_torus(N=N_points, noise=noise)

        builder = TopologyGraphBuilder(intrinsic_dim=d_prior, k_neighbors=K_neighbors, n_jobs=-1)
        E, edges, comps = builder.build(X)
        if not comps:
            for k in results: results[k].append(np.nan)
            continue

        main_comp_orig = max(comps, key=len)

        # Model A: Standard VDM (Continuous O(d), No Surgery)
        vdm_std = Fixed_VDM_H1_Analyzer(epsilon_scale=1.0)
        lam1_v1, lam2_v1, _, _, _ = vdm_std.analyze(E, edges, main_comp_orig, X)
        rho_v1 = (lam1_v1 / lam2_v1) if (lam2_v1 and lam2_v1 > 1e-4) else np.nan
        results['Standard_VDM'].append(rho_v1)

        # Data-Driven Topological Surgery (Dynamic Cutoff)
        best_cutoff = auto_select_cutoff(E, edges, target_ratio=0.90)
        filtered_edges = set()
        for i, j in edges:
            C_ij = E[i].T @ E[j]
            _, S, _ = np.linalg.svd(C_ij)
            conf = np.exp(np.mean(np.log(np.clip(S, 1e-15, 1.0))))
            if conf >= best_cutoff:
                filtered_edges.add((i, j))

        main_comp_filtered = get_lcc(filtered_edges, N_points)

        # Model B: VDM + Cutoff (Continuous O(d), With Surgery)
        vdm_cut = Fixed_VDM_H1_Analyzer(epsilon_scale=1.0)
        lam1_v2, lam2_v2, _, _, _ = vdm_cut.analyze(E, filtered_edges, main_comp_filtered, X)
        rho_v2 = (lam1_v2 / lam2_v2) if (lam2_v2 and lam2_v2 > 1e-4) else np.nan
        results['VDM_Cutoff'].append(rho_v2)

        # Model C: ESCL without cutoff (Discrete Z2 Quantization only)
        escl = ESCL_H1_Analyzer(beta=5.0, auto_beta=True, cutoff=0.0)
        lam1_e, lam2_e, _, _, _ = escl.analyze(E, edges, main_comp_orig, X)
        rho_e = (lam1_e / lam2_e) if (lam2_e and lam2_e > 1e-4) else np.nan
        results['ESCL_no_cutoff'].append(rho_e)

        # Model D: ESCL with cutoff=auto (Complete ESCL Framework)
        escl_auto = ESCL_H1_Analyzer(beta=5.0, auto_beta=True, cutoff=best_cutoff)
        lam1_e_auto, lam2_e_auto, _, _, _ = escl_auto.analyze(E, edges, main_comp_orig, X)
        rho_e_auto = (lam1_e_auto / lam2_e_auto) if (lam2_e_auto and lam2_e_auto > 1e-4) else np.nan
        results['ESCL_auto'].append(rho_e_auto)

        print(f"  -> Model A (Standard VDM)         rho : {rho_v1:.4f}")
        print(f"  -> Model B (VDM + Cutoff)         rho : {rho_v2:.4f}")
        print(f"  -> Model C (ESCL, cutoff=0.0)     rho : {rho_e:.4f}")
        print(f"  -> Model D (ESCL, cutoff=auto)    rho : {rho_e_auto:.4f}")

    # Save data for publication plotting
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/simexp6_ablation_cutoff.npz', noises=noise_levels, **results)
    print("\n Complete simulation data saved to Results/simexp6_ablation_cutoff.npz")

    plot_ablation_results(noise_levels, results, N_points)


def plot_ablation_results(noise_levels, results, N):
    """Renders the 2x2 factorial ablation study curves."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))

    macro_threshold = 1.5 / np.log(N)

    # Plot the 4 ablation curves
    ax.plot(noise_levels, results['Standard_VDM'], marker='X', markersize=10,
            linewidth=2.5, color='#C0392B', linestyle=':', label='Standard VDM (Continuous, No Surgery)')

    ax.plot(noise_levels, results['VDM_Cutoff'], marker='^', markersize=10,
            linewidth=3, color='#E67E22', linestyle='--', label='VDM + Cutoff (Continuous, With Surgery)')

    ax.plot(noise_levels, results['ESCL_no_cutoff'], marker='o', markersize=10,
            linewidth=3.5, color='#2ECC71', linestyle=':', label=r'ESCL (Discrete $\mathbb{Z}_2$, No Surgery)')

    ax.plot(noise_levels, results['ESCL_auto'], marker='s', markersize=10,
            linewidth=3.5, color='#27AE60', linestyle='-', label=r'ESCL + Cutoff (Discrete $\mathbb{Z}_2$, With Surgery)')

    # Macro gap transition line
    ax.axhline(y=macro_threshold, color='black', linestyle='-.', linewidth=2,
               label=f'Macro Gap Threshold ({macro_threshold:.2f})')

    # Formatting and labels
    ax.set_xlabel(r'Ambient Gaussian Noise ($\sigma$)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Fiedler Ratio ($\rho = \lambda_1 / \lambda_2$)', fontsize=15, fontweight='bold')
    ax.set_title(r'Ablation Study: Topological Surgery vs. $\mathbb{Z}_2$ Quantization', fontsize=17, fontweight='bold', pad=15)

    ax.set_xlim(0.0, max(noise_levels))
    ax.tick_params(axis='both', which='major', labelsize=13)

    # Shade the hallucination hazard zone
    ax.axhspan(macro_threshold, 1.05, facecolor='#FDEDEC', alpha=0.5, zorder=0)
    ax.text(0.01, 0.85, 'Topological Hallucination Zone', color='#C0392B', fontsize=14, fontweight='bold', alpha=0.7)

    plt.ylim(-0.05, 1.35)

    # Position the legend in the upper left, arranged in two columns
    plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=2, framealpha=0.95, fontsize=11, edgecolor='black')

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/simexp6_ablation_cutoff.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print(" Plot successfully saved to 'Figs/simexp6_ablation_cutoff.pdf'")
    plt.show()


# Support plotting directly from cached .npz files
def reload_and_plot():
    try:
        data = np.load('Results/simexp6_ablation_cutoff.npz')
        noise_levels = data['noises']
        results = {k: data[k] for k in data.files if k != 'noises'}
        plot_ablation_results(noise_levels, results, N=1500)
    except FileNotFoundError:
        print(" Cached data not found. Please execute the computational engine first.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        reload_and_plot()
    else:
        t0 = time.time()
        run_ablation_experiment()
        print(f"\n Experiment 6 (Ablation) completed successfully in {time.time() - t0:.1f}s")