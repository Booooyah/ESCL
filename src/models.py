"""
src/models.py
-------------
Core algorithmic implementation of the Empirical Spectral Cohomology Learning (ESCL) framework.
Includes the Vector Diffusion Maps (VDM) baseline, Two-Stage Bootstrap Testing,
and the 2D Topological Phase Space decision tree.
"""

import warnings
import concurrent.futures
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh
from numba import njit

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# PART 1: Utilities & Auto-Parameter Selection
# =============================================================================

def estimate_intrinsic_dim(X, k_est=20):
    """
    Estimates the intrinsic dimensionality (d) of the underlying manifold
    using maximum likelihood estimation on nearest neighbor distances.
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k_est + 1, algorithm='ball_tree', n_jobs=-1).fit(X)
    distances, _ = nbrs.kneighbors(X)

    distances = np.maximum(distances[:, 1:], 1e-10)
    R_k = distances[:, -1][:, np.newaxis]
    log_ratio = np.log(R_k / distances[:, :-1])
    d_local = (k_est - 1) / np.sum(log_ratio, axis=1)

    return max(2, int(np.round(np.mean(d_local))))


def auto_select_cutoff(E, edges, target_ratio=0.98):
    """
    Implementation of the Data-Driven Topological Surgery (Section 2.3).
    Adaptively determines the dynamic cutoff threshold to prune geometrically gasified edges,
    immunizing the estimator against completely shattered local neighborhoods.
    """
    if not edges: return 0.0
    edge_arr = np.array(list(edges))
    c_values = np.zeros(len(edge_arr))

    for idx, (i, j) in enumerate(edge_arr):
        C_ij = E[i].T @ E[j]
        _, S, _ = np.linalg.svd(C_ij)
        # Compute alignment confidence c_ij (Eq. 2)
        c_values[idx] = np.exp(np.mean(np.log(np.clip(S, 1e-15, 1.0))))

    # Fail-safe 1: Detect topological shattering (Global median confidence < 0.50)
    if np.median(c_values) < 0.50:
        return -1.0

    noise_floor = np.percentile(c_values, 5)

    # Fail-safe 2: Impose a hard lower bound to prevent extreme tangent space misalignment
    return float(np.clip(noise_floor, 0.40, 0.60))


def auto_select_optimal_k(X, k_range, intrinsic_dim=2, mutual_knn=False, n_jobs=-1):
    """
    Evaluates the thermodynamic threshold for graph connectivity (Condition C4).
    Scans the K parameter space to balance the deterministic secant plane bias
    and the ambient noise variance.
    """
    min_required_k = intrinsic_dim * 5
    valid_k_range = [k for k in k_range if k >= min_required_k]

    if not valid_k_range:
        return max(k_range[-1], min_required_k)

    valid_candidates = []

    for k in valid_k_range:
        builder = TopologyGraphBuilder(intrinsic_dim=intrinsic_dim, k_neighbors=k, mutual_knn=mutual_knn, n_jobs=n_jobs)
        E, edges, comps = builder.build(X, use_cknn=False)

        if not comps: continue

        best_cutoff = auto_select_cutoff(E, edges)
        edge_arr = np.array(list(edges))
        if len(edge_arr) == 0: continue

        c_values = np.array([np.exp(np.mean(np.log(np.clip(np.linalg.svd(E[i].T @ E[j])[1], 1e-15, 1.0))))
                             for i, j in edge_arr])

        valid_mask = c_values >= best_cutoff
        if not np.any(valid_mask): continue

        valid_edges = edge_arr[valid_mask]
        vals = np.ones(len(valid_edges) * 2)
        row_idx = np.concatenate([valid_edges[:, 0], valid_edges[:, 1]])
        col_idx = np.concatenate([valid_edges[:, 1], valid_edges[:, 0]])

        A_bool = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(X.shape[0], X.shape[0]))
        _, labels = connected_components(csgraph=A_bool, directed=False)
        max_component_ratio = np.max(np.bincount(labels)) / X.shape[0]

        confidences = np.mean(c_values[valid_mask])

        # Verify that the macroscopic geometric structure remains connected
        if max_component_ratio >= 0.90:
            valid_candidates.append((k, confidences))

    if valid_candidates:
        robust_candidates = [x for x in valid_candidates if x[1] >= 0.30]
        if not robust_candidates:
            return valid_candidates[0][0]
        return robust_candidates[int(len(robust_candidates) * 0.50)][0]

    return valid_k_range[-1]


# =============================================================================
# PART 2: Graph Builder & Core Solvers
# =============================================================================

class TopologyGraphBuilder:
    """
    Constructs the empirical K-NN graph and estimates the d-dimensional
    tangent space basis via local Principal Component Analysis (PCA).
    """

    def __init__(self, intrinsic_dim=2, k_neighbors=40, mutual_knn=False, use_epsilon_bound=False, n_jobs=-1):
        self.intrinsic_dim = intrinsic_dim
        self.k_neighbors = k_neighbors
        self.mutual_knn = mutual_knn
        self.use_epsilon_bound = use_epsilon_bound
        self.n_jobs = n_jobs

    def build(self, X, use_cknn=False, delta=1.0):
        n, p = X.shape
        search_k = self.k_neighbors * 3 if use_cknn else self.k_neighbors
        nbrs = NearestNeighbors(n_neighbors=search_k, algorithm='ball_tree', n_jobs=self.n_jobs).fit(X)
        distances, indices = nbrs.kneighbors(X)

        E = np.zeros((n, p, self.intrinsic_dim))
        for i in range(n):
            X_local = X[indices[i, :self.k_neighbors]]
            X_centered = X_local - np.mean(X_local, axis=0)
            # Extrinsic tangent space estimation via local PCA
            _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
            E[i] = Vt[:self.intrinsic_dim, :].T

        edges = set()
        search_indices = indices[:, :self.k_neighbors]
        if self.mutual_knn:
            for i in range(n):
                for j in search_indices[i]:
                    if i != j and i in search_indices[j]:
                        edges.add(tuple(sorted((i, j))))
        else:
            for i in range(n):
                for j in search_indices[i]:
                    if i != j: edges.add(tuple(sorted((i, j))))

        if not edges: return E, edges, []

        row_idx = [e[0] for e in edges] + [e[1] for e in edges]
        col_idx = [e[1] for e in edges] + [e[0] for e in edges]
        vals = np.ones(len(row_idx))
        A_bool = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(n, n)).tocsr()

        _, labels = connected_components(csgraph=A_bool, directed=False)
        unique_labels, counts = np.unique(labels, return_counts=True)

        valid_threshold = max(10, self.k_neighbors // 2)
        valid_labels = unique_labels[counts >= valid_threshold]
        components = [np.where(labels == label)[0] for label in valid_labels]

        return E, edges, components


def _compute_spectra_from_coo(row_idx, col_idx, values, is_null_model=False):
    """
    Computes the fundamental energy (Fiedler root, lambda_1) and macroscopic
    algebraic connectivity (lambda_2) of the normalized Sheaf Laplacian (Eq. 3).
    """
    if not row_idx: return 0.0, 0.0, None, None, False
    size_c = max(max(row_idx), max(col_idx)) + 1
    A_conn = sparse.coo_matrix((values, (row_idx, col_idx)), shape=(size_c, size_c)).tocsr()

    n_comps, labels = connected_components(csgraph=np.abs(A_conn), directed=False)
    if n_comps > 1:
        main_label = np.argmax(np.bincount(labels))
        mask = (labels == main_label)
        A_sub = A_conn[mask, :][:, mask]
    else:
        mask = np.ones(size_c, dtype=bool)
        A_sub = A_conn

    D_sub_diag = np.array(np.abs(A_sub).sum(axis=1)).flatten()
    D_sub_inv_sqrt = np.zeros_like(D_sub_diag)
    valid_mask = D_sub_diag > 1e-12
    D_sub_inv_sqrt[valid_mask] = 1.0 / np.sqrt(D_sub_diag[valid_mask])

    D_sub_mat = sparse.diags(D_sub_inv_sqrt)
    L_sub_norm = sparse.eye(A_sub.shape[0]) - D_sub_mat @ A_sub @ D_sub_mat
    L_sub_sym = (L_sub_norm + L_sub_norm.T) / 2.0

    eig_tol = 1e-2 if is_null_model else 1e-5

    try:
        evals_conn, evecs_conn = eigsh(L_sub_sym, k=2, which='SA', tol=eig_tol)
        idx = np.argsort(evals_conn)
        evals_conn = evals_conn[idx]

        lam1_conn = max(0.0, float(evals_conn[0]))
        lam2_conn = max(0.0, float(evals_conn[1]))

        f1_sub = evecs_conn[:, idx[0]]
        f1 = np.zeros(size_c)
        f1[mask] = f1_sub
        is_degenerate = (lam2_conn < 1e-7)

        return lam1_conn, lam2_conn, L_sub_sym, f1, is_degenerate
    except Exception:
        return 0.0, 0.0, None, None, False


# =============================================================================
# PART 3: Topological Analyzers (ESCL & VDM)
# =============================================================================

class ESCL_H1_Analyzer:
    """
    Empirical Spectral Cohomology Learning (ESCL):
    Elevates the continuous gauge to a discrete Z2 orientation sheaf to establish
    a strictly positive topological mass gap.
    """

    def __init__(self, beta=5.0, cutoff=0.0, w_min=0.0, auto_beta=True):
        self.beta = beta
        self.cutoff = cutoff
        self.w_min = w_min
        self.auto_beta = auto_beta

    @staticmethod
    def _compute_adaptive_beta(E, valid_edges, comp_set, fallback_beta):
        errors = [1.0 - np.mean(np.linalg.svd(E[i].T @ E[j])[1])
                  for i, j in valid_edges if i in comp_set and j in comp_set]
        if not errors: return fallback_beta
        D_median = np.median(errors)
        return np.log(2) / max(D_median, 0.02) if max(D_median, 0.02) >= 1e-10 else fallback_beta

    def analyze(self, E, valid_edges, comp_indices, X=None):
        size_c = len(comp_indices)
        if size_c < 3: return 0.0, 0.0, None, None, None

        global_to_local = {g: l for l, g in enumerate(comp_indices)}
        comp_set = set(comp_indices)
        current_beta = self._compute_adaptive_beta(E, valid_edges, comp_set, self.beta) if self.auto_beta else self.beta

        row_idx, col_idx, values, w_list, ortho_collisions = [], [], [], [], 0

        for i, j in valid_edges:
            if i in comp_set and j in comp_set:
                C_ij = E[i].T @ E[j]
                U, S, Vt = np.linalg.svd(C_ij)

                conf = np.exp(np.mean(np.log(np.clip(S, 1e-15, 1.0))))
                local_sigma = 1.0 - conf

                if conf < self.cutoff: continue

                # The Discrete Homomorphic Projection (Section 2.3)
                # Determines the Z2 gauge assignment acting as an algebraic shock absorber
                z_ij = np.sign(np.linalg.det(U @ Vt))
                if z_ij == 0:
                    z_ij = 1.0
                    ortho_collisions += 1

                w_ij = np.exp(-current_beta * local_sigma)
                if self.w_min > 0 and w_ij < self.w_min: continue

                u, v = global_to_local[i], global_to_local[j]
                row_idx.extend([u, v])
                col_idx.extend([v, u])
                values.extend([w_ij * z_ij, w_ij * z_ij])
                w_list.extend([w_ij, w_ij])

        if not row_idx: return 0.0, 0.0, 0.0, None, None

        lam1_conn, lam2_conn, L_conn_sym, f1, is_degenerate = _compute_spectra_from_coo(row_idx, col_idx, values)
        weights_array = np.abs(values)
        cv = np.std(weights_array) / np.mean(weights_array) if np.mean(weights_array) > 0 else 0.0

        surrogate_data = {
            'type': 'ESCL', 'size_mat': size_c, 'd': E.shape[2],
            'row_idx': row_idx, 'col_idx': col_idx, 'w_data': w_list,
            'ortho_collisions': ortho_collisions, 'is_degenerate': is_degenerate,
            'values': values, 'f1_field': f1
        }
        return lam1_conn, lam2_conn, cv, L_conn_sym, surrogate_data


class VDM_H1_Analyzer:
    """
    Vector Diffusion Maps (VDM): Baseline continuous O(d) gauge representation.
    Subject to asymptotic inconsistency due to continuous metric leakage (Theorem 1).
    """

    def __init__(self, epsilon_scale=1.0):
        self.epsilon_scale = epsilon_scale

    def analyze(self, E, valid_edges, comp_indices, X=None):
        size_c = len(comp_indices)
        if size_c < 3: return 0.0, 0.0, None, None, None

        d = E.shape[2]
        global_to_local = {g: l for l, g in enumerate(comp_indices)}
        comp_set = set(comp_indices)

        squared_dists = [np.sum((X[i] - X[j]) ** 2) for i, j in valid_edges if i in comp_set and j in comp_set]
        if not squared_dists: return 0.0, 0.0, None, None, None
        epsilon = np.median(squared_dists) * self.epsilon_scale + 1e-10

        row_idx, col_idx, values_A = [], [], []
        u_list, v_list, w_list, degrees = [], [], [], np.zeros(size_c)

        for i, j in valid_edges:
            if i in comp_set and j in comp_set:
                u, v = global_to_local[i], global_to_local[j]
                w_ij = np.exp(-np.sum((X[i] - X[j]) ** 2) / epsilon)

                u_list.append(u);
                v_list.append(v);
                w_list.append(w_ij)

                U, _, Vt = np.linalg.svd(E[i].T @ E[j])

                # Analog estimator lacking a truncation mechanism
                O_ij, O_ji = U @ Vt, (U @ Vt).T

                for r in range(d):
                    for c in range(d):
                        row_idx.append(u * d + r);
                        col_idx.append(v * d + c);
                        values_A.append(w_ij * O_ij[r, c])
                        row_idx.append(v * d + r);
                        col_idx.append(u * d + c);
                        values_A.append(w_ij * O_ji[r, c])

                degrees[u] += w_ij
                degrees[v] += w_ij

        if not row_idx: return 0.0, 0.0, 0.0, None, None

        A_conn = sparse.coo_matrix((values_A, (row_idx, col_idx)), shape=(size_c * d, size_c * d)).tocsr()
        D_diag = np.repeat(degrees, d)
        D_inv_sqrt = np.zeros_like(D_diag)
        D_inv_sqrt[D_diag > 1e-12] = 1.0 / np.sqrt(D_diag[D_diag > 1e-12])
        D_inv_sqrt_mat = sparse.diags(D_inv_sqrt)

        L_conn_norm = sparse.eye(size_c * d) - D_inv_sqrt_mat @ A_conn @ D_inv_sqrt_mat
        L_conn_sym = (L_conn_norm + L_conn_norm.T) / 2.0

        try:
            evals, _ = eigsh(L_conn_sym, k=d + 1, which='SA', tol=1e-2, maxiter=500)
            evals = np.sort(evals)
            obs_lam1, obs_lam2 = max(0.0, float(evals[d - 1])), max(0.0, float(evals[d]))
        except Exception:
            return np.nan, np.nan, 0.0, None, None

        cv = np.std(w_list) / np.mean(w_list) if np.mean(w_list) > 0 else 0.0
        surrogate_data = {
            'type': 'VDM', 'size_mat': size_c * d, 'd': d,
            'row_idx': row_idx, 'col_idx': col_idx,
            'w_data': w_list, 'deg_data': degrees, 'values': values_A,
            'u_list': u_list, 'v_list': v_list
        }
        return obs_lam1, obs_lam2, cv, L_conn_sym, surrogate_data


# =============================================================================
# PART 4: Statistical Inference & Resampling Workers
# =============================================================================

@njit(nogil=True)
def _njit_vdm_noise(w_list, d, seed_val):
    """JIT-compiled generation of continuous orthogonal O(d) noise for VDM null bath."""
    np.random.seed(seed_val)
    n_edges = len(w_list)
    values_null = np.empty(n_edges * d * d * 2, dtype=np.float64)
    idx_out = 0

    for i in range(n_edges):
        w_ij = w_list[i]
        noise_matrix = np.random.normal(0.0, 1.0, size=(d, d))
        Q, R = np.linalg.qr(noise_matrix)
        d_sign = np.sign(np.diag(R))
        for j in range(d):
            if d_sign[j] == 0: d_sign[j] = 1.0

        O_null = Q * d_sign
        O_null_ji = O_null.T

        for r in range(d):
            for c in range(d):
                values_null[idx_out] = w_ij * O_null[r, c]
                values_null[idx_out + 1] = w_ij * O_null_ji[r, c]
                idx_out += 2
    return values_null


def _gauge_randomization_worker(surrogate_data):
    """
    Phase I: Microcanonical Thermodynamic Null Bath (Section 2.4).
    Randomizes the discrete gauge assignments to verify if the observed signal
    is statistically distinguishable from pure topological chaos.
    """
    rng = np.random.default_rng()
    a_type = surrogate_data['type']

    if a_type == 'ESCL':
        w_unique = np.array(surrogate_data['w_data'], dtype=np.float64)[0::2]
        z_random = rng.choice(np.array([-1.0, 1.0]), size=len(w_unique))
        values_null = np.empty(len(w_unique) * 2, dtype=np.float64)
        values_null[0::2] = values_null[1::2] = w_unique * z_random

        lam1_null, _, _, _, _ = _compute_spectra_from_coo(
            surrogate_data['row_idx'], surrogate_data['col_idx'], values_null, is_null_model=True)
        return lam1_null

    elif a_type == 'VDM':
        d = surrogate_data['d']
        values_null = _njit_vdm_noise(np.array(surrogate_data['w_data'], dtype=np.float64), d,
                                      rng.integers(0, 2 ** 31 - 1))
        A_conn = sparse.coo_matrix((values_null, (surrogate_data['row_idx'], surrogate_data['col_idx'])),
                                   shape=(surrogate_data['size_mat'], surrogate_data['size_mat'])).tocsr()

        D_diag = np.repeat(np.array(surrogate_data['deg_data'], dtype=np.float64), d)
        D_inv_sqrt = np.zeros_like(D_diag)
        D_inv_sqrt[D_diag > 1e-12] = 1.0 / np.sqrt(D_diag[D_diag > 1e-12])
        L_sym = sparse.eye(surrogate_data['size_mat']) - sparse.diags(D_inv_sqrt) @ A_conn @ sparse.diags(D_inv_sqrt)

        try:
            evals, _ = eigsh(L_sym, k=d, which='SA', tol=1e-2, maxiter=500)
            return max(0.0, float(evals[d - 1]))
        except Exception:
            return 0.0


def _node_subsampling_worker(surrogate_data, keep_ratio=0.95, mu_null=1.0):
    """
    Phase II: Node Subsampling to evaluate macroscopic frustration energy.
    Constructs robust Confidence Intervals while preserving intrinsic spatial
    dependencies and Cheeger bottlenecks of the manifold.
    """
    rng = np.random.default_rng()
    a_type = surrogate_data['type']
    size_mat = surrogate_data['size_mat']
    d = surrogate_data['d']

    orig_n = size_mat if a_type == 'ESCL' else size_mat // d
    node_mask = rng.random(orig_n) < keep_ratio

    if np.sum(node_mask) < 0.5 * orig_n: return np.nan, np.nan, np.nan

    row_idx = np.array(surrogate_data['row_idx'])
    col_idx = np.array(surrogate_data['col_idx'])
    values = np.array(surrogate_data['values'])

    if a_type == 'ESCL':
        edge_mask = node_mask[row_idx] & node_mask[col_idx]
        if not np.any(edge_mask): return np.nan, np.nan, np.nan

        obs_lam1, obs_lam2, _, _, _ = _compute_spectra_from_coo(
            row_idx[edge_mask].tolist(), col_idx[edge_mask].tolist(), values[edge_mask].tolist(), is_null_model=False)

    else:
        u_list = np.array(surrogate_data['u_list'])
        v_list = np.array(surrogate_data['v_list'])
        edge_mask_base = node_mask[u_list] & node_mask[v_list]
        if not np.any(edge_mask_base): return np.nan, np.nan, np.nan

        full_mask = node_mask[row_idx // d] & node_mask[col_idx // d]
        sub_row, sub_col, sub_vals = row_idx[full_mask], col_idx[full_mask], values[full_mask]

        vals_bool = np.ones(np.sum(edge_mask_base) * 2)
        A_bool = sparse.coo_matrix((vals_bool, (np.concatenate([u_list[edge_mask_base], v_list[edge_mask_base]]),
                                                np.concatenate([v_list[edge_mask_base], u_list[edge_mask_base]]))),
                                   shape=(orig_n, orig_n))

        _, labels = connected_components(csgraph=A_bool, directed=False)
        main_label = np.argmax(np.bincount(labels))
        lcc_mask = (labels == main_label)

        sub_w = np.array(surrogate_data['w_data'])[edge_mask_base]
        degrees_sub = (np.bincount(u_list[edge_mask_base], weights=sub_w, minlength=orig_n) +
                       np.bincount(v_list[edge_mask_base], weights=sub_w, minlength=orig_n))[lcc_mask]

        D_diag_sub = np.repeat(degrees_sub, d)
        D_inv_sqrt_sub = np.zeros_like(D_diag_sub)
        D_inv_sqrt_sub[D_diag_sub > 1e-12] = 1.0 / np.sqrt(D_diag_sub[D_diag_sub > 1e-12])

        A_conn_sub = sparse.coo_matrix((sub_vals, (sub_row, sub_col)), shape=(size_mat, size_mat)).tocsr()
        lcc_mask_full = np.repeat(lcc_mask, d)
        A_conn_sub = A_conn_sub[lcc_mask_full, :][:, lcc_mask_full]

        L_norm_sub = sparse.eye(A_conn_sub.shape[0]) - sparse.diags(D_inv_sqrt_sub) @ A_conn_sub @ sparse.diags(
            D_inv_sqrt_sub)
        L_sym_sub = (L_norm_sub + L_norm_sub.T) / 2.0

        try:
            evals, _ = eigsh(L_sym_sub, k=d + 1, which='SA', tol=1e-2, maxiter=200)
            evals = np.sort(evals)
            obs_lam1, obs_lam2 = max(0.0, float(evals[d - 1])), max(0.0, float(evals[d]))
        except Exception:
            return np.nan, np.nan, np.nan

    if obs_lam2 is None or obs_lam2 < 1e-7: return np.nan, np.nan, np.nan
    return obs_lam1, obs_lam2, max(mu_null, 1e-8)


# =============================================================================
# PART 5: The Master Orchestrator (Topological Phase Space)
# =============================================================================

class TwoStageBootstrapTester:
    def __init__(self, builder_kwargs, analyzer_class, analyzer_kwargs, b_bootstrap=50, max_workers=4, alpha_mcmc=0.05,
                 solver_tol=1e-5):
        self.builder_kwargs = builder_kwargs
        self.analyzer_class = analyzer_class
        self.analyzer_kwargs = analyzer_kwargs
        self.b_bootstrap = b_bootstrap
        self.max_workers = max_workers
        self.alpha_mcmc = alpha_mcmc
        self.solver_tol = solver_tol

    def test(self, X_obs):
        n_samples = X_obs.shape[0]
        builder = TopologyGraphBuilder(**self.builder_kwargs)
        E_obs, edges_obs, comps_obs = builder.build(X_obs)

        def _fail_res(status):
            return [{
                'cv': 0.0, 'ratio': 0.0, 'obs_lam1': 0.0, 'obs_lam2': 0.0, 'p_noise': 1.0,
                'fiedler_ci_lower': 0.0, 'fiedler_ci_upper': 0.0,
                'chaos_ci_lower': 1.0, 'chaos_ci_median': 1.0, 'chaos_ci_upper': 1.0,
                'status': status, 'rho_nulls': [], 'is_degenerate': True,
                'ortho_collisions': 0, 'f1_field': None, 'main_comp_indices': None
            }]

        if not comps_obs:
            return 0, [], _fail_res("💥 Degenerate (Disconnected)")

        main_comp = max(comps_obs, key=len)
        if len(main_comp) < 0.50 * n_samples:
            return len(comps_obs), [len(main_comp)], _fail_res("💥 Degenerate (Percolation)")

        current_kwargs = self.analyzer_kwargs.copy()
        if current_kwargs.get('cutoff') == 'auto':
            best_cutoff = auto_select_cutoff(E_obs, edges_obs)
            if best_cutoff < 0.0:
                return len(comps_obs), [len(main_comp)], _fail_res("💥 Degenerate (Geometric Chaos)")
            current_kwargs['cutoff'] = best_cutoff

        analyzer = self.analyzer_class(**current_kwargs)
        rho_obs, lam2_obs, cv, _, surrogate_data = analyzer.analyze(E_obs, edges_obs, main_comp, X=X_obs)

        if rho_obs is None or np.isnan(rho_obs) or lam2_obs < 1e-5:
            return len(comps_obs), [len(main_comp)], _fail_res("💥 Degenerate (Solver/Disconnected)")

        frustration_ratio_obs = rho_obs / lam2_obs

        # --- STAGE 1: Gauge Randomization (MCMC Null) ---
        rho_nulls = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_gauge_randomization_worker, surrogate_data) for _ in range(self.b_bootstrap)]
            for f in concurrent.futures.as_completed(futures):
                rho_nulls.append(f.result())

        rho_nulls = np.array(rho_nulls)
        p_val_noise = (1.0 + np.sum(rho_nulls <= rho_obs)) / (1.0 + len(rho_nulls))
        mean_mu_null = np.mean(rho_nulls) if len(rho_nulls) > 0 else 1.0

        # --- STAGE 2: Node Subsampling (Phase Boundaries) ---
        keep_ratio = max(0.60, (n_samples ** 0.85) / n_samples)
        sub_ratios, sub_chaos_ratios = [], []

        # Incorporates the dimension-invariant scaling factor driven by the
        # BBP phase transition in Random Matrix Theory (Eq. 4)
        D_ambient = X_obs.shape[1]
        d_prior = self.builder_kwargs.get('intrinsic_dim', 2)
        dim_factor_ambient = np.sqrt(D_ambient / d_prior)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_node_subsampling_worker, surrogate_data, keep_ratio, mean_mu_null)
                       for _ in range(self.b_bootstrap)]
            for f in concurrent.futures.as_completed(futures):
                r_lam1, r_lam2, r_mu_null = f.result()
                if not np.isnan(r_lam1) and r_lam2 > 1e-7:
                    sub_ratios.append(r_lam1 / r_lam2)
                    sub_chaos_ratios.append((r_lam1 / max(r_mu_null, 1e-8)) * dim_factor_ambient)

        if len(sub_ratios) < 10:
            return len(comps_obs), [len(main_comp)], _fail_res("💥 Degenerate (Percolation)")

        sub_ratios, sub_chaos_ratios = np.array(sub_ratios), np.array(sub_chaos_ratios)
        f_l, f_u = np.percentile(sub_ratios, [5.0, 95.0])
        c_l, c_m, c_u = np.percentile(sub_chaos_ratios, [5.0, 50.0, 95.0])

        # --- The Topological Decision Tree (Section 2.4) ---
        if p_val_noise > self.alpha_mcmc:
            status = "💥 Degenerate (Pure Noise / Null Hypo)"
        elif f_l > 0.50:
            status = "🚨 Twisted (Macroscopic H1)"
        elif c_m > 0.25:
            status = "💥 Degenerate (High-Dim Gasification)"
        elif f_l > 0.01 or c_m > 0.01:
            status = "✅ Trivial (w/ Local Frustration)"
        else:
            status = "✨ Trivial (Perfectly Orientable)"

        results = [{
            'cv': cv, 'ratio': frustration_ratio_obs, 'obs_lam1': rho_obs, 'obs_lam2': lam2_obs,
            'p_noise': p_val_noise, 'fiedler_ci_lower': f_l, 'fiedler_ci_upper': f_u,
            'chaos_ci_lower': c_l, 'chaos_ci_median': c_m, 'chaos_ci_upper': c_u,
            'status': status, 'rho_nulls': rho_nulls.tolist(),
            'ortho_collisions': surrogate_data.get('ortho_collisions', 0),
            'is_degenerate': surrogate_data.get('is_degenerate', False),
            'f1_field': surrogate_data.get('f1_field', None),
            'main_comp_indices': main_comp
        }]
        return len(comps_obs), [len(main_comp)], results


def format_topological_verdict(algo_name, res_dict, exec_time, expected_topology=None, intrinsic_dim=2):
    """Formats the raw dictionary outputs into publication-ready logs."""
    if not res_dict: return f"{algo_name:<15} | 💥 No Result"

    r = res_dict[0]
    row = (f"{algo_name:<15} | {r.get('obs_lam1', 0.0):<8.4f} | {r.get('obs_lam2', 0.0):<8.4f} | "
           f"{r.get('p_noise', 1.0):<7.4f} | [{r.get('fiedler_ci_lower', 0.0):.3f}, {r.get('fiedler_ci_upper', 0.0):.3f}]".ljust(
        22) +
           f" | [{r.get('chaos_ci_lower', 0.0):.3f}, {r.get('chaos_ci_upper', 0.0):.3f}]".ljust(18) +
           f" | {r.get('status', 'Unknown'):<28} | {exec_time:>6.1f}s")
    return row