"""
realexp/realexp2_natural_image.py
-------------------------------------
Real-World Positive Control: The Klein Bottle of Natural Image Patches.
Demonstrating ESCL as an Unsupervised Topological Feature Extractor,
mapping abstract high-dimensional singularities back to physical pixel coordinates.
"""

import os
import gc
import sys
import time
import numpy as np
import scipy.datasets
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

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

# =============================================================================
# 1. Image Patch Extraction & Spatial Coordinate Tracking
# =============================================================================
def extract_natural_image_patches(max_samples=15000):
    print("\n" + "=" * 135)
    print("📸 [Data Preparation] Extracting Patches & Tracking Spatial Coordinates (Mode: Klein Bottle)")
    print("=" * 135)

    img = scipy.datasets.face(gray=True).astype(np.float32)
    H, W = img.shape
    shape = (H - 2, W - 2, 3, 3)
    strides = img.strides + img.strides
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides).reshape(-1, 9)
    coords = np.array([(i, j) for i in range(H - 2) for j in range(W - 2)])

    # Filter low-frequency homogeneous regions to isolate high-contrast textures
    variances = np.var(patches, axis=1)
    threshold = np.percentile(variances, 80)
    high_contrast_mask = variances >= threshold

    high_contrast = patches[high_contrast_mask]
    hc_coords = coords[high_contrast_mask]

    # Spherical normalization: mean-centering and contrast-normalization onto the unit hypersphere
    centered = high_contrast - np.mean(high_contrast, axis=1, keepdims=True)
    normalized = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8)

    np.random.seed(42)
    print("  -> Applying Core-Restricted Inverse Density Sampling...")
    sub_idx = np.random.choice(len(normalized), min(50000, len(normalized)), replace=False)
    sub_patches, sub_coords = normalized[sub_idx], hc_coords[sub_idx]

    nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree', n_jobs=-1).fit(sub_patches)
    distances, _ = nbrs.kneighbors(sub_patches)
    k_dists = distances[:, -1]

    # Strategic Filtering 1: Truncate the sparsest 30% outliers to guarantee manifold continuity
    core_threshold = np.percentile(k_dists, 70)
    core_mask = k_dists <= core_threshold
    core_patches, core_coords, core_dists = sub_patches[core_mask], sub_coords[core_mask], k_dists[core_mask]

    # Strategic Filtering 2: Inverse density sampling to elevate critical topological cross-caps
    # while suppressing heavily overrepresented primary bulk edges.
    probs = core_dists / np.sum(core_dists)
    actual_samples = min(max_samples, len(core_patches))
    final_idx = np.random.choice(len(core_patches), size=actual_samples, replace=False, p=probs)

    X_final, C_final = core_patches[final_idx], core_coords[final_idx]
    print(f"  -> Final Empirical Manifold: Shape {X_final.shape}")
    return X_final, C_final, img

# =============================================================================
# 2. Topological Inverse Mapping Renderer (Publication-Ready)
# =============================================================================
def visualize_dual_fi_field(X, coords, img_raw, res_escl):
    print("\n🎨 [Visualization] Rendering the Topological Inverse Mapping...")

    f1_val = res_escl.get('f1_field', None)
    indices = res_escl.get('main_comp_indices', None)

    if f1_val is None or indices is None:
        print("⚠️ Error: 'f1_field' data missing. Cannot render singularity map.")
        return

    # Reconstruct the global scalar frustration field (F_i)
    N_total = X.shape[0]
    F_field = np.zeros(N_total)
    f1_squared = np.abs(f1_val) ** 2
    f1_normalized = f1_squared / (np.max(f1_squared) + 1e-12)
    F_field[indices] = f1_normalized

    # Isolate the top 10% most frustrated points indicating geometric bottlenecks / cross-caps
    top_p = 90
    threshold = np.percentile(F_field[F_field > 0], top_p)
    bg_mask = F_field < threshold
    fg_mask = F_field >= threshold

    # Render the dual-panel composite figure
    plt.style.use('dark_background')  # Optimize contrast for inverse mapping
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

    # Panel A: High-Dimensional State Space (3D PCA)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # Enforce Z-order sorting to render highly frustrated singularities on the top layer
    F_fg = F_field[fg_mask]
    sort_idx = np.argsort(F_fg)
    X_pca_fg = X_pca[fg_mask][sort_idx]
    F_fg_sorted = F_fg[sort_idx]

    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.scatter(X_pca[bg_mask, 0], X_pca[bg_mask, 1], X_pca[bg_mask, 2],
               c='#34495E', s=10, alpha=0.3, edgecolors='none', label='Primary Bulk (Orientable Edges)')

    ax1.scatter(X_pca_fg[:, 0], X_pca_fg[:, 1], X_pca_fg[:, 2],
                c=F_fg_sorted, cmap='magma', s=40, alpha=1.0, edgecolors='white', linewidth=0.2,
                label='Topological Cross-caps (High $\mathcal{F}_i$)')

    ax1.set_title(f"(a) The High-Dimensional Topology\n(PCA 3D, Mode: KLEIN BOTTLE)", fontsize=16, fontweight='bold', color='white', pad=10)
    ax1.legend(loc='upper right', frameon=True, fontsize=12, facecolor='black', edgecolor='white', labelcolor='white')
    ax1.set_axis_off()

    # Panel B: Topological Inverse Mapping to Geometric Space
    ax2 = fig.add_subplot(gs[1])
    # Use the raw image as a semi-transparent background plate
    ax2.imshow(img_raw, cmap='gray', alpha=0.6)

    # Project the highly frustrated coordinates back onto the source image
    # Note: image coordinate system maps (col, row) to (x, y)
    high_fi_coords = coords[fg_mask][sort_idx]

    # Apply a +1 offset to align the coordinate with the physical center of the 3x3 patch
    ax2.scatter(high_fi_coords[:, 1] + 1, high_fi_coords[:, 0] + 1,
                c=F_fg_sorted, cmap='magma', s=10, alpha=0.9)

    ax2.set_title(f"(b) Topological Inverse Mapping to Geometric Space\n(Highlighted Top {100-top_p}% Frustration)", fontsize=16, fontweight='bold', color='white', pad=10)
    ax2.axis('off')

    plt.tight_layout()
    os.makedirs('Figs', exist_ok=True)
    plt.savefig('Figs/Fig5_Natural_Image_Mapping.pdf', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

# =============================================================================
# 3. Core Experimental Engine
# =============================================================================
def run_image_patches_experiment():
    try:
        X_target, C_target, img_raw = extract_natural_image_patches(max_samples=15000)
    except Exception as e:
        print(f"🚨 Data processing failed: {e}")
        return

    d_prior = 2
    print("\n" + "=" * 135)
    print("🧠 [Phase 2] Topological Inference on Natural Image Klein Bottle (H1)")
    print("=" * 135)

    print(f"  [Auto-Radar] Scanning optimal intrinsic scale...")
    t_auto_start = time.time()
    k_opt_dynamic = auto_select_optimal_k(X_target, k_range=np.arange(15, 55, 10), intrinsic_dim=d_prior, mutual_knn=False, n_jobs=-1)
    print(f"  [Auto-Config] Selected Optimal K={k_opt_dynamic} (Elapsed: {time.time() - t_auto_start:.2f}s)")
    print("-" * 135)

    header = f"{'Algorithm':<15} | {'Obs lambda1':<8} | {'Gap lambda2':<8} | {'P_noise':<7} | {'Fiedler CI (Ratio)':<22} | {'Chaos CI (Therm)':<18} | {'Verdict':<28} | {'Time(s)'}"
    print(header)
    print("-" * 135)

    tester_escl = TwoStageBootstrapTester(
        builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_dynamic, 'mutual_knn': False, 'n_jobs': 4},
        analyzer_class=ESCL_H1_Analyzer,
        analyzer_kwargs={'beta': 5.0, 'auto_beta': True, 'cutoff': 'auto'},
        b_bootstrap=100, max_workers=5, alpha_mcmc=0.05, solver_tol=1e-8
    )

    tester_vdm = TwoStageBootstrapTester(
        builder_kwargs={'intrinsic_dim': d_prior, 'k_neighbors': k_opt_dynamic, 'mutual_knn': False, 'n_jobs': 4},
        analyzer_class=VDM_H1_Analyzer,
        analyzer_kwargs={'epsilon_scale': 1.0},
        b_bootstrap=30, max_workers=2, alpha_mcmc=0.05, solver_tol=1e-8
    )

    t0 = time.time()
    _, _, res_escl = tester_escl.test(X_target)
    print(format_topological_verdict(f"K={k_opt_dynamic:<2} (ESCL)", res_escl, time.time() - t0))

    t1 = time.time()
    _, _, res_vdm = tester_vdm.test(X_target)
    print(format_topological_verdict(f"K={k_opt_dynamic:<2} (VDM)", res_vdm, time.time() - t1))
    print("-" * 135)

    # Save empirical data to disk
    os.makedirs('Results', exist_ok=True)
    np.savez_compressed('Results/realexp2_natural_image.npz',
                        res_escl=res_escl[0], res_vdm=res_vdm[0],
                        X_target=X_target, C_target=C_target, img_raw=img_raw)
    print(" Complete empirical data saved to Results/realexp2_natural_image.npz")

    # Render Topological Inverse Mapping
    if res_escl and 'status' in res_escl[0] and 'Error' not in res_escl[0]['status']:
        visualize_dual_fi_field(X_target, C_target, img_raw, res_escl[0])

if __name__ == "__main__":
    t_global = time.time()
    run_image_patches_experiment()
    print(f"\n Natural image topology experiment completed successfully in {time.time() - t_global:.1f}s")