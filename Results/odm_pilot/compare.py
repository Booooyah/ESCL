"""Paired pilot of repository ESCL and a numerical ODM histogram rule."""
import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
if (HERE / "_deps").is_dir():
    sys.path.insert(0, str(HERE / "_deps"))

from src import models
from simexp.simexp9_monte_carlo import (
    apply_noise_and_rotate, generate_flat_torus, generate_klein_bottle,
)

N_POINTS = 1500
DIMENSION = 2
NEIGHBORHOOD_SIZE = 40
B_BOOTSTRAP = 60
NOISE_LEVELS = (0.0, 0.2, 0.3, 0.5, 0.6, 0.8)
ALPHA = 0.05
ODM_GRAPHS = ("mutual", "union")


@lru_cache(maxsize=2)
def clean_cloud(geometry):
    generator = generate_flat_torus if geometry == "torus" else generate_klein_bottle
    return generator(N_POINTS)


def odm_scores(points, seed):
    # The repository's unpruned union graph gives common PCA frames and candidate
    # edges. Select the reciprocal neighbors for the paper's mutual graph.
    builder = models.TopologyGraphBuilder(
        intrinsic_dim=DIMENSION, k_neighbors=NEIGHBORHOOD_SIZE, n_jobs=1,
    )
    frames, edges, _ = builder.build(points)
    neighbors = models.NearestNeighbors(
        n_neighbors=NEIGHBORHOOD_SIZE, algorithm="ball_tree", n_jobs=1,
    ).fit(points).kneighbors(points, return_distance=False)
    directed = sparse.csr_matrix(
        (np.ones(neighbors.size, dtype=bool),
         (np.repeat(np.arange(len(points)), neighbors.shape[1]), neighbors.ravel())),
        shape=(len(points), len(points)),
    )
    edges = np.array(sorted(edges))
    i, j = edges.T
    cross = frames[i].transpose(0, 2, 1) @ frames[j]
    signs = np.linalg.slogdet(cross)[0]
    if np.any(signs == 0):
        raise ValueError("An ODM Procrustes alignment has a singular cross matrix")
    is_mutual = np.asarray(directed[i, j]).ravel() & np.asarray(directed[j, i]).ravel()
    scores = {}
    for graph, keep in (("mutual", is_mutual), ("union", slice(None))):
        row, col = i[keep], j[keep]
        values = signs[keep]
        signed = sparse.csr_matrix(
            (np.tile(values, 2), (np.r_[row, col], np.r_[col, row])),
            shape=(len(points), len(points)),
        )
        _, labels = connected_components(signed, directed=False)
        component = labels == np.bincount(labels).argmax()
        signed = signed[component, :][:, component]
        degree = np.asarray(abs(signed).sum(axis=1)).ravel()
        if len(degree) < 3 or np.any(degree == 0):
            raise ValueError("ODM has no usable connected component")
        inverse_sqrt_degree = 1.0 / np.sqrt(degree)
        scale = sparse.diags(inverse_sqrt_degree)
        symmetric = scale @ signed @ scale
        initial = np.random.default_rng(seed + 200_000).normal(size=len(degree))
        eig, vectors = eigsh(symmetric, k=2, which="LA", tol=1e-8, v0=initial)
        order = np.argsort(eig)
        # Right eigenvector of D^{-1} Z, as in ODM Algorithm 1.
        abs_right = np.abs(inverse_sqrt_degree * vectors[:, order[-1]])
        score = float(abs_right.std() / abs_right.mean())
        scores[f"odm_{graph}_cv"] = 0.0 if score < 1e-7 else score
        scores[f"odm_{graph}_lcc_fraction"] = float(component.mean())
        scores[f"odm_{graph}_eigen_gap"] = float(eig[order[-1]] - eig[order[-2]])
    return scores


def escl_verdict(points, seed):
    tester = models.TwoStageBootstrapTester(
        builder_kwargs={"intrinsic_dim": DIMENSION, "k_neighbors": NEIGHBORHOOD_SIZE, "n_jobs": 1},
        analyzer_class=models.ESCL_H1_Analyzer,
        analyzer_kwargs={"beta": 5.0, "auto_beta": True, "cutoff": "auto"},
        b_bootstrap=B_BOOTSTRAP, max_workers=1,
    )
    # The repository workers call default_rng() without a seed. Give each draw
    # a reproducible stream, retaining the workers and all decision thresholds.
    original_rng, original_eigsh = np.random.default_rng, models.eigsh
    draw_sequence = np.random.SeedSequence([20261123, seed])
    eig_rng = original_rng([37, seed])

    def seeded_rng(rng_seed=None):
        return original_rng(draw_sequence.spawn(1)[0] if rng_seed is None else rng_seed)

    def seeded_eigsh(matrix, *args, **kwargs):
        if "v0" not in kwargs:
            kwargs["v0"] = eig_rng.normal(size=matrix.shape[0])
        return original_eigsh(matrix, *args, **kwargs)

    np.random.default_rng, models.eigsh = seeded_rng, seeded_eigsh
    try:
        result = tester.test(points)[2][0]
    finally:
        np.random.default_rng, models.eigsh = original_rng, original_eigsh
    status = result["status"]
    if "Twisted" in status:
        verdict = "twisted"
    elif "Trivial" in status:
        verdict = "orientable"
    elif "Degenerate" in status:
        verdict = "inconclusive"
    else:
        raise ValueError(f"Unexpected ESCL status {status!r}")
    return {
        "escl_verdict": verdict,
        "escl_status": status,
        "escl_p_noise": result["p_noise"],
        "escl_L_rho": result["fiedler_ci_lower"],
        "escl_C_m": result["chaos_ci_median"],
        "escl_lambda1": result["obs_lam1"],
        "escl_lambda2": result["obs_lam2"],
    }


def run_trial(task):
    phase, sigma, geometry, trial = task
    seed = round(sigma * 1000) + 7 * trial
    if geometry == "klein":
        seed += 10_000
    if phase == "calibration":
        seed += 1_000_000
    points = apply_noise_and_rotate(clean_cloud(geometry), sigma, seed)
    row = dict(phase=phase, sigma=sigma, geometry=geometry, trial=trial, seed=seed)
    row.update(odm_scores(points, seed))
    if phase == "evaluation":
        row.update(escl_verdict(points, seed))
    return row


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(evaluation, calibration):
    summary = []
    for sigma in sorted({row["sigma"] for row in evaluation}):
        by_geom = {
            geometry: [row for row in evaluation if row["sigma"] == sigma and row["geometry"] == geometry]
            for geometry in ("torus", "klein")
        }
        row = {"sigma": sigma}
        for geometry, rows in by_geom.items():
            row[f"n_{geometry}"] = len(rows)
            row[f"escl_{geometry}_twisted"] = sum(r["escl_verdict"] == "twisted" for r in rows)
            row[f"escl_{geometry}_inconclusive"] = sum(r["escl_verdict"] == "inconclusive" for r in rows)
        for graph in ODM_GRAPHS:
            score_key = f"odm_{graph}_cv"
            control = np.array([r[score_key] for r in calibration if r["sigma"] == sigma])
            for r in by_geom["torus"] + by_geom["klein"]:
                # With ties counted conservatively, p is the upper-tail rank
                # against an independent same-sigma Torus calibration set.
                if sigma == 0.0:
                    p = float(r[score_key] <= control[0])
                else:
                    p = (1 + np.count_nonzero(control >= r[score_key])) / (len(control) + 1)
                r[f"odm_{graph}_p"] = float(p)
                r[f"odm_{graph}_twisted"] = bool(p <= ALPHA)
            for geometry, rows in by_geom.items():
                row[f"odm_{graph}_{geometry}_twisted"] = sum(r[f"odm_{graph}_twisted"] for r in rows)
                row[f"odm_{graph}_{geometry}_cv_median"] = float(np.median([r[score_key] for r in rows]))
                row[f"odm_{graph}_{geometry}_lcc_min"] = min(r[f"odm_{graph}_lcc_fraction"] for r in rows)
                row[f"escl_minus_odm_{graph}_{geometry}_twisted"] = (
                    row[f"escl_{geometry}_twisted"] - row[f"odm_{graph}_{geometry}_twisted"]
                )
            klein = [r[score_key] for r in by_geom["klein"]]
            torus = [r[score_key] for r in by_geom["torus"]]
            row[f"odm_{graph}_auc"] = float(mannwhitneyu(klein, torus).statistic / (len(klein) * len(torus)))
            row[f"odm_{graph}_calibration_n"] = len(control)
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=24, help="replications per geometry at each positive noise level")
    parser.add_argument("--calibration", type=int, default=59, help="independent Torus clouds per positive noise level")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--noise", type=float, nargs="+", default=NOISE_LEVELS)
    args = parser.parse_args()
    if args.trials < 1 or args.calibration < 19 or args.workers < 1:
        parser.error("need positive trial/worker counts and at least 19 calibration clouds")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration, evaluation = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sigma in args.noise:
            # The clean designs are deterministic. One observed evaluation for
            # each suffices and is marked separately in the report.
            evaluation_tasks = [
                ("evaluation", sigma, geometry, i)
                for i in range(1 if sigma == 0.0 else args.trials)
                for geometry in ("torus", "klein")
            ]
            n_calibration = 1 if sigma == 0.0 else args.calibration
            cal_tasks = [("calibration", sigma, "torus", j) for j in range(n_calibration)]
            calibration.extend(executor.map(run_trial, cal_tasks, chunksize=2))
            cases = []
            for n, output in enumerate(executor.map(run_trial, evaluation_tasks), 1):
                cases.append(output)
                if n % 8 == 0:
                    print(f"sigma={sigma:g}: {n}/{len(evaluation_tasks)} paired evaluations", flush=True)
            evaluation.extend(cases)
            summary = summarize(evaluation, calibration)
            csv_write(args.output / "evaluation.csv", evaluation)
            csv_write(args.output / "calibration.csv", calibration)
            csv_write(args.output / "summary.csv", summary)
            print(json.dumps(summary[-1]), flush=True)
    meta = {
        "n": N_POINTS, "d": DIMENSION, "p": 24, "K": NEIGHBORHOOD_SIZE,
        "B": B_BOOTSTRAP, "alpha_for_odm_cv_rank": ALPHA,
        "replications_per_geometry_at_positive_noise": args.trials,
        "independent_calibration_torus_at_each_positive_noise": args.calibration,
        "noiseless_evaluation_count_per_geometry": 1,
        "noiseless_null": "Point mass at CV of the fixed clean Torus.",
        "odm_threshold": "Conservative rank p versus independent noise-matched perturbations of the fixed Torus template. This aide supplies a numeric decision for the original visual rule.",
        "odm_graphs": ["mutual kNN as in Singer-Wu algorithm", "union kNN as in ESCL"],
        "source": "Singer and Wu (2011), Algorithm 1, https://web.math.princeton.edu/~amits/publications/ODM-ACHA-published.pdf",
        "escl": "Repository TwoStageBootstrapTester, seeded worker draws and ARPACK starts; thresholds unchanged.",
        "elapsed_seconds": time.perf_counter() - started,
        "numpy": np.__version__,
    }
    (args.output / "design.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
