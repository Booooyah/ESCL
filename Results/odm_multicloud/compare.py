"""Replicate the paired ESCL/ODM experiment across independent clean designs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.stats import ortho_group

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from odm_pilot import compare as pilot

MASTER_SEED = 40_271_133
SIGMAS = (0.3, 0.5, 0.6)
GEOMETRIES = ("torus", "klein")
PHASES = ("evaluation", "calibration")
GRAPHS = pilot.ODM_GRAPHS


def seed_for(stream, *identifiers):
    return int(np.random.SeedSequence([MASTER_SEED, stream, *identifiers]).generate_state(1, dtype=np.uint64)[0])


@lru_cache(maxsize=128)
def clean_cloud(geometry, seed):
    """Use the pilot's embeddings and sampling laws with an explicit RNG."""
    rng = np.random.default_rng(seed)
    n = pilot.N_POINTS
    if geometry == "torus":
        u, v = rng.uniform(0.0, 2 * np.pi, size=(2, n))
        coordinates = (np.cos(u), np.sin(u), np.cos(v), np.sin(v))
    else:
        major, minor = 4.0, 1.0
        u_list, v_list = [], []
        while len(u_list) < n:
            batch = n - len(u_list)
            u_cand, v_cand = rng.uniform(0.0, 2 * np.pi, size=(2, batch))
            limit = minor * np.sqrt((major + minor * np.cos(v_cand)) ** 2 + (minor ** 2 / 4) * np.sin(v_cand) ** 2)
            accept = rng.uniform(0.0, minor * (major + minor), size=batch) <= limit
            u_list.extend(u_cand[accept])
            v_list.extend(v_cand[accept])
        u, v = np.asarray(u_list), np.asarray(v_list)
        radius = major + minor * np.cos(v)
        coordinates = (radius * np.cos(u), radius * np.sin(u),
                       minor * np.sin(v) * np.cos(u / 2), minor * np.sin(v) * np.sin(u / 2))
    return np.pad(np.column_stack(coordinates), ((0, 0), (0, 20)))


def run_trial(phase, sigma, geometry, design, draw):
    phase_idx, geo_idx = PHASES.index(phase), GEOMETRIES.index(geometry)
    sigma_id = round(1000 * sigma)
    # Evaluation templates are shared across noise levels. In calibration each
    # score uses its own pair of freshly sampled Torus design and noise.
    clean_seeds_by_sigma = 0 if phase == "evaluation" else sigma_id
    clean_seed = seed_for(0, phase_idx, geo_idx, design, clean_seeds_by_sigma)
    ids = (phase_idx, geo_idx, design, sigma_id, draw)
    noise_seed = seed_for(1, *ids)
    rotation_seed = seed_for(2, *ids)
    solver_seed = seed_for(3, *ids)
    base = clean_cloud(geometry, clean_seed)
    noisy = base + np.random.default_rng(noise_seed).normal(0.0, sigma, base.shape)
    rotated = noisy @ ortho_group.rvs(base.shape[1], random_state=np.random.default_rng(rotation_seed))
    result = dict(
        phase=phase, sigma=sigma, geometry=geometry, design=design, draw=draw,
        clean_seed=clean_seed, clean_hash=sha256(base.tobytes()).hexdigest(),
        noise_seed=noise_seed, rotation_seed=rotation_seed, solver_seed=solver_seed,
    )
    result.update(pilot.odm_scores(rotated, solver_seed))
    if phase == "evaluation":
        result.update(pilot.escl_verdict(rotated, solver_seed))
    return result


def run_level(executor, sigma, designs, draws, calibration_n):
    # Write the small calibration split first so its scores are frozen before
    # evaluating either geometry. Reuse the same controls for both graph choices.
    cal = list(executor.map(run_trial, ["calibration"] * calibration_n,
                           [sigma] * calibration_n, ["torus"] * calibration_n,
                           range(1, calibration_n + 1), [1] * calibration_n,
                           chunksize=2))
    print(f"sigma={sigma:g}: {len(cal)} calibration clouds ready", flush=True)
    tasks = [executor.submit(run_trial, "evaluation", sigma, geometry, design, draw)
             for design in range(1, designs + 1) for draw in range(1, draws + 1)
             for geometry in GEOMETRIES]
    eval_rows = []
    for i, future in enumerate(as_completed(tasks), 1):
        eval_rows.append(future.result())
        if i % 20 == 0 or i == len(tasks):
            print(f"sigma={sigma:g}: {i}/{len(tasks)} evaluation clouds", flush=True)
    return sorted(eval_rows, key=lambda r: (r["geometry"], r["design"], r["draw"])), cal


def by_design(evaluation):
    rows = []
    keys = sorted({(r["sigma"], r["geometry"], r["design"]) for r in evaluation})
    for sigma, geometry, design in keys:
        test = [r for r in evaluation if (r["sigma"], r["geometry"], r["design"]) == (sigma, geometry, design)]
        es = [r["escl_verdict"] == "twisted" for r in test]
        row = dict(sigma=sigma, geometry=geometry, design=design, clean_seed=test[0]["clean_seed"], n=len(test),
                   escl_twisted=sum(es), escl_inconclusive=sum(r["escl_verdict"] == "inconclusive" for r in test))
        for graph in GRAPHS:
            od = [r[f"odm_{graph}_twisted"] for r in test]
            row[f"odm_{graph}_twisted"] = sum(od)
            row[f"escl_only_twisted_vs_{graph}"] = sum(x and not y for x, y in zip(es, od))
            row[f"odm_only_twisted_{graph}"] = sum(y and not x for x, y in zip(es, od))
            row[f"escl_minus_odm_{graph}_twisted"] = sum(es) - sum(od)
        rows.append(row)
    return rows


def summarize_repeatability(design_rows):
    rows = []
    for sigma, geometry in sorted({(r["sigma"], r["geometry"]) for r in design_rows}):
        relevant = [r for r in design_rows if r["sigma"] == sigma and r["geometry"] == geometry]
        for graph in GRAPHS:
            diffs = [r[f"escl_minus_odm_{graph}_twisted"] / r["n"] for r in relevant]
            rows.append(dict(
                sigma=sigma, geometry=geometry, odm_graph=graph, n_designs=len(diffs),
                designs_ESCL_more=int(np.count_nonzero(np.array(diffs) > 0)),
                designs_equal=int(np.count_nonzero(np.array(diffs) == 0)),
                designs_ODM_more=int(np.count_nonzero(np.array(diffs) < 0)),
                min_escl_minus_odm_pp=100 * min(diffs),
                median_escl_minus_odm_pp=100 * float(np.median(diffs)),
                max_escl_minus_odm_pp=100 * max(diffs),
                cloud_ESCL_only=sum(r[f"escl_only_twisted_vs_{graph}"] for r in relevant),
                cloud_ODM_only=sum(r[f"odm_only_twisted_{graph}"] for r in relevant),
            ))
    return rows


def validate(evaluation, calibration, design_rows, designs, draws, calibration_n):
    assert len(evaluation) == len(SIGMAS) * len(GEOMETRIES) * designs * draws
    assert len(calibration) == len(SIGMAS) * calibration_n
    assert len({r["clean_hash"] for r in calibration}) == len(calibration)
    assert len({r["clean_hash"] for r in evaluation}) == len(GEOMETRIES) * designs
    assert {r["clean_hash"] for r in calibration}.isdisjoint(r["clean_hash"] for r in evaluation)
    for seed_type in ("noise_seed", "rotation_seed", "solver_seed"):
        assert len({r[seed_type] for r in evaluation + calibration}) == len(evaluation) + len(calibration)
    assert all(r["n"] == draws for r in design_rows)
    assert all(np.isfinite(r[f"odm_{graph}_cv"]) for graph in GRAPHS for r in evaluation + calibration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", type=int, default=10)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--calibration", type=int, default=99)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    if min(args.designs, args.draws, args.workers) < 1 or args.calibration < 19:
        parser.error("need positive design/draw/worker counts and at least 19 calibration clouds per level")
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    evaluation, calibration = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sigma in SIGMAS:
            ev, cal = run_level(executor, sigma, args.designs, args.draws, args.calibration)
            evaluation.extend(ev)
            calibration.extend(cal)
            summary = pilot.summarize(evaluation, calibration)
            design_rows = by_design(evaluation)
            # Each completed noise level is available for inspection.
            pilot.csv_write(args.output / "evaluation.csv", evaluation)
            pilot.csv_write(args.output / "calibration.csv", calibration)
            pilot.csv_write(args.output / "summary.csv", summary)
            pilot.csv_write(args.output / "by_design.csv", design_rows)
            pilot.csv_write(args.output / "repeatability.csv", summarize_repeatability(design_rows))
            print(json.dumps(summary[-1]), flush=True)
    validate(evaluation, calibration, design_rows, args.designs, args.draws, args.calibration)
    meta = dict(
        rng_seed=MASTER_SEED, n=pilot.N_POINTS, p=24, d=pilot.DIMENSION,
        K=pilot.NEIGHBORHOOD_SIZE, B=pilot.B_BOOTSTRAP, sigmas=SIGMAS,
        independent_clean_designs_per_geometry=args.designs, draws_per_design_and_sigma=args.draws,
        independent_torus_calibration_clouds_per_sigma=args.calibration,
        odm_calibration_level=pilot.ALPHA, evaluation_clouds=len(evaluation),
        calibration_clouds=len(calibration), clean_design_holdout_and_pairing_verified=True,
        calibration_note="Each score uses a fresh independently sampled Torus and noise. Rank references are shared across all evaluation cases at the same known sigma and are separate for the two ODM graph scores.",
        evaluation_note="A clean design is reused over the three noise levels and over the eight draws per level. All methods use the same noisy point coordinates within each case. Summarize repeatability by clean design.",
        numerical_rules="Use the unchanged ESCL and ODM settings from Results/odm_pilot/compare.py.",
        elapsed_seconds=round(time.perf_counter() - t0, 2),
    )
    (args.output / "design.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
