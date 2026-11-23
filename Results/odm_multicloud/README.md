# Direct comparison with orientable diffusion maps

This directory contains the synthetic point-cloud comparison in
Supplementary Appendix G.3 (Table S2) of *Robust Non-parametric Inference of
Manifold Orientability via Empirical Spectral Cohomology Learning*.

The design uses ten independent clean point clouds per geometry and eight
noise realizations per cloud at each noise level. Both ODM graph variants
are calibrated on separate torus clouds.

- `summary.csv` contains the counts for Table S2.
- `evaluation.csv` and `calibration.csv` contain the evaluation and reference scores.
- `by_design.csv` and `repeatability.csv` summarize comparisons by clean point cloud.
- `design.json` records the simulation settings.

From the repository root, use Python 3.11 to install the dependencies and
run the comparison:

```sh
python -m pip install -r Results/odm_multicloud/requirements.txt
python Results/odm_multicloud/compare.py --workers 4 --output odm_rerun
```

The script generates the point clouds from the recorded random seed.
The output directory keeps newly generated results separate from the
archived CSV files. Adjust the worker count to suit available memory.
