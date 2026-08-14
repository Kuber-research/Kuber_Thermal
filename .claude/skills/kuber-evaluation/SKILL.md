---
name: kuber-evaluation
description: Evaluate Kuber checkpoints and publish the numbers — eval_only on src.test/tgt.test, the edge_proof stability harness, the metric definitions (nRMSE, relL2, near-wall T-RMSE, roughness ratio), the pytest suite, and the honesty rules for putting a result into docs/RESULTS.md, results/*.json, and the leaderboard. Use whenever measuring, reporting, or auditing a result.
---

# Evaluating a Kuber surrogate

Two commands produce every published number. Both rebuild the model from the checkpoint's `meta`
block, so you never pass the training flags again.

```bash
# accuracy: in-distribution (src.test) + OOD (tgt.test), with fidelity metrics
python -m kuber.train_simshift \
    --data <npz_dir> --splits <splits.json> --difficulty medium --eval_only <ckpt.pt>

# numerical stability / no-explosion proof (CPU by default, does not touch the GPU)
python -m kuber.edge_proof \
    --ckpt <ckpt.pt> --data <npz_dir> --splits <splits.json> --difficulty medium \
    --out edge_proof_medium.json
```

`scripts/eval.sh <ckpt> <data> <splits> [difficulty]` runs both in sequence. Because all normalizers
are fit on `src.train` only, they are difficulty-invariant: a medium-trained checkpoint is evaluated
on `easy`/`hard` `tgt.test` by just changing `--difficulty`.

## What the metrics mean

`evaluate()` in `kuber/train_simshift.py` is the source of truth for the exact formulas;
`kuber/metrics.py` re-implements the same definitions in dependency-free NumPy for tests and reuse.
If you change one, change both — `tests/test_metrics.py` pins them.

| metric | meaning | why it exists |
|---|---|---|
| **nRMSE** per field | RMSE on the z-normalized targets (std fit on `src.train`) | scale-free, comparable across the 5 channels |
| **mean nRMSE** | mean of the five | SIMSHIFT's primary model-selection metric; this is what checkpointing tracks |
| **RMSE_phys** | de-normalized to K, m/s, Pa | the engineering-legible number quoted in the README |
| **relL2** | ‖pred−gt‖₂ / ‖gt‖₂ | standard operator-learning metric |
| **near-wall T-RMSE** | physical T-RMSE over the closest 15 % of nodes by SDF | the tips/corners/walls where the hot spot lives |
| **T-roughness ratio** | pred/gt mean local temperature slope; <1 = over-smoothed | catches surrogates that blur the field to win on RMSE |

Fidelity metrics (the last two) are computed only for final / eval-only reporting, never per-epoch —
they are expensive and are not a selection signal. The near-wall band is a *percentile* of SDF, not a
fixed metric threshold, because CFD meshes cluster near the solid.

**The headline number is the OOD one.** On SIMSHIFT medium, `tgt.test` fin counts never appear in
training, so it is a zero-shot result; report `src.test` alongside it, never instead of it. The
trainer also prints `OOD gap = mean nRMSE(ood) − mean nRMSE(in-dist)`.

## The stability proof

`edge_proof.py` exists because RMSE alone cannot tell a deployable surrogate from a benchmark-overfit
curve fit. It compares predicted vs CFD local |∇T| (kNN slope estimator, distances floored at half the
median node spacing so subsampling cannot manufacture a 1/ε spike) in three regions and asserts by
measurement:

- edge-band and p99.9 ∇T ratios ≲ 1.0 — faithful, neither over-smoothed nor exploding
- `explosion_frac` (nodes with |∇T|_pred > 2× the CFD max) == 0
- `value_overshoot_frac` — predicted T outside the CFD range ±10 %
- `nan_inf_total` == 0

A ratio well under 1 means over-smoothing (the safe direction, but say so); a ratio above 1 or any
nonzero explosion fraction is a blocking result, not a caveat.

## Test suite

```bash
pip install pytest && pytest -q     # tests/
python -m py_compile kuber/*.py     # quick syntax check
```
Covers the model forward shape contract, the metric definitions, the `data_sample/` data contract,
and the stability harness. Tests needing the GeoTransolver core self-skip where PhysicsNeMo is absent,
which is why CI (`.github/workflows/tests.yml`, CPU-only torch) stays green. Keep that property: guard
any new physicsnemo-dependent test with `pytest.importorskip("physicsnemo")`.

## Publishing a number — the honesty rules

These are the project's stated ground rules (`docs/BENCHMARK.md`, `CONTRIBUTING.md`). A number is not
publishable until all four are attached:

1. **UDA** — did it use unsupervised domain adaptation? Kuber's answer is *no*, everywhere; that is the
   headline claim against baselines that all use it. If a run ever touches the target distribution,
   the row must say so.
2. **External pretraining** — name the corpus. The pretrained variant is labelled in every table it
   appears in. Never pretrain on any held-out evaluation split.
3. **Parameter count** — measured (`sum(p.numel())/1e6`, printed by the trainer), not rounded away.
4. **Reproduction command** — the exact command plus torch / CUDA / PhysicsNeMo versions.

Mark estimates as estimates. Inference latency is currently an *estimate* pending per-GPU timing, and
`docs/RESULTS.md` says so with an asterisk — keep that discipline rather than quietly promoting a
number to "measured".

## Where a result has to land

A new headline number is not done until these agree with each other:

- `results/simshift_medium.json`, `results/multigeo.json`, `results/leaderboard.csv` — machine-readable
- `docs/RESULTS.md` — the narrative tables and caveats (§7 is the caveat list; extend it, don't prune it)
- `README.md` — the benchmark table and any figure captions
- `assets/make_figures.py` — the figure numbers are hard-coded from `docs/RESULTS.md`; regenerate with
  `python assets/make_figures.py` and commit the regenerated SVGs
- `paper/kuber.tex` — if the number appears in the report, and re-render the PDF

Cross-corpus comparisons are not comparable: the multi-geometry cold-plate/heatsink numbers are on
Kuber's own corpus with no public benchmark behind them, and `docs/RESULTS.md` states that explicitly.
Do not put them in the SIMSHIFT table.
