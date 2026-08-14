# The Kuber benchmark

A neural surrogate for electronics cooling has to do one thing: given a geometry and an operating
point, predict the steady thermal-fluid field fast and faithfully - *including on geometries it was
never trained on*. This document defines the task, the splits, the metrics, and how to get a number
onto the leaderboard.

## Task

Given
- **geometry** - a surface point cloud + outward normals, or a (directional) signed-distance field, or nothing (conditions only);
- **operating conditions** - fluid properties (Pr, ρ, Cp, μ), boundary conditions (wall temperature or heat flux, ambient/inlet temperature), inlet velocity;
- a set of **query points** (an arbitrary cloud in the fluid domain),

predict the steady field at each query point:

```
(U_x, U_y, U_z, T, p_rgh)      # velocity (3), temperature, reduced pressure
```

## Splits

### Primary: SIMSHIFT heatsink (public)

The [SIMSHIFT](https://github.com/psetinek/simshift) heatsink benchmark defines a **distribution
shift** by fin count: models train on one range and are tested on a shifted range.

| difficulty | shift | notes |
|---|---|---|
| easy | small fin-count shift | |
| **medium** | fins 5–8 (train) → 10–12 (test) | the primary leaderboard split |
| hard | largest fin-count shift | |

The split file maps `difficulty → {src,tgt} → {train,val,test}`. `src.*` is the source (in-distribution)
domain; `tgt.*` is the shifted target (OOD) domain. Normalizers are fit on **`src.train` only** - no
leakage, and difficulty-invariant, so a medium-trained model can be evaluated on easy/hard `tgt.test`.

### Secondary: the Kuber corpus (self-generated)

Adds axes SIMSHIFT does not cover - see [`DATASET.md`](DATASET.md):
- **fluid**: air / water / oil / glycol
- **shape**: fins / plate / cube / pin-fin
- **device class**: heatsink (wall-temp BC, buoyancy-driven) *and* cold plate (heat-flux BC, forced liquid)

Stronger OOD studies (leave-one-shape-out, leave-one-fluid-out) are on the roadmap.

## Metrics

All computed by `evaluate()` in [`kuber/train_simshift.py`](../kuber/train_simshift.py)
(the source of truth for exact formulas) and reported per field and averaged.

| metric | definition | why |
|---|---|---|
| **nRMSE** (per field) | RMSE in the per-channel z-normalized target space (≈ physical RMSE ÷ that field's std, std fit on `src.train`) | scale-free, comparable across fields |
| **mean nRMSE** | mean of the five per-field nRMSEs | **SIMSHIFT's primary model-selection metric** |
| **RMSE (physical)** | RMSE in physical units - K, m/s, Pa | engineering-legible |
| **relative L2** | ‖pred−gt‖₂ / ‖gt‖₂ per field | standard operator-learning metric |
| **near-wall T-RMSE** | physical T-RMSE over the closest 15 % of nodes to the solid (by SDF) | the tips/corners/walls, where the hot spot lives |
| **T-roughness ratio** | predicted vs. true local temperature slope; <1 = over-smoothed, ≈1 = faithful | catches surrogates that blur the field to cheat RMSE |

### Stability / no-explosion proof

Run by [`kuber/edge_proof.py`](../kuber/edge_proof.py) on `src.test` + `tgt.test`:

- **edge ∇T ratio** (pred / CFD) in the near-wall band - 1.0 is faithful; ≪1 over-smooths, ≫1 explodes.
- **steepest-peak (p99.9) ∇T ratio** - the worst hot spot.
- **explosion fraction** - fraction of nodes with |∇T|_pred > 2× the CFD max (a real surrogate: 0).
- **NaN/Inf count** - numerical soundness (a real surrogate: 0).

A benchmark-overfit curve-fit passes RMSE but fails these; a deployable surrogate passes both.

## Rules & disclosures

To keep the leaderboard honest, every entry must state:

1. **UDA** - did you use unsupervised domain adaptation on the target domain? (Ours: no.)
2. **External pretraining** - any data beyond the split's own training set? Name it. (Our pretrained variant discloses its corpus.)
3. **Parameter count** - measured, not rounded away.
4. **Reproduction command** - the exact command + environment that produced the number.

## Submitting a result

1. Produce your numbers with a script anyone can run (ideally the harness here).
2. Fork, and add a row to the leaderboard table in the top-level `README.md` with the four
   disclosures above.
3. Open a PR. Include the command, the environment (torch / CUDA / PhysicsNeMo versions), and - if
   possible - a checkpoint link so the result can be independently re-evaluated with
   `kuber.train_simshift --eval_only`.

We review for honesty (UDA/pretraining disclosed, params real, command reproducible), not for
whether you beat us. A well-documented result that loses is more valuable than an undocumented win.
