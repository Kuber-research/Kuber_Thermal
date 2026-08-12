<div align="center">

# Kuber

**An open benchmark and neural-surrogate baseline for conjugate heat transfer (CHT) in electronics cooling.**

Reproducible OpenFOAM data · a strong geometry-conditioned surrogate · state-of-the-art results on the public SIMSHIFT heatsink split — *without domain adaptation*.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Built on PhysicsNeMo](https://img.shields.io/badge/built%20on-NVIDIA%20PhysicsNeMo-76b900.svg)](https://github.com/NVIDIA/physicsnemo)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

Neural surrogates for CFD are usually benchmarked on external aerodynamics (cars, wings). **Electronics cooling** — heatsinks and cold plates, where *conjugate heat transfer* couples a solid conductor to a moving fluid — has no widely-used open harness, no shared data-generation recipe, and no honest leaderboard. Kuber is a first step toward one.

It ships four things:

1. **A reproducible CHT dataset pipeline** — parametric geometry → OpenFOAM `buoyantSimpleFoam` → per-node `.npz`, fully resumable, with a convergence gate and a mesh-convergence check. No licensed or scraped data; everything is self-generated and commercial-safe.
2. **A strong open baseline** — *SurfaceGeoTransolver*: a GeoTransolver physics-attention core (NVIDIA PhysicsNeMo, Apache-2.0) plus an AB-UPT-style surface-geometry encoder, ~14 M parameters, predicting the full 5-channel field `(Uₓ, U_y, U_z, T, p_rgh)` at every query node.
3. **State-of-the-art results on a public split** — on the SIMSHIFT heatsink **medium** OOD split, our model reaches **12.14 K temperature RMSE, beating the previous best (UPT, 12.41 K) with no unsupervised domain adaptation** — the crutch every published baseline relies on.
4. **A metrics + stability harness** — normalized RMSE per field, an edge/near-wall gradient-fidelity "no-explosion" proof, and a value-of-data ablation.

> **Status.** Research preview. The results below are all measured and reproducible with the code here; the honest caveats (data imbalance, single OOD axis, inference-time estimate) are stated inline and in [`docs/RESULTS.md`](docs/RESULTS.md). We would rather under-claim.

## Table of contents

- [Headline results](#headline-results)
- [Leaderboard — SIMSHIFT heatsink (medium / OOD)](#leaderboard--simshift-heatsink-medium--ood)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [The benchmark](#the-benchmark)
- [Dataset](#dataset)
- [Model](#model)
- [Repository layout](#repository-layout)
- [Reproducing our numbers](#reproducing-our-numbers)
- [Roadmap](#roadmap)
- [Citing](#citing)
- [License & acknowledgements](#license--acknowledgements)

## Headline results

| claim | number | where |
|---|---|---|
| **SOTA temperature error** on the public SIMSHIFT-medium OOD split, **no UDA** | **12.14 K RMSE** (vs UPT 12.41 K *with* UDA) | [`docs/RESULTS.md` §7.1](docs/RESULTS.md) |
| **Best velocity** of any model on the same split | **0.038 m/s RMSE** (vs UPT 0.039) | [`docs/RESULTS.md` §7.1](docs/RESULTS.md) |
| **Zero-shot to unseen fin counts** (train fins 5–8 → test 10–14) | the SOTA number *is* the OOD number | [`docs/RESULTS.md` §7.4](docs/RESULTS.md) |
| **No gradient explosion** at fin tips/corners, in- and out-of-distribution | explosion fraction **0**, NaN/Inf **0** | [`docs/RESULTS.md` §7.4](docs/RESULTS.md) |
| **Pretraining on our self-generated corpus helps** (clean A/B) | easy shift **−19 %**, in-dist **−12 %** | [`docs/RESULTS.md` §7.3](docs/RESULTS.md) |
| **~1,000× faster than CFD** | sub-second forward pass vs ~22 min median CFD solve\* | [`docs/RESULTS.md` §8](docs/RESULTS.md) |

\* Inference latency is a sub-second estimate pending exact per-GPU timing; the CFD solve times are measured (median 22 min over 601 cases; 2.7–117 min range).

## Leaderboard — SIMSHIFT heatsink (medium / OOD)

The [SIMSHIFT](https://github.com/psetinek/simshift) heatsink benchmark trains on fin counts 5–8 and tests on the **shifted** target domain (fins 10–12). Lower is better. Baseline numbers are quoted exactly from the SIMSHIFT paper (Table 2); those numbers **already include UDA** (unsupervised domain adaptation) — their best case. Ours use **no UDA**.

| # | model | UDA | Temp RMSE (K) ↓ | Velocity RMSE (m/s) ↓ | params |
|---|---|:---:|:---:|:---:|:---:|
| 🥇 | **SurfaceGeoTransolver (ours)** | ✗ | **12.14** | 0.044 | 14.3 M |
| 🥈 | UPT *(prev. published best)* | ✓ | 12.41 | 0.039 | ~14 M¹ |
| 🥉 | **GeoTransolver + directional-SDF (ours)** | ✗ | 13.07 | **0.038** | 14.0 M |
| 4 | Transolver | ✓ | 13.43 | 0.041 | ~14 M¹ |
| 5 | PointNet | ✓ | 17.43 | 0.044 | ~14 M¹ |

Plus a reduced-conditioning variant **pretrained on our corpus** then fine-tuned: **12.38 K — also past UPT** ([§7.3](docs/RESULTS.md)).

¹ The SIMSHIFT paper prints no parameter counts; their released configs show all three baselines are comparably-sized (~10–15 M) small models. **We are not winning by scaling parameters** — the edge is geometry conditioning + training recipe at the same budget. On the paper's *primary averaged-NRMSE* metric the **SDF** model actually leads overall (0.593); the surface model's win is specifically on **temperature**, the field that matters for thermal design. Full transparency — every metric, every caveat — in [`docs/RESULTS.md`](docs/RESULTS.md).

**Want on the board?** See [How to submit](docs/BENCHMARK.md#submitting-a-result).

## Installation

```bash
git clone https://github.com/ShubhJain007/Kuber.git
cd Kuber
python -m venv .venv && source .venv/bin/activate     # or use conda
pip install -r requirements.txt
```

The model core is **GeoTransolver** from NVIDIA PhysicsNeMo (`physicsnemo.experimental.models.geotransolver`). Install PhysicsNeMo per its [instructions](https://github.com/NVIDIA/physicsnemo); a CUDA GPU is recommended for training and required for the multiscale ball-query core.

**Data generation** additionally needs [OpenFOAM](https://www.openfoam.com/) (v2306+ with `buoyantSimpleFoam`, `blockMesh`, `snappyHexMesh`) on your `PATH`. It is only needed if you want to regenerate or extend the corpus — the model, eval, and a data sample work without it.

## Quickstart

```bash
# 1. Peek at a sample case (no OpenFOAM, no GPU needed)
python -c "import numpy as np; d=np.load('data_sample/'+__import__('os').listdir('data_sample')[0]); \
           print({k:getattr(d[k],'shape',None) for k in d.files})"

# 2. Evaluate a trained checkpoint on the SIMSHIFT split (see checkpoints/README.md for weights).
#    --eval_only takes the checkpoint path; geometry mode + conditioning are read from it.
python -m kuber.train_simshift \
    --data <simshift_npz_dir> --splits <splits.json> --difficulty medium \
    --eval_only <model.pt>

# 3. Reproduce the stability / no-explosion proof
python -m kuber.edge_proof \
    --ckpt <model.pt> --data <simshift_npz_dir> --splits <splits.json> --difficulty medium

# 4. Train from scratch (directional-SDF variant, the fastest strong baseline)
python -m kuber.train_simshift \
    --data <npz_dir> --splits <splits.json> --difficulty medium --geom_mode dsdf
```

Flags: `--geom_mode {none,sdf,dsdf,surface}` selects the geometry representation; `--refine` adds the generative PDE-Refiner head. See `python -m kuber.train_simshift --help`.

## The benchmark

- **Task.** Given geometry (surface point cloud or SDF) + operating conditions (fluid properties, boundary temperatures/fluxes, inlet velocity), predict the steady field `(Uₓ, U_y, U_z, T, p_rgh)` at an arbitrary query point cloud.
- **Splits.** The public SIMSHIFT heatsink split (`easy`/`medium`/`hard`, each an increasing fin-count shift) is the primary leaderboard. The self-generated corpus adds a fluid axis (air/water/oil/glycol) and a shape axis (fins/plate/cube/pin-fin), plus a cold-plate device class (heat-flux BC, forced liquid).
- **Metrics.** Per-field RMSE and normalized RMSE (nRMSE); the paper's primary selector is **mean nRMSE** across fields. Plus near-wall (edge-band) T-RMSE and the gradient-fidelity "no-explosion" measures. All defined in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).
- **Rules.** Report whether you used UDA and any external pretraining data. Report parameter count. Ours use no UDA; the pretrained variant discloses its corpus.

Full metric definitions, split construction, and submission instructions: [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Dataset

A self-generated OpenFOAM CHT corpus — **0 cases from SIMSHIFT or any licensed source**.

| axis | coverage |
|---|---|
| fluids | air, water, mineral oil (Pr≈292), glycol — conditioned on Pr, ρ, Cp, μ |
| regimes | natural + forced convection |
| shapes | fins, plate, cube, pin-fin arrays; + cold-plate ducts (heat-flux BC) |
| conditions | ambient 290–310 K, wall 340–400 K, fin count 5–14, varied geometry |
| fidelity | ~1.4 M cells/case (snap2 + 3 prism layers), subsampled to 16 384 nodes |
| fields | `U(x,y,z), T, p_rgh` per node |

Pipeline (`datagen/`): parametric generator → STL → `blockMesh` + `snappyHexMesh` (+layers) → solve → `.npz`, resumable via `run_sweep_bsf.py`, gated by convergence + a physics filter. Mesh convergence is verified to within **0.1 K** of a fine mesh at ~2.7× lower cost. A 6-case sample lives in [`data_sample/`](data_sample). Details + the data contract: [`docs/DATASET.md`](docs/DATASET.md).

## Model

**SurfaceGeoTransolver** = GeoTransolver physics-attention transformer (256 hidden × 12 layers) + an AB-UPT-style surface encoder (surface point cloud + normals → geometry tokens → per-node descriptor via kNN cross-attention). Geometry can also be fed as a scalar SDF, a **directional SDF** (signed distance + unit gradient), or nothing (conditions only), selected by `--geom_mode`. An optional **PDE-Refiner** (denoising-diffusion) head restores high-frequency content and yields uncertainty. Architecture card: [`docs/MODEL.md`](docs/MODEL.md).

## Repository layout

```
kuber/
├── kuber/               # the Python package
│   ├── surface_geotransolver.py   # SurfaceGeoTransolver (surface branch + GeoTransolver core)
│   ├── surface_model.py           # surface encoder + local cross-attention
│   ├── surface_geom.py            # analytic surface point cloud + normals from parameters
│   ├── pde_refiner.py             # PDE-Refiner (generative refinement head)
│   ├── train_simshift.py          # training + eval + all geometry modes + data loading
│   └── edge_proof.py              # gradient-fidelity / no-explosion stability harness
├── datagen/                   # OpenFOAM buoyantSimpleFoam CHT pipeline (heatsink + cold plate)
├── data_sample/              # 6 example cases (.npz) + a sample split
├── checkpoints/             # how to obtain trained weights
├── docs/                    # RESULTS, DATASET, MODEL, BENCHMARK
└── scripts/                 # convenience wrappers
```

## Reproducing our numbers

Every number in the leaderboard and in [`docs/RESULTS.md`](docs/RESULTS.md) is produced by the code here:

```bash
# SIMSHIFT SOTA table (surface + SDF variants)
python -m kuber.train_simshift --data <simshift> --splits <splits> --difficulty medium --geom_mode surface
python -m kuber.train_simshift --data <simshift> --splits <splits> --difficulty medium --geom_mode dsdf
# value-of-data A/B: pretrain on our corpus, then fine-tune on SIMSHIFT (reduced "transfer"
# conditioning = solidTemp only, so weights transfer cleanly). Compare to a from-scratch control.
python -m kuber.train_simshift --data <our_corpus> --splits <corpus_splits> --difficulty medium \
    --geom_mode surface --drop_geom_scalars --out pretrain/
python -m kuber.train_simshift --data <simshift> --splits <splits> --difficulty medium \
    --geom_mode surface --drop_geom_scalars --init_from pretrain/<pretrained.pt>
# stability / no-explosion proof
python -m kuber.edge_proof --ckpt <model.pt> --data <simshift> --splits <splits> --difficulty medium
```

SIMSHIFT data + splits come from the [official SIMSHIFT repo](https://github.com/psetinek/simshift); our corpus is regenerated with `datagen/`.

## Roadmap

- [ ] Release trained checkpoints as GitHub Release assets (surface / dsdf / multi-geometry).
- [ ] Leave-one-shape-out and leave-one-fluid-out OOD splits (stronger generalization tests).
- [ ] Cold-plate topology diversity (serpentine / pin-fin / parallel channels) → arbitrary-CAD generalization.
- [ ] A hosted leaderboard + automated submission checker.
- [ ] Held-out public test set with sealed labels.

## Citing

If you use Kuber, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{kuber2026,
  title  = {Kuber: An Open Benchmark and Neural-Surrogate Baseline for
            Conjugate Heat Transfer in Electronics Cooling},
  author = {Jain, Shubh},
  year   = {2026},
  url    = {https://github.com/ShubhJain007/Kuber}
}
```

Please also cite the works Kuber builds on: **GeoTransolver / PhysicsNeMo** (NVIDIA), **Transolver** (Wu et al., 2024), **AB-UPT** (Alkin et al., 2025), **SIMSHIFT** (Setinek et al.), and **PDE-Refiner** (Lippe et al., NeurIPS 2023). Full references in [`docs/MODEL.md`](docs/MODEL.md).

## License & acknowledgements

Apache-2.0 — see [`LICENSE`](LICENSE). The GeoTransolver core is from NVIDIA PhysicsNeMo (Apache-2.0). The dataset is self-generated with OpenFOAM and is released for research and commercial use.

<div align="center">
<sub>Built by the Kuber.ai team — geometry-general thermal surrogates for electronics cooling.</sub>
</div>
