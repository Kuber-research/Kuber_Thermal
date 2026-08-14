<div align="center">

<img src="assets/Kuber_logo.png" alt="Kuber logo" width="130">

# Kuber

**An open framework for conjugate-heat-transfer AI** - build, train, and operate geometry-general neural surrogates for coupled fluid-heat problems (datacenters, CPUs/GPUs, heatsinks, cold plates, heat exchangers, power electronics, battery packs). First results: **electronics cooling**, where Kuber beats the previous best on the SIMSHIFT heatsink benchmark with **no domain adaptation**.

[![Interactive demo](https://img.shields.io/badge/%F0%9F%A7%8A%20interactive%20demo-live-1F4E79.svg)](https://shubhoo7-kuber-live.hf.space/)
[![Paper](https://img.shields.io/badge/paper-PDF-B31B1B.svg)](paper/kuber.pdf)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kuber-research/Kuber_Thermal/blob/main/notebooks/quickstart.ipynb)
[![tests](https://github.com/Kuber-research/Kuber_Thermal/actions/workflows/tests.yml/badge.svg)](https://github.com/Kuber-research/Kuber_Thermal/actions/workflows/tests.yml)
[![License: PolyForm NC](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-2E7D5B.svg)](LICENSE)

</div>

| Kuber prediction vs. CFD ground truth | |
|:---:|:---:|
| ![Heatsink - Kuber vs CFD](assets/sim/heat-sink-comparison.png) | ![Cold plate - Kuber vs CFD](assets/sim/cold-plate-comparison.png) |
| **Heatsink** - ±2.11 K, **7,000× faster** than CFD | **Cold plate** - ±1.33 K, **445× faster** than CFD |

**[Read the paper (PDF)](paper/kuber.pdf)** · [Project page](https://kuber-research.github.io/Kuber_Thermal/) · [Live demo](https://shubhoo7-kuber-live.hf.space/)

## Table of contents

[Highlights](#highlights) · [Installation](#installation) · [Quickstart](#quickstart) · [Recipes](#recipes) · [Dataset & Training](#dataset--training) · [Performance Benchmarks](#performance-benchmarks) · [Roadmap & Milestones](#roadmap--milestones) · [Contributing](#contributing) · [Supported systems](#supported-systems) · [Licensing](#licensing) · [Endorsed by](#endorsed-by) · [Citing](#citing)

## Highlights

- **Geometry-general.** KuberNet reads raw boundary geometry (surface point cloud + normals) and predicts the full field `(Uₓ, U_y, U_z, T, p_rgh)` at any query point - no analytic SDF, works on arbitrary CAD. Its *anisotropic boundary-layer* (ABL) cross-attention biases attention along the wall.
- **State of the art, no UDA.** **11.84 K** temperature RMSE on the SIMSHIFT heatsink OOD split, beating the prior best (UPT, 12.41 K) - while every baseline relies on unsupervised domain adaptation and Kuber uses none.
- **One model, many device classes.** Heatsinks and cold plates from a single set of weights, distinguished only by a device flag + physics conditioning.
- **Reproducible data engine.** Parametric geometry → OpenFOAM `buoyantSimpleFoam` → per-node `.npz`; resumable, convergence-gated, mesh-verified, license-clean (0 external cases).
- **Up to 10,000× faster than CFD**, sub-second and geometry-independent.

![KuberNet architecture](assets/sim/architecture_updated.png)

## Installation

### Prerequisites

- Git, **Python 3.10+**, and (recommended) a CUDA GPU for training.
- The model core is **GeoTransolver** from [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo).
- **Optional** (data generation only): [OpenFOAM](https://www.openfoam.com/) v2306+ on `PATH`.

### From source

No PyPI release yet - install from source:

```bash
git clone https://github.com/Kuber-research/Kuber_Thermal.git && cd Kuber_Thermal
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt
```

### Fresh install

```bash
deactivate 2>/dev/null; rm -rf .venv
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Quickstart

**No install:** the [Colab notebook](https://colab.research.google.com/github/Kuber-research/Kuber_Thermal/blob/main/notebooks/quickstart.ipynb) loads a real CHT case on CPU.

```bash
# Peek at a sample case (no OpenFOAM/GPU needed)
python -c "import numpy as np, os; d=np.load('data_sample/'+sorted(os.listdir('data_sample'))[0]); print({k:getattr(d[k],'shape',None) for k in d.files})"

# Evaluate a checkpoint (geometry mode + conditioning read from it)
python -m kuber.train_simshift --eval_only <model.pt> --data <npz_dir> --splits <splits.json> --difficulty medium

# Train the production model
python -m kuber.train_simshift --data <npz_dir> --splits <splits.json> --difficulty medium --geom_mode surface
```

## Recipes

| Task | Command |
|---|---|
| **Train** | `python -m kuber.train_simshift --data <d> --splits <s> --difficulty medium --geom_mode surface` |
| **Evaluate** (in-dist + OOD) | `python -m kuber.train_simshift --eval_only <model.pt> --data <d> --splits <s> --difficulty medium` |
| **Pretrain → fine-tune** | `... --geom_mode surface --drop_geom_scalars --init_from pretrain/<ckpt>` |
| **Stability proof** | `python -m kuber.edge_proof --ckpt <model.pt> --data <d> --splits <s> --difficulty medium` |
| **Generate data** (OpenFOAM) | `python datagen/run_sweep_bsf.py --cases <cases> --out <corpus> --scripts <scripts>` |
| **Rebuild figures** | `python assets/make_figures.py` |

`--geom_mode {none,sdf,dsdf,surface}` picks the geometry representation (`surface` = production); `--refine` adds the PDE-Refiner head. See `--help` for all flags.

## Dataset & Training

A **self-generated OpenFOAM CHT corpus** for electronics cooling - **0 cases from SIMSHIFT or any licensed source**. Each case: one steady field sampled to a 16,384-node fluid cloud + 2,048-point surface cloud, across two device classes (heatsinks, cold plates), four fluids (air/water/oil/glycol), natural and forced convection. Built by a parametric generator → mesh (`snappyHexMesh`/`blockMesh` + prism layers) → `buoyantSimpleFoam` → `.npz`, resumable and convergence-gated. Format, coverage, mesh-convergence study: **[`docs/DATASET.md`](docs/DATASET.md)**.

One **KuberNet** (~14.3 M params) trains across both classes with geometry read *only* from the surface cloud, under a unified physics-conditioning vector; targets z-scored on the training split (no leakage); Adam + Warmup-Stable-Decay; **no UDA**. Full recipe: **[`docs/TRAINING.md`](docs/TRAINING.md)**.

## Performance Benchmarks

**SIMSHIFT heatsink, medium / OOD split** (train fins 5–8 → test 10–12). Baselines include UDA; Kuber uses none. Lower is better. Full metrics + caveats: [`docs/RESULTS.md`](docs/RESULTS.md).

> **UDA** (*unsupervised domain adaptation*) adapts a model to the unlabeled **test** distribution at training time - so a UDA result is **not zero-shot**. Every baseline below uses it; Kuber uses none and is evaluated fully zero-shot.

| model | UDA | Temp RMSE (K) ↓ | Velocity RMSE (m/s) ↓ | params |
|---|:---:|:---:|:---:|:---:|
| **Kuber - KuberNet (ABL)** | ✗ | **11.84** | 0.044 | 14.3 M |
| UPT *(prev. best)* | ✓ | 12.41 | 0.039 | ~14 M |
| Transolver | ✓ | 13.43 | 0.041 | ~14 M |
| PointNet | ✓ | 17.43 | 0.044 | ~14 M |

![SIMSHIFT leaderboard](assets/fig_leaderboard.svg)

The lead number is **zero-shot** (test fin counts never seen in training); predicted ∇T stays at or below physical everywhere (explosion fraction 0, no NaN/Inf). Pretraining on the corpus lowers error further - a physics prior, not test-set exposure ([why it isn't leakage](docs/RESULTS.md)).

**One model, two device classes** (held-out per class on our corpus; not comparable to SIMSHIFT above):

| held-out class | Temp RMSE (K) ↓ | mean nRMSE ↓ |
|---|:---:|:---:|
| cold plates | 3.11 | 0.028 |
| heatsinks | 5.13 | 0.145 |
| in-distribution (both) | 1.72 | 0.027 |

## Roadmap & Milestones

An open **suite**, not a finished benchmark. Three milestones, each gated on a verifiable **outcome** rather than an activity.

### Milestone 1 · Production-ready on real customer geometry

*Done when: the model clears a design partner's written acceptance criteria on their own parts.*

- [ ] **Generalize to as-supplied CAD**, beyond parametric families. The surface encoder already accepts arbitrary meshes - this is a data + validation problem, not an architecture one.
- [ ] **Grow the corpus** past its air-dominated composition - substantially more liquid-cooled cases, working fluids, and regimes.
- [ ] **Scaling laws for physical AI.** Extend the two-point [value-of-data](docs/RESULTS.md) ablation into measured curves: how OOD error scales with corpus size, model parameters, and - the question specific to physics - *geometry and physics diversity* versus raw case count.
- [ ] **A graded distribution-shift protocol.** Replace single-axis holdouts with a shift taxonomy scored independently per axis, reported as degradation curves - with **(i)** measured shift magnitude (MMD / Wasserstein in a geometry-physics feature space), **(ii)** a leakage audit (nearest-neighbour over the corpus), and **(iii)** provenance-shift as the ceiling test (a *different* solver + mesher, ultimately experimental data - the only test that separates learned physics from learned numerics).

  | Axis | Weak shift | Strong shift |
  |---|---|---|
  | Geometry | parametric extrapolation (current SIMSHIFT split) | topology change → human-authored CAD |
  | Regime | Ra/Re beyond the training envelope | natural → forced convection |
  | Fluid | unseen Prandtl number | unseen material class |
  | Boundary condition | wall-temperature → heat-flux | mixed / conjugate interfaces |
  | Scale | characteristic length outside range | order-of-magnitude change |

- [ ] **Release trained checkpoints** as tagged GitHub Releases, and replace estimated inference latency with measured per-GPU timings.

### Milestone 2 · Trustworthy enough to design against

*Done when: an engineer can act on a prediction without re-running CFD to check it.*

- [ ] **Calibrated per-node uncertainty** from the PDE-Refiner denoising ensemble - real error bars that say where to trust the surrogate and where to fall back to CFD. This is the gate on adoption; accuracy alone is not sufficient.
- [ ] **Cold-plate topologies beyond straight-channel** - serpentine, pin-fin, parallel micro-channel.
- [ ] **Additional device classes** - heat exchangers, power electronics, battery packs - prioritized by what design partners bring us.
- [ ] **Architecture** - hierarchical multi-resolution surface tokens; per-block geometry cross-attention.
- [ ] **Active learning driven by the scaling curves** - target the data engine where error falls fastest per case generated.

### Milestone 3 · From predictor to design tool

*Done when: a user goes from CAD file to an optimized geometry without leaving the loop.*

- [ ] **Connectors.** STEP / IGES / STL ingest and native OpenFOAM case import, plus export back into standard thermal workflows, exposed as a Python API and CLI.
- [ ] **Agentic geometry optimization.** Propose → predict → score → refine against thermal and pressure-drop objectives, with CFD in the loop only to verify the winner.
- [ ] **Deployment.** Hosted inference API, ONNX / TensorRT export, and a public versioned leaderboard with a sealed test set.

Contributions to any milestone are welcome - see [Contributing](#contributing). Rationale and full method are in the [paper](paper/kuber.pdf).

## Contributing

Rigor is welcome - new baselines, harder splits, better data, or caught over-claims.

### Guidelines

- **No licensed or scraped data** - self-generated or redistribution-permitted only.
- **Reproducibility** - every result ships with its exact command and environment.
- **Honesty** - state plainly whether a number used UDA or external pretraining.

Full details, including how to submit a leaderboard row, in [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/BENCHMARK.md`](docs/BENCHMARK.md#submitting-a-result).

### Checks before a PR

```bash
python -m py_compile kuber/*.py     # syntax
pip install pytest && pytest -q     # test suite (tests/); CI runs the same
```

### Reporting issues

Open a [GitHub issue](https://github.com/Kuber-research/Kuber_Thermal/issues) for bugs or over-claims - we would rather fix a number than defend it.

### IDE setup

VS Code and PyCharm work out of the box; point the interpreter at `.venv`. No repo-specific config required.

## Supported systems

- **OS:** Linux (tested). **Python:** 3.10+. Pinned: `torch==2.12.1`, `numpy==2.4.6`, `nvidia-physicsnemo==2.1.1`.
- **Hardware:** CUDA GPU recommended for training (required for the multiscale ball-query core); eval + data sample run on CPU.
- **Data generation** (optional): OpenFOAM v2306+ with `buoyantSimpleFoam`, `blockMesh`, `snappyHexMesh` on `PATH`.

## Licensing

**[PolyForm Noncommercial 1.0.0](LICENSE)** - free for research, education, and evaluation. Commercial use requires a separate license (contact the Kuber.ai team). The GeoTransolver core is a dependency from NVIDIA PhysicsNeMo (Apache-2.0).

## Endorsed by

Kuber is a new open framework. Using it in research or evaluating it for production? Reach out via a [GitHub issue](https://github.com/Kuber-research/Kuber_Thermal/issues) - research partners and pilots will be listed here.

## Citing

Builds on **GeoTransolver / PhysicsNeMo** (NVIDIA); draws on **Transolver**, **AB-UPT**, **SIMSHIFT**, **PDE-Refiner**, and **OpenFOAM**. Full references in [`docs/MODEL.md`](docs/MODEL.md) and the [paper](paper/kuber.pdf).

```bibtex
@techreport{kubernet2026,
  title       = {KuberNet: A Geometry-General Surrogate for Conjugate Heat Transfer with Boundary-Layer Attention},
  author      = {Jain, Shubh and Agarwal, Hardik},
  institution = {Kuber.ai},
  year        = {2026},
  url         = {https://github.com/Kuber-research/Kuber_Thermal}
}
```
