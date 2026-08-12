<div align="center">

# Kuber

**An engineering-AI suite for electronics-cooling thermal simulation.**

Generate physics data → train geometry-general thermal surrogates → benchmark them honestly → *(on the roadmap)* connect your CAD, quantify uncertainty, and let agents optimize the design.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Built on PhysicsNeMo](https://img.shields.io/badge/built%20on-NVIDIA%20PhysicsNeMo-76b900.svg)](https://github.com/NVIDIA/physicsnemo)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

Designing a heatsink or a cold plate means running CFD — minutes to hours per geometry — inside a loop you'd like to run thousands of times. Kuber is a suite for that loop: a **data engine** that generates conjugate-heat-transfer (CHT) physics, a **geometry-general surrogate** that predicts the full thermal-fluid field in a sub-second, and an **honest evaluation harness** that proves the surrogate generalizes to geometries it never saw — with a roadmap toward the pieces that turn a fast predictor into a design tool: CAD connectors, calibrated uncertainty, and agentic geometry optimization.

Today the surrogate already reaches **state of the art on a public benchmark** (SIMSHIFT heatsink), beating the previous best **with no domain adaptation**.

## What's in the suite

| pillar | status | what it does |
|---|:---:|---|
| **Data Engine** — [`datagen/`](datagen) | ✅ shipping | parametric geometry → OpenFOAM `buoyantSimpleFoam` → per-node `.npz`; resumable, convergence-gated, commercial-safe |
| **Surrogate models** — [`kuber/`](kuber) | ✅ shipping | *SurfaceGeoTransolver* (~14 M params): GeoTransolver core + AB-UPT-style surface encoder; predicts `(Uₓ,U_y,U_z,T,p_rgh)` at any query cloud |
| **Eval & benchmark harness** — [`docs/`](docs) | ✅ shipping | per-field nRMSE, near-wall fidelity, a no-explosion stability proof, and a public leaderboard |
| **Connectors** | 🔜 roadmap | bring-your-own CAD (STEP/STL/mesh) + OpenFOAM case ingest; drop the surrogate into existing thermal workflows |
| **Bayesian uncertainty** | 🔜 roadmap | calibrated per-node predictive uncertainty — know where to trust the surrogate, where to fall back to CFD |
| **Agentic geometry optimization** | 🔜 roadmap | a closed design loop: agent proposes geometry edits, surrogate scores them in ms, CFD verifies the winner |

## Table of contents

- [Results](#results)
- [Leaderboard](#leaderboard--simshift-heatsink-medium--ood)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [The surrogate](#the-surrogate)
- [Roadmap](#roadmap)
- [Repository layout](#repository-layout)
- [Reproducing our numbers](#reproducing-our-numbers)
- [Citing](#citing)
- [License & acknowledgements](#license--acknowledgements)

## Results

All numbers are measured and reproducible with the code here; the honest caveats (data imbalance, single OOD axis, sub-second inference is an estimate) are stated inline and in [`docs/RESULTS.md`](docs/RESULTS.md). We would rather under-claim.

### Leaderboard — SIMSHIFT heatsink (medium / OOD)

On the public [SIMSHIFT](https://github.com/psetinek/simshift) heatsink split (train fin counts 5–8 → test 10–12), the surrogate leads on temperature — the field that matters for thermal design — **without the unsupervised domain adaptation (UDA) every published baseline relies on**.

![SIMSHIFT heatsink leaderboard — temperature RMSE](assets/fig_leaderboard.svg)

| # | model | UDA | Temp RMSE (K) ↓ | Velocity RMSE (m/s) ↓ | params |
|---|---|:---:|:---:|:---:|:---:|
| 🥇 | **Kuber — SurfaceGeoTransolver** | ✗ | **12.14** | 0.044 | 14.3 M |
| 🥈 | UPT *(prev. published best)* | ✓ | 12.41 | 0.039 | ~14 M¹ |
| 🥉 | **Kuber — GeoTransolver + directional-SDF** | ✗ | 13.07 | **0.038** | 14.0 M |
| 4 | Transolver | ✓ | 13.43 | 0.041 | ~14 M¹ |
| 5 | PointNet | ✓ | 17.43 | 0.044 | ~14 M¹ |

¹ The SIMSHIFT paper prints no parameter counts; its configs show all three baselines are comparably sized (~10–15 M). **We are not winning by scaling parameters** — the edge is geometry conditioning + training recipe at the same budget. On the paper's *primary averaged-NRMSE* metric the SDF model actually leads overall (0.593); the surface model's win is specifically on **temperature**. Every metric, every caveat: [`docs/RESULTS.md`](docs/RESULTS.md). **Want on the board?** → [How to submit](docs/BENCHMARK.md#submitting-a-result).

### Generalization — the SOTA number is zero-shot

The leaderboard result is an **out-of-distribution** result: fin counts 10–14 never appear in training. The surrogate predicts them zero-shot and still leads.

![In-distribution vs out-of-distribution temperature RMSE](assets/fig_indist_vs_ood.svg)

### Our data is a moat — pretraining helps

Same model, same SIMSHIFT fine-tuning; the only added ingredient is pretraining on our self-generated corpus. It moves the OOD number down — materially at easy shift (−19 %) and in-distribution (−12 %).

![Value of our data — from scratch vs pretrained on the Kuber corpus](assets/fig_value_of_data.svg)

### It doesn't blow up at the edges

At fin tips and corners the true temperature gradient is steep — where brittle surrogates either over-smooth or emit unphysical spikes. Measured predicted-vs-CFD ∇T in the near-wall band: **every ratio ≤ 1.0** (faithful, never exploding), **explosion fraction 0, zero NaN/Inf**, in- and out-of-distribution.

![Stability proof — edge temperature-gradient fidelity](assets/fig_stability.svg)

### ~1,000× faster than the CFD it learns from

![Speed — surrogate vs CFD, log scale](assets/fig_speed.svg)

## Dataset

A self-generated OpenFOAM CHT corpus — **0 cases from SIMSHIFT or any licensed source** — free for research and commercial use.

![The Kuber corpus at a glance](assets/fig_corpus.svg)

Pipeline ([`datagen/`](datagen)): parametric generator → STL → `blockMesh` + `snappyHexMesh` (+ prism layers) → solve → `.npz`, resumable, gated by convergence + a physics filter. A 6-case sample (3 heatsinks + 3 cold plates) is in [`data_sample/`](data_sample); the full contract is in [`docs/DATASET.md`](docs/DATASET.md).

**Fidelity is verified, not assumed** — prism layers recover the near-wall hot spot to within 0.1 K of a fine mesh at ~2.7× lower cost:

![Mesh convergence of the hot spot](assets/fig_mesh_convergence.svg)

## Installation

```bash
git clone https://github.com/ShubhJain007/Kuber.git
cd Kuber
python -m venv .venv && source .venv/bin/activate     # or use conda
pip install -r requirements.txt
```

The surrogate core is **GeoTransolver** from NVIDIA PhysicsNeMo (`physicsnemo.experimental.models.geotransolver`) — install per its [instructions](https://github.com/NVIDIA/physicsnemo); a CUDA GPU is recommended for training and required for the multiscale ball-query core. **Data generation** additionally needs [OpenFOAM](https://www.openfoam.com/) (v2306+) on your `PATH`; the model, eval, and the data sample work without it.

## Quickstart

```bash
# 1. Peek at a sample case (no OpenFOAM, no GPU needed)
python -c "import numpy as np, os; d=np.load('data_sample/'+sorted(os.listdir('data_sample'))[0]); \
           print({k:getattr(d[k],'shape',None) for k in d.files})"

# 2. Evaluate a trained checkpoint (geometry mode + conditioning are read from it)
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

`--geom_mode {none,sdf,dsdf,surface}` selects the geometry representation; `--refine` adds the generative PDE-Refiner head. See `python -m kuber.train_simshift --help`.

## The surrogate

**SurfaceGeoTransolver** = a GeoTransolver physics-attention core (256 hidden × 12 layers) + an AB-UPT-style surface encoder (surface point cloud + normals → geometry tokens → per-node descriptor via kNN cross-attention). Geometry can also be fed as a scalar SDF, a **directional SDF** (distance + unit gradient), or nothing (conditions only), chosen by `--geom_mode`. An optional **PDE-Refiner** head restores high-frequency content and yields uncertainty — the seed of the Bayesian-UQ pillar on the roadmap. Architecture card: [`docs/MODEL.md`](docs/MODEL.md).

## Roadmap

The three shipping pillars (data engine, surrogate, eval harness) are the foundation. The suite becomes a *design tool* with:

1. **Connectors — bring your own geometry.** Native ingest of STEP / STL / CAD and OpenFOAM cases, plus export back into standard thermal workflows, exposed as a Python API + CLI. The surface-cloud interface already accepts arbitrary meshes; connectors make it turnkey so the surrogate drops into an existing pipeline without hand-conversion.
2. **Bayesian uncertainty predictor.** Calibrated, per-node predictive uncertainty so an engineer knows *where* to trust the surrogate and where to fall back to CFD. Builds on the existing PDE-Refiner sampling, extended with deep-ensemble / variational / conformal calibration — and it feeds active learning (below).
3. **Agentic geometry optimization.** A closed design loop: an agent proposes geometry edits (fin pitch/height, channel routing, pin layout), the surrogate scores thermal + pressure-drop objectives in milliseconds, and the agent searches the design space — with CFD-in-the-loop only to verify the winner. This is where a fast predictor becomes a fast *designer*.
4. **And next:** active learning (uncertainty flags the cases worth simulating → the data engine targets them); more device classes / topologies (serpentine, pin-fin, parallel cold plates; immersion; vapor chambers); leave-one-shape-out / leave-one-fluid-out OOD splits; a hosted leaderboard with a sealed test set; and released checkpoints as GitHub Release assets.

Contributions to any of these are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Repository layout

```
Kuber/
├── kuber/                     # the surrogate + harness (Python package)
│   ├── surface_geotransolver.py   # SurfaceGeoTransolver (surface branch + GeoTransolver core)
│   ├── surface_model.py           # surface encoder + local cross-attention
│   ├── surface_geom.py            # analytic surface point cloud + normals from parameters
│   ├── pde_refiner.py             # PDE-Refiner (generative refinement head)
│   ├── train_simshift.py          # training + eval + all geometry modes + data loading
│   └── edge_proof.py              # gradient-fidelity / no-explosion stability harness
├── datagen/                   # OpenFOAM buoyantSimpleFoam CHT pipeline (heatsink + cold plate)
├── data_sample/              # 6 example cases (.npz) + a sample split
├── assets/                   # result figures (SVG) + their generator
├── checkpoints/             # how to obtain trained weights
├── docs/                    # RESULTS, DATASET, MODEL, BENCHMARK
└── scripts/                 # convenience wrappers
```

## Reproducing our numbers

Every number above is produced by the code here (SIMSHIFT data + splits from the [official repo](https://github.com/psetinek/simshift); our corpus regenerated with `datagen/`):

```bash
# leaderboard (surface + SDF variants)
python -m kuber.train_simshift --data <simshift> --splits <splits> --difficulty medium --geom_mode surface
python -m kuber.train_simshift --data <simshift> --splits <splits> --difficulty medium --geom_mode dsdf
# value-of-data A/B: pretrain on our corpus (reduced "transfer" conditioning), then fine-tune
python -m kuber.train_simshift --data <our_corpus> --splits <corpus_splits> --difficulty medium \
    --geom_mode surface --drop_geom_scalars --out pretrain/
python -m kuber.train_simshift --data <simshift> --splits <splits> --difficulty medium \
    --geom_mode surface --drop_geom_scalars --init_from pretrain/<pretrained.pt>
# stability / no-explosion proof
python -m kuber.edge_proof --ckpt <model.pt> --data <simshift> --splits <splits> --difficulty medium
```

Regenerate the figures in this README with `python assets/make_figures.py`.

## Citing

```bibtex
@software{kuber2026,
  title  = {Kuber: An Engineering-AI Suite for Electronics-Cooling Thermal Simulation},
  author = {Jain, Shubh},
  year   = {2026},
  url    = {https://github.com/ShubhJain007/Kuber}
}
```

Please also cite the works Kuber builds on: **GeoTransolver / PhysicsNeMo** (NVIDIA), **Transolver** (Wu et al., 2024), **AB-UPT** (Alkin et al., 2025), **SIMSHIFT** (Setinek et al.), and **PDE-Refiner** (Lippe et al., NeurIPS 2023). Full references in [`docs/MODEL.md`](docs/MODEL.md).

## License & acknowledgements

Apache-2.0 — see [`LICENSE`](LICENSE). The GeoTransolver core is from NVIDIA PhysicsNeMo (Apache-2.0). The dataset is self-generated with OpenFOAM and released for research and commercial use.

<div align="center">
<sub>Built by the Kuber.ai team — geometry-general thermal surrogates for electronics cooling.</sub>
</div>
