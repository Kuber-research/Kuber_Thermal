# The Kuber corpus

A self-generated OpenFOAM conjugate-heat-transfer (CHT) corpus for electronics cooling —
**0 cases from SIMSHIFT or any licensed/scraped source**. Everything here is produced by the
pipeline in [`../datagen/`](../datagen) and is free for research and other noncommercial use.

## What a case is

One `.npz` per case, sampled to a fixed **16 384-node** point cloud in the fluid domain:

| key | shape | meaning |
|---|---|---|
| `coords` | `[16384, 3]` | node positions (metres), physical frame |
| `U` | `[16384, 3]` | velocity `(U_x, U_y, U_z)` (m/s) |
| `T` | `[16384, 1]` | temperature (K) |
| `p_rgh` | `[16384, 1]` | reduced pressure `p − ρgh` (Pa) |
| `surf_pts` | `[2048, 3]` | solid-boundary surface point cloud (the geometry input) |
| `surf_normals` | `[2048, 3]` | outward unit normals at `surf_pts` |
| `sdf` | `[16384]` | signed distance to the solid (heatsink cases; negative inside) — optional |
| `sdf_grad` | `[16384, 3]` | unit SDF gradient (directional-SDF mode) — optional |
| `conditions` | `()` object | JSON dict of operating conditions (see below) |

`conditions` carries the physics the model is conditioned on. Across device classes the unified keys are:

```
rho, mu, Cp, Pr          # fluid properties
u_in                     # inlet velocity (0 for natural convection)
envTemp                  # ambient / inlet temperature (K)
solidTemp                # wall temperature BC (heatsink; K)   — 0 when a flux BC is used
heatFlux                 # heat-flux BC (cold plate; W/m^2)    — 0 when a wall-temp BC is used
device                   # 0 = heatsink, 1 = cold plate
# heatsinks additionally carry geometry scalars: fins, gap, height1, height2, length, width, thickness_fins
```

## Coverage

| axis | coverage |
|---|---|
| fluids | air, water, mineral oil (Pr≈292), glycol — conditioned on Pr, ρ, Cp, μ |
| regimes | natural + forced convection |
| shapes | fins, plate, cube, pin-fin arrays (heatsinks); straight-channel ducts (cold plates) |
| conditions | ambient 290–310 K, wall 340–400 K, fin count 5–14, varied geometry |
| device classes | heatsink (wall-temp BC, buoyancy) and cold plate (heat-flux BC, forced liquid) |
| fidelity | ~1.4 M cells/case (snap-level-2 + 3 prism layers), subsampled to 16 384 nodes |

**Honest limits.** The corpus is air-dominated; oil/glycol are thin (~30–40 cases each), so the model
is strongest on air. The primary OOD axis is fin count; fluids/shapes appear in both train and test.
More liquid data and leave-one-out splits are the roadmap.

## How it is generated

Pipeline (`../datagen/`), fully resumable, gated by convergence + a physics filter:

```
parametric generator  →  STL  →  blockMesh + snappyHexMesh (+ prism layers)  →  buoyantSimpleFoam  →  .npz
     gen_bsf.py           make_stl.py      mesh_geom.py                          run_sweep_bsf.py    to_npz_bsf.py
```

- **Solver:** OpenFOAM `buoyantSimpleFoam` — single fluid region, the solid modelled as a heated wall
  (heatsink) or a heat-flux surface (cold plate).
- **Validity gate:** a case is written only after a converged solve; a physics filter rejects
  unphysical results (e.g. hot cold plates exceeding a temperature ceiling with no boiling model).
- **Resumable:** re-run the same command to continue; `status.json` is rewritten after every case.

### Mesh-convergence check

Prism layers recover the near-wall hot spot to within **0.1 K of a fine mesh at ~2.7× fewer cells**,
which is why the production corpus uses snap-level-2 + 3 layers:

| mesh | cells | T-max (hot spot) |
|---|---|---|
| snap2 (no layers) | 124 k | 359.5 K |
| **snap2 + 3 prism layers** | 142 k | **378.8 K** |
| snap3 (fine) | 382 k | 378.9 K |

## Sample

[`../data_sample/`](../data_sample) contains 6 ready-to-load cases (3 heatsinks, 3 cold plates) plus a
sample split file. Load one:

```python
import numpy as np, json, os
d = np.load("data_sample/" + sorted(os.listdir("data_sample"))[0], allow_pickle=True)
print({k: getattr(d[k], "shape", None) for k in d.files})
print(json.loads(str(d["conditions"])))
```

## License

The corpus is generated with OpenFOAM (GPL solver, but the *output data* is yours) and is released
under the repository's PolyForm Noncommercial License 1.0.0 (noncommercial use; commercial use
requires a separate license). If you extend it, please keep the no-licensed-data rule
(see [`../CONTRIBUTING.md`](../CONTRIBUTING.md)).
