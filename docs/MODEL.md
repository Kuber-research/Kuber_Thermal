# SurfaceGeoTransolver — architecture card

A geometry-conditioned neural operator that predicts the full steady CHT field at an arbitrary query
point cloud. It pairs a **GeoTransolver** physics-attention core (NVIDIA PhysicsNeMo, Apache-2.0) with
an **AB-UPT-style surface-geometry encoder**, and can instead take geometry as a signed-distance field
or nothing at all — selected at the command line by `--geom_mode`.

Source: [`../kuber/surface_geotransolver.py`](../kuber/surface_geotransolver.py),
[`surface_model.py`](../kuber/surface_model.py),
[`surface_geom.py`](../kuber/surface_geom.py),
[`pde_refiner.py`](../kuber/pde_refiner.py).

## Inputs / output

```
inputs :  conditions  [B,N,C]   physics scalars, broadcast per query node
          vol_coords  [B,N,3]   query point cloud (per-case normalized to [0,1])
          geometry              one of:
                                  surface cloud + normals  [B,Ns,3]+[B,Ns,3]   (--geom_mode surface)
                                  (directional) SDF scalar/field appended to conditions (--geom_mode sdf|dsdf)
                                  nothing                                       (--geom_mode none)
output :  field       [B,N,5]   (U_x, U_y, U_z, T, p_rgh), per-channel z-normalized
```

## Geometry wiring

| `--geom_mode` | geometry signal | notes |
|---|---|---|
| `none` | conditions only | ablation floor |
| `sdf` | scalar signed distance per node | cheap, strong when an analytic SDF exists |
| `dsdf` | signed distance **+ unit gradient** (4 ch) | tells the model *which way the wall is* → windward/leeward points differ; **best accuracy/velocity at our data scale** |
| `surface` | surface point cloud + normals → encoder → per-node descriptor | no analytic SDF needed → the path to **arbitrary CAD**; **best temperature** |

The surface branch (`geom_wiring="concat"`, the default) runs a shallow permutation-invariant
**SurfaceGeometryEncoder** over `(surf_pts, surf_normals)` to produce geometry tokens, then a
**LocalSurfaceCrossAttention** (kNN, k=16) turns them into a per-query-node geometry descriptor that is
concatenated into GeoTransolver's local embedding. A `"deep"` wiring instead feeds the raw surface
cloud to GeoTransolver's native per-block cross-attention (AB-UPT-style deep conditioning).

## Core

GeoTransolver: physics-attention transformer, **256 hidden × 12 layers**, 8 heads, 64 slices,
multiscale local ball-query features (radii 0.05/0.25, 8/32 neighbours). ~14 M parameters total
(surface branch adds ~0.3 M over the SDF variants).

## Optional: PDE-Refiner head

One-shot MSE is spectrally biased — it drops low-amplitude high-frequency modes (fin-tip gradients).
The optional **PDE-Refiner** (`--refine`) produces the field as an initial prediction + K denoising
refinement steps at exponentially decreasing noise, forcing the network to model all scales, and
yields an uncertainty estimate. Faithful adaptation of Lippe et al. (NeurIPS 2023) to steady fields;
see [`pde_refiner.py`](../kuber/pde_refiner.py).

## Which variant should I use?

- **Parametric / known geometry family** → `dsdf`. An exact analytic SDF is a near-perfect,
  data-efficient signal; at our current data scale it is the most accurate and the cheapest.
- **Arbitrary customer CAD** → `surface`. No analytic SDF exists for general meshes, so the surface
  encoder becomes necessary. It already wins on temperature; more data closes the rest (see
  [`RESULTS.md` §6](RESULTS.md)).

## References

- **GeoTransolver / PhysicsNeMo** — NVIDIA. https://github.com/NVIDIA/physicsnemo
- **Transolver** — Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024.
- **AB-UPT** — Alkin et al., *Anchored / Branched Universal Physics Transformers*, 2025. https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers
- **SIMSHIFT** — Setinek et al., *A Benchmark for Distribution Shift in Simulation*. https://github.com/psetinek/simshift
- **PDE-Refiner** — Lippe, Perdikaris, Brandstetter, Cranmer et al., NeurIPS 2023.
