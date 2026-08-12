# Data generation — OpenFOAM CHT pipeline

Generates the Kuber corpus: parametric electronics-cooling geometries solved with OpenFOAM
`buoyantSimpleFoam`, exported to per-case `.npz`. Fully resumable, gated by convergence + a physics
filter. **Requires OpenFOAM** (v2306+ with `buoyantSimpleFoam`, `blockMesh`, `snappyHexMesh`) on your
`PATH`; the model/eval code does not.

## Pipeline

```
gen_bsf.py        sample parametric heatsink cases (fins/plate/cube/pin-fin, fluids, BCs)   → params
make_stl.py       parametric geometry → STL solid
mesh_geom.py      blockMesh + snappyHexMesh (+ prism layers)
run_sweep_bsf.py  resumable parallel driver: mesh → solve → filter → prune raw
to_npz_bsf.py     converged solve → 16384-node .npz (coords, U, T, p_rgh, surface, sdf, conditions)
check_bsf.py      convergence + physics validity gate
```

Cold-plate variants (heat-flux BC, forced liquid duct): `gen_coldplate.py`, `build_coldplate.py`,
`adapt_coldplate.py`, `check_coldplate.py`. `adapt_bsf.py` maps raw OpenFOAM output into the unified
conditioning schema (see [`../docs/DATASET.md`](../docs/DATASET.md)).

## Run

```bash
# 1. sample a batch of cases
python gen_bsf.py --n 200 --out cases/

# 2. mesh + solve + export, resumable and parallel
python -u run_sweep_bsf.py --cases cases/ --out corpus/ \
    --scripts <bsf_scripts_dir> --jobs 5 --iters 800 --timeout 5400

# re-run the exact same command to resume; status.json tracks progress
```

Each converged case lands in `corpus/<id>.npz`. Rejected cases (non-converged or unphysical) are moved
aside, not written. See `python run_sweep_bsf.py --help` for all flags.

## Notes

- **Validity:** a case is written only after a converged solve; the physics filter rejects unphysical
  fields (e.g. cold plates exceeding a temperature ceiling — there is no boiling model).
- **Fidelity:** snap-level-2 + 3 prism layers matches a fine mesh's hot spot within 0.1 K at ~2.7×
  lower cost ([`../docs/DATASET.md`](../docs/DATASET.md)).
- **Determinism:** geometry/condition sampling is seeded; re-running reproduces the same corpus.
