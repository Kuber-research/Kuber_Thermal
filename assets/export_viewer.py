#!/usr/bin/env python3
"""Export ground-truth + Kuber-prediction data for the interactive viewer.

Runs the trained multi-geometry model over one heatsink and one cold-plate case and writes
`assets/viewer_data.json` (consumed by build_viewer.py: coords, GT + predicted T and velocity,
and the geometry boxes). Provenance/reference script — it needs a trained checkpoint + the corpus
+ the inference `Engine` wrapper (not shipped in this repo; see checkpoints/ and datagen/). Paths
are env-var placeholders. Run in the PhysicsNeMo env:

    KUBER_CKPT=multigeo.pt KUBER_DATA=corpus/ KUBER_SPLITS=splits.json python assets/export_viewer.py
"""
import os, json

CKPT = os.environ.get("KUBER_CKPT", "multigeo.pt")
CORPUS = os.environ.get("KUBER_DATA", "corpus")
SPLITS = os.environ.get("KUBER_SPLITS", "splits.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_data.json")

from infer import Engine  # the demo inference wrapper: loads model + corpus, predicts GT + pred + geometry

e = Engine(CKPT, CORPUS, SPLITS)                 # device="cpu" if the GPU is busy
cases = e.list_cases()
hs_pick = max((c for c in cases if c["device"] == "heatsink"), key=lambda c: c.get("solidTemp", 0))
cp_pick = max((c for c in cases if c["device"] == "coldplate"), key=lambda c: c.get("heatFlux", 0))

out = {}
for key, pick in [("heatsink", hs_pick), ("coldplate", cp_pick)]:
    d = e.predict(pick["index"], max_points=8000)   # coords, fields{T,velocity}{pred,gt}, boxes, metrics
    d["title"], d["sub"] = pick["title"], pick["sub"]
    d["fields"].pop("p_rgh", None)
    out[key] = d
    m = d["metrics"]
    print(f"{key}: {pick['title']} | T_rmse {m['T_rmse']} K | peak_T {m['peak_T']}/{m['peak_T_gt']} | {d['n_points']} pts")

json.dump(out, open(OUT, "w"))
print("saved", os.path.getsize(OUT) // 1024, "KB ->", OUT)
