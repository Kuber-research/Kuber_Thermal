"""Parametric case generator for the buoyantSimpleFoam heatsink sweep.

LHS over SIMSHIFT-matched ranges. Key geometric constraint: the fin GAP is
DERIVED from base width + fin count + fin thickness -> g = (W - N*t_f)/(N-1).
This both guarantees the fins fit the base AND naturally reproduces SIMSHIFT's
gap range (0.0023-0.01625 m). Also emits some plate and cube shapes.

Writes <out>/case_XXXX/params.json and <out>/manifest.json.

Usage: python gen_bsf.py --n 60 --out cases --seed 0
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

# fixed domain + mesh (metres)
DOMAIN = dict(Lx=0.5, Ly=0.14, Lz=0.14, cx=0.25, cz=0.07,
              bg_cell=0.005, snap_level=3, T_amb=300.0, turb="kOmegaSST")

# sampled ranges  (name: (lo, hi))
RANGES = dict(
    N=(5, 14),            # fins (int) — in-dist 5-9, OOD 10-14 (SIMSHIFT split_param)
    W=(0.06, 0.10),       # base width (fins span this; gap derived)
    L=(0.08, 0.12),       # base length (streamwise)
    t_b=(0.004, 0.008),   # base thickness
    t_f=(0.003, 0.004),   # fin thickness (SIMSHIFT)
    h_f=(0.053, 0.083),   # fin height   (SIMSHIFT)
    T_wall=(340.0, 400.0),# heated wall temperature (SIMSHIFT)
    u_in=(0.05, 0.5),     # inlet velocity (SIMSHIFT range unknown; natural->mild forced)
)
G_MIN = 0.0018            # reject cases whose derived gap is too small to mesh


def lhs(n, d, seed):
    rng = np.random.default_rng(seed)
    s = np.zeros((n, d))
    for j in range(d):
        cut = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(cut)
        s[:, j] = cut
    return s, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plate_frac", type=float, default=0.1)
    ap.add_argument("--cube_frac", type=float, default=0.1)
    ap.add_argument("--pinfin_frac", type=float, default=0.0)
    ap.add_argument("--fluid", choices=["air", "water", "oil", "glycol", "mixed"], default="air",
                    help="coolant: air (natural conv), water/oil/glycol (forced liquid cold-plate), or mixed")
    ap.add_argument("--snap_level", type=int, default=None,
                    help="override snappyHexMesh refinement level (default 3; use 2 for faster/coarser)")
    ap.add_argument("--regime", choices=["natural", "forced", "mixed"], default="mixed",
                    help="convection regime -> inlet velocity range (natural=buoyancy-driven low speed, "
                         "forced=high-speed forced convection)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    keys = list(RANGES.keys())
    S, rng = lhs(a.n, len(keys), a.seed)
    manifest = []
    made = 0
    for i in range(a.n):
        p = dict(DOMAIN)
        if a.snap_level is not None:
            p["snap_level"] = a.snap_level
        p["fluid"] = rng.choice(["air", "water"]) if a.fluid == "mixed" else a.fluid
        for j, k in enumerate(keys):
            lo, hi = RANGES[k]
            v = lo + S[i, j] * (hi - lo)
            p[k] = int(round(v)) if k == "N" else float(v)
        # inlet velocity by regime (overrides the default RANGES u_in)
        regime_uin = {"natural": (0.02, 0.3), "forced": (1.0, 5.0), "mixed": (0.05, 0.5)}
        lo_u, hi_u = regime_uin[a.regime]
        us = S[i, keys.index("u_in")]
        p["u_in"] = float(lo_u + us * (hi_u - lo_u))
        p["regime"] = a.regime
        if p["fluid"] in ("water", "oil", "glycol"):   # liquid cold-plate: forced convection
            p["u_in"] = float(0.2 + us * (1.0 - 0.2))  # 0.2-1.0 m/s
        # ambient / coolant inlet temperature (realistic room/coolant range)
        p["T_amb"] = float(rng.uniform(290.0, 310.0))
        # decide shape
        r = rng.random()
        f_plate, f_cube, f_pin = a.plate_frac, a.cube_frac, a.pinfin_frac
        if r < f_plate:
            p["shape"] = "plate"           # base slab only (flat plate)
            p["g"] = 0.0; p["h_f"] = 0.0
        elif r < f_plate + f_cube:
            p["shape"] = "cube"            # compact heated block
            p["L"] = float(rng.uniform(0.03, 0.06))
            p["W"] = float(rng.uniform(0.03, 0.06))
            p["t_b"] = float(rng.uniform(0.03, 0.06))   # cube height
            p["g"] = 0.0; p["h_f"] = 0.0
        elif r < f_plate + f_cube + f_pin:
            p["shape"] = "pinfin"          # grid of square pins (pin size t_f, height h_f)
            p["g"] = float(rng.uniform(0.004, 0.012))   # pin pitch gap
        else:
            p["shape"] = "fins"
            g = (p["W"] - p["N"] * p["t_f"]) / (p["N"] - 1)
            if g < G_MIN:                  # widen base so fins fit with a mesh-able gap
                p["W"] = p["N"] * p["t_f"] + (p["N"] - 1) * G_MIN * 1.5
                g = (p["W"] - p["N"] * p["t_f"]) / (p["N"] - 1)
            p["g"] = float(g)
        cid = f"case_{made:04d}"
        (out / cid).mkdir(exist_ok=True)
        (out / cid / "params.json").write_text(json.dumps(p, indent=2))
        rec = dict(case_id=cid, fluid=p["fluid"], shape=p["shape"], N=p.get("N"), W=round(p["W"], 4),
                   g=round(p.get("g", 0), 5), h_f=round(p["h_f"], 4),
                   T_wall=round(p["T_wall"], 1), u_in=round(p["u_in"], 3),
                   ood="ood" if (p["shape"] == "fins" and p["N"] >= 10) else "in_dist")
        manifest.append(rec)
        made += 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    nf = sum(1 for m in manifest if m["shape"] == "fins")
    print(f"[gen] wrote {made} cases -> {out}  (fins={nf}, "
          f"plate={sum(1 for m in manifest if m['shape']=='plate')}, "
          f"cube={sum(1 for m in manifest if m['shape']=='cube')}, "
          f"ood_fins={sum(1 for m in manifest if m['ood']=='ood')})")


if __name__ == "__main__":
    main()
