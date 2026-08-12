"""Adapt our buoyantSimpleFoam npz -> SIMSHIFT-format npz + OOD splits.

Our npz:      coords[N,3], T[N], U[N,3], p_rgh[N], params(JSON str)
SIMSHIFT fmt: coords[Ns,3], U[Ns,3], T[Ns,1], p_rgh[Ns,1], conditions(JSON str)
              all cases share Ns=16384 (subsampled); splits.json selects train/val/test.

For the first BASELINE we take AIR + NATURAL + FIN cases only (most homogeneous,
SIMSHIFT-like) and split OOD by fin count: src = fins 5-9, tgt(OOD) = fins 10-14.

Usage:
  python adapt_bsf.py --in bsf_out_night bsf_out_airnatural_L bsf_out_ov_air_natural_s200 \
      --out train_air_natural --splits splits_air_natural.json --n 16384
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_stl import heatsink_boxes                 # exact box geometry in OUR frame
from mesh_geom import sample_mesh                    # full triangulated-mesh surface sampler

# geometry + BC condition columns
GEOM_KEYS = ["N", "g", "t_f", "h_f", "t_b", "L", "W", "T_wall", "T_amb", "u_in"]
# fluid thermophysical properties for conditioning (Pr, rho, Cp, mu) — lets ONE model
# span air/water/oil/glycol (same geometry, very different physics)
FLUID_PROPS = {  # Pr,     rho,    Cp,     mu
    "air":    (0.705,  1.18,   1004.4, 1.831e-5),
    "water":  (6.1,    997.0,  4181.0, 8.9e-4),
    "oil":    (292.0,  850.0,  1900.0, 2.0e-2),
    "glycol": (29.0,   1070.0, 3300.0, 3.5e-3),
}


def _box_sdf(p, lo, hi):
    """Signed distance from points p[N,3] to an axis-aligned box [lo,hi]."""
    c = (lo + hi) / 2.0; h = (hi - lo) / 2.0
    q = np.abs(p - c) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def bsf_sdf(coords, params):
    """Exact signed distance to the union of heatsink boxes (any shape), OUR frame."""
    sdf = np.full(coords.shape[0], 1e9, np.float64)
    for (x0, x1, y0, y1, z0, z1) in heatsink_boxes(params):
        sdf = np.minimum(sdf, _box_sdf(coords, np.array([x0, y0, z0]), np.array([x1, y1, z1])))
    return sdf


def bsf_sdf_grad(coords, params, eps=1e-4):
    """Directional SDF: signed distance + unit gradient (which way the wall is)."""
    sdf = bsf_sdf(coords, params)
    g = np.zeros((coords.shape[0], 3), np.float64)
    for ax in range(3):
        d = np.zeros(3); d[ax] = eps
        g[:, ax] = (bsf_sdf(coords + d, params) - bsf_sdf(coords - d, params)) / (2 * eps)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return sdf.astype(np.float32), (g / np.where(n > 1e-9, n, 1.0)).astype(np.float32)


def sample_surface(params, n_surf, rng):
    """Sample exactly n_surf points on the heatsink surface (union of box faces) with
    outward unit normals. Points interior to the union (e.g. fin-base overlap) are
    discarded via the SDF so only the true outer surface remains."""
    boxes = heatsink_boxes(params)
    P, Nn = [], []
    over = 6 * n_surf
    for (x0, x1, y0, y1, z0, z1) in boxes:
        dims = [(x1 - x0), (y1 - y0), (z1 - z0)]
        faces = [(0, x0, -1), (0, x1, 1), (1, y0, -1), (1, y1, 1), (2, z0, -1), (2, z1, 1)]
        for axis, val, sgn in faces:
            a = [(axis + 1) % 3, (axis + 2) % 3]
            area = dims[a[0]] * dims[a[1]]
            k = max(1, int(over * area / (sum(dims[i] * dims[j]
                     for i in range(3) for j in range(3) if i < j) * 2 * len(boxes) + 1e-12)))
            pts = np.zeros((k, 3)); pts[:, axis] = val
            lo0 = [x0, y0, z0]; hi0 = [x1, y1, z1]
            pts[:, a[0]] = rng.uniform(lo0[a[0]], hi0[a[0]], k)
            pts[:, a[1]] = rng.uniform(lo0[a[1]], hi0[a[1]], k)
            nrm = np.zeros((k, 3)); nrm[:, axis] = sgn
            P.append(pts); Nn.append(nrm)
    P = np.concatenate(P); Nn = np.concatenate(Nn)
    keep = bsf_sdf(P, params) > -1e-4                 # drop points inside the union
    P, Nn = P[keep], Nn[keep]
    if len(P) == 0:                                   # degenerate fallback
        P = np.array([[params["cx"], params["t_b"], params["cz"]]]); Nn = np.array([[0., 1., 0.]])
    idx = rng.choice(len(P), n_surf, replace=len(P) < n_surf)
    return P[idx].astype(np.float32), Nn[idx].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True, help="input bsf_out_* dirs")
    ap.add_argument("--out", required=True, help="output dir for adapted npz")
    ap.add_argument("--splits", required=True, help="output splits.json path")
    ap.add_argument("--n", type=int, default=16384, help="subsample node count")
    ap.add_argument("--n_surf", type=int, default=2048, help="surface points per case (geom encoder)")
    ap.add_argument("--src_fins", type=int, nargs=2, default=[5, 9], help="in-dist fin range")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    kept, fin_of = [], {}
    n_scanned = n_skip = 0
    for d in a.inp:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            n_scanned += 1
            z = np.load(f, allow_pickle=True)
            p = json.loads(str(z["params"]))
            fluid = p.get("fluid", "air"); regime = p.get("regime", "natural")
            shape = p.get("shape", "fins")
            N = int(p.get("N", 0))
            coords = np.asarray(z["coords"], np.float32)
            M = coords.shape[0]
            if M < a.n:
                n_skip += 1; continue                     # too few nodes to subsample
            idx = rng.choice(M, a.n, replace=False)
            T = np.asarray(z["T"], np.float32).reshape(-1)[idx][:, None]
            U = np.asarray(z["U"], np.float32)[idx]
            prgh = np.asarray(z["p_rgh"], np.float32).reshape(-1)[idx][:, None]
            pr, rho, cp, mu = FLUID_PROPS.get(fluid, FLUID_PROPS["air"])
            cond = {k: float(p.get(k, 0.0)) for k in GEOM_KEYS}
            cond.update({"Pr": pr, "rho": rho, "Cp": cp, "mu": mu,
                         "forced": 1.0 if regime == "forced" else 0.0})
            # SIMSHIFT-compatible aliases -> enables pretrain->finetune transfer on a
            # shared conditioning (wall/ambient temperature) via --drop_geom_scalars
            cond["solidTemp"] = float(p.get("T_wall", 0.0))
            cond["envTemp"] = float(p.get("T_amb", 0.0))
            # full SIMSHIFT-named geometry conditioning so pretrain+finetune can share the
            # exact 5-scalar cond_keys (fins,gap,height2,thickness_fins,solidTemp) -> full-cond transfer
            cond["fins"] = float(p.get("N", 0.0))
            cond["gap"] = float(p.get("g", 0.0))
            cond["height2"] = float(p.get("h_f", 0.0))
            cond["thickness_fins"] = float(p.get("t_f", 0.0))
            cond["height1"] = float(p.get("t_b", 0.0))
            cond["length"] = float(p.get("L", 0.0))
            cond["width"] = float(p.get("W", 0.0))
            cs = coords[idx].astype(np.float64)
            sdf, sdf_grad = bsf_sdf_grad(cs, p)       # exact SDF + unit gradient in our frame
            surf_pts, surf_nrm = sample_mesh(p, a.n_surf, seed=a.seed)  # full-mesh surface cloud
            # unique id from source dir + case name
            sid = f"{Path(f).parent.name}__{Path(f).stem}"
            np.savez_compressed(out / f"{sid}.npz",
                                coords=coords[idx], U=U, T=T, p_rgh=prgh,
                                sdf=sdf, sdf_grad=sdf_grad,
                                surf_pts=surf_pts, surf_normals=surf_nrm,
                                conditions=json.dumps(cond))
            kept.append(sid); fin_of[sid] = N if shape == "fins" else -1
    # OOD split: held-out HIGH-fin-count fins (tgt); everything else (non-fins +
    # low-fin fins, all fluids/regimes/shapes) is in-distribution (src)
    lo, hi = a.src_fins
    tgt = [s for s in kept if fin_of[s] > hi]
    src = [s for s in kept if s not in set(tgt)]
    rng.shuffle(src)
    n = len(src); n_tr = int(0.7 * n); n_va = int(0.15 * n)
    splits = {"medium": {           # loader indexes by difficulty key
        "src": {"train": src[:n_tr], "val": src[n_tr:n_tr + n_va], "test": src[n_tr + n_va:]},
        "tgt": {"train": [], "val": [], "test": tgt}}}
    Path(a.splits).write_text(json.dumps(splits, indent=2))
    print(f"[adapt] scanned={n_scanned} kept={len(kept)} skipped={n_skip}")
    from collections import Counter
    print(f"[adapt] fin-count dist: {dict(sorted(Counter(fin_of.values()).items()))}")
    print(f"[adapt] src(in-dist {lo}-{hi})={len(src)} "
          f"(train {n_tr}/val {n_va}/test {n - n_tr - n_va})  tgt(OOD >{hi})={len(tgt)}")
    print(f"[adapt] wrote npz -> {out}  splits -> {a.splits}")


if __name__ == "__main__":
    main()
