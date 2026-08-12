"""Adapt cold-plate buoyantSimpleFoam npz -> SIMSHIFT-format npz for MULTI-GEOMETRY training.

Cold plate = rectangular coolant duct (domain IS the coolant): x in [0,L] (streamwise),
y in [0,H] (thin height), z in [0,W] (width). Generates the CHANNEL-WALL surface cloud
+ inward normals (the geometry the surface encoder reads), and UNIFIED physics conditioning
shared with heatsinks so ONE model spans both device classes via the surface cloud:
    rho, mu, Cp, Pr, u_in, envTemp, solidTemp, heatFlux, device
cold plate -> solidTemp=0, heatFlux=q, device=1 (heatsinks use the mirror: heatFlux=0, device=0).

Usage: python adapt_coldplate.py --in cp_out_3d cp_out_s0 --out cp_adapt --n 16384 --n_surf 2048
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np

# Pr, rho, Cp, mu  (identical to adapt_bsf so fluid props align across device classes)
FLUID_PROPS = {
    "air":    (0.705,  1.18,   1004.4, 1.831e-5),
    "water":  (6.1,    997.0,  4181.0, 8.9e-4),
    "oil":    (292.0,  850.0,  1900.0, 2.0e-2),
    "glycol": (29.0,   1070.0, 3300.0, 3.5e-3),
}


def duct_surface(L, W, H, n_surf, rng):
    """Sample the 4 long wetted walls of the rectangular duct (exclude inlet/outlet x-faces),
    with INWARD unit normals (toward the coolant). Area-proportional; fixed n_surf points."""
    faces = [  # (fixed axis, value, inward normal)
        (1, 0.0, (0.0, 1.0, 0.0)),    # y=0 bottom (heated wall) -> +y
        (1, H,   (0.0, -1.0, 0.0)),   # y=H top                  -> -y
        (2, 0.0, (0.0, 0.0, 1.0)),    # z=0 side                 -> +z
        (2, W,   (0.0, 0.0, -1.0)),   # z=W side                 -> -z
    ]
    hi = (L, H, W)
    areas = [L * W, L * W, L * H, L * H]
    tot = sum(areas) + 1e-12
    P, Nn = [], []
    for (ax, val, nrm), a in zip(faces, areas):
        k = max(1, int(round(n_surf * a / tot)))
        pts = np.zeros((k, 3))
        pts[:, ax] = val
        others = [i for i in range(3) if i != ax]
        pts[:, others[0]] = rng.uniform(0.0, hi[others[0]], k)
        pts[:, others[1]] = rng.uniform(0.0, hi[others[1]], k)
        P.append(pts); Nn.append(np.tile(np.asarray(nrm), (k, 1)))
    P = np.concatenate(P); Nn = np.concatenate(Nn)
    if len(P) >= n_surf:
        sel = rng.choice(len(P), n_surf, replace=False)
    else:
        sel = rng.choice(len(P), n_surf, replace=True)
    return P[sel].astype(np.float32), Nn[sel].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--n_surf", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    kept = skip = 0
    for d in a.inp:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            p = json.loads(str(z["params"]))
            coords = np.asarray(z["coords"], np.float32)
            M = coords.shape[0]
            if M < a.n:
                skip += 1; continue
            idx = rng.choice(M, a.n, replace=False)
            T = np.asarray(z["T"], np.float32).reshape(-1)[idx][:, None]
            U = np.asarray(z["U"], np.float32).reshape(M, -1)[idx]
            if U.shape[1] < 3:                                  # pad to 3 comp if needed
                U = np.pad(U, ((0, 0), (0, 3 - U.shape[1])))
            prgh = np.asarray(z["p_rgh"], np.float32).reshape(-1)[idx][:, None]
            fluid = p.get("fluid", "water")
            pr, rho, cp, mu = FLUID_PROPS.get(fluid, FLUID_PROPS["water"])
            L, W, H = float(p["L"]), float(p["W"]), float(p["H"])
            cond = dict(rho=rho, mu=mu, Cp=cp, Pr=pr,
                        u_in=float(p.get("u_in", 0.0)),
                        envTemp=float(p.get("T_in", 300.0)),   # inlet coolant temp (== ambient role)
                        solidTemp=0.0,                          # cold plate: no fixed wall temp
                        heatFlux=float(p.get("q", 0.0)),        # chip heat flux (W/m^2)
                        device=1.0)                             # 1 = cold plate
            surf_pts, surf_nrm = duct_surface(L, W, H, a.n_surf, rng)
            sid = f"cp__{Path(f).parent.name}__{Path(f).stem}"
            np.savez_compressed(out / f"{sid}.npz",
                                coords=coords[idx], U=U, T=T, p_rgh=prgh,
                                surf_pts=surf_pts, surf_normals=surf_nrm,
                                conditions=json.dumps(cond))
            kept += 1
    print(f"[adapt_coldplate] scanned dirs={a.inp}  kept={kept} skipped={skip} -> {out}")


if __name__ == "__main__":
    main()
