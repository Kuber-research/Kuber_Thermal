"""Parametric heatsink -> binary STL (axis-aligned boxes; no external deps).

Shapes (all sit on the domain floor y=0, centered at (cx, *, cz)):
  fins  : base slab + N plate fins (run full base length in x, arrayed in z)
  plate : just the base slab (a flat plate)
  cube  : a single block (base slab with t_b = its height, no fins)

Coordinate convention (matches build_bsf.py domain):
  x = streamwise (inlet -> outlet), y = vertical (gravity -y), z = spanwise.

STL is written as a single solid "heatsink". Fins are embedded EMBED metres into
the base so the union is robust for snappyHexMesh castellation.
"""
from __future__ import annotations
import argparse, json, struct
import numpy as np

EMBED = 5e-4   # fin foot sinks 0.5 mm into base -> clean overlapping union


def _box_tris(x0, x1, y0, y1, z0, z1):
    """12 outward-facing triangles for an axis-aligned box."""
    p = {
        (0, 0, 0): (x0, y0, z0), (1, 0, 0): (x1, y0, z0),
        (1, 1, 0): (x1, y1, z0), (0, 1, 0): (x0, y1, z0),
        (0, 0, 1): (x0, y0, z1), (1, 0, 1): (x1, y0, z1),
        (1, 1, 1): (x1, y1, z1), (0, 1, 1): (x0, y1, z1),
    }
    P = {k: np.array(v, dtype=np.float64) for k, v in p.items()}
    # 6 faces, each split into 2 CCW triangles when viewed from outside
    quads = [
        # -x                                   +x
        [(0,0,0),(0,1,0),(0,1,1),(0,0,1)],  [(1,0,0),(1,0,1),(1,1,1),(1,1,0)],
        # -y                                   +y
        [(0,0,0),(0,0,1),(1,0,1),(1,0,0)],  [(0,1,0),(1,1,0),(1,1,1),(0,1,1)],
        # -z                                   +z
        [(0,0,0),(1,0,0),(1,1,0),(0,1,0)],  [(0,0,1),(0,1,1),(1,1,1),(1,0,1)],
    ]
    tris = []
    for q in quads:
        a, b, c, d = (P[i] for i in q)
        tris.append((a, b, c))
        tris.append((a, c, d))
    return tris


def heatsink_boxes(p):
    L, W, t_b = p["L"], p["W"], p["t_b"]
    cx, cz = p["cx"], p["cz"]
    shape = p.get("shape", "fins")
    x0, x1 = cx - L / 2, cx + L / 2
    z0b, z1b = cz - W / 2, cz + W / 2
    boxes = [(x0, x1, 0.0, t_b, z0b, z1b)]           # base slab
    if shape == "fins":
        N, t_f, h_f, g = p["N"], p["t_f"], p["h_f"], p["g"]
        span = N * t_f + (N - 1) * g
        zs = cz - span / 2
        for k in range(N):
            zf = zs + k * (t_f + g)
            boxes.append((x0, x1, t_b - EMBED, t_b + h_f, zf, zf + t_f))
    elif shape == "pinfin":
        # grid of square pins (t_f x t_f cross-section, height h_f) over the base
        t_f, h_f, g = p["t_f"], p["h_f"], p["g"]
        pitch = t_f + g
        nx, nz = max(2, int(L / pitch)), max(2, int(W / pitch))
        while nx * nz > 100:                          # cap pin count for meshing tractability
            if nx >= nz: nx -= 1
            else: nz -= 1
        gx, gz = nx * t_f + (nx - 1) * g, nz * t_f + (nz - 1) * g
        xs, zs = cx - gx / 2, cz - gz / 2
        for ix in range(nx):
            for iz in range(nz):
                xp, zp = xs + ix * pitch, zs + iz * pitch
                boxes.append((xp, xp + t_f, t_b - EMBED, t_b + h_f, zp, zp + t_f))
    # plate / cube: base slab only (cube -> caller sets t_b == height, L==W small)
    return boxes


def write_stl(path, boxes, name="heatsink"):
    tris = []
    for b in boxes:
        tris.extend(_box_tris(*b))
    with open(path, "wb") as f:
        f.write(struct.pack("<80sI", name.encode()[:80].ljust(80, b"\0"), len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            nl = np.linalg.norm(n)
            n = n / nl if nl > 0 else np.zeros(3)
            f.write(struct.pack("<12f", *n, *a, *b, *c, ))
            f.write(struct.pack("<H", 0))
    return len(tris)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="json file with geometry params")
    ap.add_argument("--out", required=True, help="output .stl path")
    a = ap.parse_args()
    p = json.loads(open(a.params).read())
    boxes = heatsink_boxes(p)
    ntri = write_stl(a.out, boxes)
    # report bounding box + a guaranteed-fluid point (near inlet, high up)
    print(f"[stl] shape={p.get('shape','fins')} boxes={len(boxes)} tris={ntri} -> {a.out}")


if __name__ == "__main__":
    main()
