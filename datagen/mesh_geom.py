"""Exact triangulated surface MESH of the heatsink (union of axis-aligned boxes).

This is the "full mesh" geometry representation, richer than a scalar SDF or a
jittered surface cloud: it gives the actual boundary as vertices + triangles +
outward per-face normals + areas, with interior faces culled. Area-weighted
sampling (SDF-filtered) then draws a surface point cloud of any density from the
true mesh — the geometry the encoder attends to.

Depends on make_stl.heatsink_boxes for the box list (fins / plate / cube / pinfin).
"""
from __future__ import annotations
import numpy as np
from make_stl import heatsink_boxes


# ---- box -> triangle mesh -------------------------------------------------
_BOX_VERTS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
# 12 triangles (2 per face) as vertex indices
_BOX_TRIS = np.array([
    [0, 3, 2], [0, 2, 1],       # z=0
    [4, 5, 6], [4, 6, 7],       # z=1
    [0, 1, 5], [0, 5, 4],       # y=0
    [3, 7, 6], [3, 6, 2],       # y=1
    [0, 4, 7], [0, 7, 3],       # x=0
    [1, 2, 6], [1, 6, 5],       # x=1
], int)


def _box_mesh(x0, x1, y0, y1, z0, z1):
    """Vertices[8,3] and 12 triangles with OUTWARD-oriented normals for one box."""
    lo = np.array([x0, y0, z0]); hi = np.array([x1, y1, z1])
    v = lo + _BOX_VERTS * (hi - lo)
    tris = _BOX_TRIS.copy()
    center = (lo + hi) / 2.0
    # face normal via cross product; flip to point away from box center
    n = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
    cent = v[tris].mean(1)
    flip = np.sum(n * (cent - center), axis=1) < 0
    n[flip] *= -1
    tris[flip] = tris[flip][:, ::-1]            # keep winding consistent with normal
    return v, tris, n


def _box_sdf(p, lo, hi):
    c = (lo + hi) / 2.0; h = (hi - lo) / 2.0
    q = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)


def union_sdf(coords, params):
    """Signed distance to the union of heatsink boxes (any shape), our frame."""
    boxes = heatsink_boxes(params)
    sdf = np.full(coords.shape[0], 1e9)
    for (x0, x1, y0, y1, z0, z1) in boxes:
        sdf = np.minimum(sdf, _box_sdf(coords, np.array([x0, y0, z0]), np.array([x1, y1, z1])))
    return sdf


def heatsink_mesh(params, cull_eps=1e-5):
    """Return (verts[V,3], faces[F,3], normals[F,3], areas[F]) of the union surface.

    Faces whose centroid lies strictly INSIDE another box (union SDF < -cull_eps) are
    culled, so only the outer surface remains (e.g. the fin foot embedded in the base
    is dropped). Culling is centroid-based (approximate at partial overlaps); sampling
    additionally SDF-filters, so residual interior area does not leak into the cloud.
    """
    boxes = heatsink_boxes(params)
    V, Fidx, Nf, off = [], [], [], 0
    for b in boxes:
        v, t, n = _box_mesh(*b)
        V.append(v); Fidx.append(t + off); Nf.append(n); off += len(v)
    V = np.concatenate(V); F = np.concatenate(Fidx); Nf = np.concatenate(Nf)
    cent = V[F].mean(1)
    keep = union_sdf(cent, params) > -cull_eps
    F, Nf = F[keep], Nf[keep]
    tri = V[F]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return V.astype(np.float32), F.astype(np.int64), Nf.astype(np.float32), areas.astype(np.float32)


def sample_mesh(params, n, seed=0, sdf_tol=1e-4):
    """Area-weighted surface sample of n points + outward normals from the full mesh.
    Points that fall interior to the union (residual overlap area) are SDF-filtered."""
    V, F, Nf, A = heatsink_mesh(params)
    rng = np.random.default_rng(seed)
    pts, nrm = [], []
    need = n
    for _ in range(6):                                   # resample until n valid points
        k = max(need * 2, 1024)
        fi = rng.choice(len(F), k, p=A / A.sum())
        r1 = np.sqrt(rng.random(k)); r2 = rng.random(k)
        tri = V[F[fi]]
        p = ((1 - r1)[:, None] * tri[:, 0]
             + (r1 * (1 - r2))[:, None] * tri[:, 1]
             + (r1 * r2)[:, None] * tri[:, 2])
        valid = union_sdf(p.astype(np.float64), params) > -sdf_tol
        pts.append(p[valid]); nrm.append(Nf[fi][valid])
        if sum(len(x) for x in pts) >= n:
            break
        need = n - sum(len(x) for x in pts)
    P = np.concatenate(pts); Nn = np.concatenate(nrm)
    idx = rng.choice(len(P), n, replace=len(P) < n)
    return P[idx].astype(np.float32), Nn[idx].astype(np.float32)
