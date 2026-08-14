#!/usr/bin/env python3
"""Render publication-style CFD field images from Kuber sample cases (data_sample/).

Produces temperature + velocity-magnitude point-cloud renders for a heatsink and a cold
plate - the actual OpenFOAM ground-truth fields the surrogate is trained to predict.
Run with a matplotlib+numpy env: python assets/render_sim.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DS = os.path.join(ROOT, "data_sample")
OUT = os.path.join(HERE, "sim")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

_PR2FLUID = {0.705: "air", 6.1: "water", 292.0: "oil", 29.0: "glycol"}


def load(prefix):
    f = sorted(p for p in os.listdir(DS) if p.startswith(prefix) and p.endswith(".npz"))[0]
    d = np.load(os.path.join(DS, f), allow_pickle=True)
    cond = json.loads(str(d["conditions"]))
    return f, np.asarray(d["coords"], float), np.asarray(d["T"], float).ravel(), np.asarray(d["U"], float), cond


def caption(device, cond):
    fluid = _PR2FLUID.get(round(float(cond.get("Pr", 0)), 3), "coolant")
    if device == "heatsink":
        bc = f"wall {float(cond.get('solidTemp', 0)):.0f} K"
        reg = "natural convection"
    else:
        q = float(cond.get("heatFlux", 0))
        bc = f"heat flux {q/1000:.0f} kW/m²"
        reg = f"forced, u={float(cond.get('u_in',0)):.2f} m/s"
    return f"{fluid}, {bc}, {reg}, ambient {float(cond.get('envTemp',0)):.0f} K"


def panel(ax, coords, vals, cmap, label, title, elev, azim):
    lo, hi = np.percentile(vals, [2, 98])
    norm = np.clip((vals - lo) / (hi - lo + 1e-12), 0, 1)
    colors = plt.get_cmap(cmap)(norm)
    colors[:, 3] = 0.10 + 0.80 * norm          # value-weighted opacity: hot/fast opaque, cold/slow faint
    order = np.argsort(norm)                    # draw hot/fast points last (on top)
    ax.scatter(coords[order, 0], coords[order, 1], coords[order, 2],
               c=colors[order], s=4, linewidths=0, depthshade=False)
    ax.set_title(title, fontsize=12, pad=-2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
        axis.pane.set_alpha(0.0)
    ax.set_xlabel("x [m]", labelpad=-8, fontsize=8)
    ax.set_ylabel("y [m]", labelpad=-8, fontsize=8)
    ax.set_zlabel("z [m]", labelpad=-8, fontsize=8)
    ax.tick_params(labelsize=6.5, pad=-3)
    ax.grid(True, alpha=0.15)
    ext = coords.max(0) - coords.min(0)
    ax.set_box_aspect(tuple(ext / ext.max()))
    ax.view_init(elev=elev, azim=azim)
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(lo, hi)); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.11)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=7)


def device_figure(prefix, device, elev, azim, fname):
    f, coords, T, U, cond = load(prefix)
    Umag = np.linalg.norm(U, axis=1)
    fig = plt.figure(figsize=(11, 4.4))
    ax1 = fig.add_subplot(121, projection="3d")
    panel(ax1, coords, T, "inferno", "T [K]", "Temperature", elev, azim)
    ax2 = fig.add_subplot(122, projection="3d")
    panel(ax2, coords, Umag, "viridis", "|U| [m/s]", "Velocity magnitude", elev, azim)
    fig.suptitle(f"{device.title()} - OpenFOAM CHT ground truth", fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.02, caption(device, cond), ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(OUT, fname), dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", fname, "from", f, "| T range", f"{T.min():.0f}-{T.max():.0f}K")
    return coords, T, cond


def hero(hs, cp):
    (hc, hT, _), (cc, cT, _) = hs, cp
    fig = plt.figure(figsize=(11, 4.4))
    a1 = fig.add_subplot(121, projection="3d")
    panel(a1, hc, hT, "inferno", "T [K]", "Heatsink (air, natural convection)", 22, -58)
    a2 = fig.add_subplot(122, projection="3d")
    panel(a2, cc, cT, "inferno", "T [K]", "Cold plate (liquid, forced)", 20, -68)
    fig.suptitle("One model, coupled fluid-heat fields across geometries", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "hero.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote hero.png")


hs = device_figure("bsf_out_ov_air_natural", "heatsink", 22, -58, "sim_heatsink.png")
cp = device_figure("cp__cp_out_3d", "cold plate", 20, -68, "sim_coldplate.png")
hero((hs[0], hs[1], hs[2]), (cp[0], cp[1], cp[2]))
print("done")
