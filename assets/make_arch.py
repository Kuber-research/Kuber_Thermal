#!/usr/bin/env python3
"""Generate the SurfaceGeoTransolver architecture flowchart as a self-contained SVG.

Inputs (surface geometry, query points, physics conditioning) feed a neural spine
(surface encoder -> local kNN cross-attention -> concat embedding -> GeoTransolver core)
that outputs the 5-channel field at every query node. Pure stdlib; render to PNG with
inkscape for the paper.
"""
import html, os

OUT = os.path.dirname(os.path.abspath(__file__))
INK, BLUE, EMBER = "#14181F", "#1F4E79", "#C2410C"
GFILL, GTXT, LINE, MUTED, BG = "#EEF2F6", "#334155", "#94A3B8", "#5B6672", "#FFFFFF"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
W, H = 1160, 470


def esc(s):
    return html.escape(str(s))


def box(x, y, w, h, lines, fill, txt, sub=None, dash=False, rx=11, fs=15, weight=700):
    d = ' stroke-dasharray="6 5"' if dash else ""
    stroke = BLUE if fill == "#FFFFFF" else "none"
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
         f'stroke="{stroke if not dash else MUTED}" stroke-width="1.4"{d}/>']
    ls = lines if isinstance(lines, list) else [lines]
    n = len(ls)
    cy = y + h / 2 - (n - 1) * (fs + 3) / 2 + fs / 2 - 2
    for i, ln in enumerate(ls):
        p.append(f'<text x="{x + w/2}" y="{cy + i*(fs+3):.1f}" font-size="{fs}" font-weight="{weight}" '
                 f'fill="{txt}" text-anchor="middle">{esc(ln)}</text>')
    if sub:
        p.append(f'<text x="{x + w/2}" y="{y + h - 9}" font-size="11.5" fill="{txt}" '
                 f'opacity="0.72" text-anchor="middle">{esc(sub)}</text>')
    return "".join(p)


def arrow(x1, y1, x2, y2, label=None, dash=False, color=None):
    color = color or "#475569"
    d = ' stroke-dasharray="5 4"' if dash else ""
    p = [f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" stroke-width="2"{d} marker-end="url(#ah)"/>']
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        w = len(label) * 6.6 + 12
        p.append(f'<rect x="{mx - w/2:.1f}" y="{my - 10}" width="{w:.1f}" height="19" rx="4" fill="{BG}"/>')
        p.append(f'<text x="{mx:.1f}" y="{my + 3.5:.1f}" font-size="11.5" fill="{MUTED}" text-anchor="middle">{esc(label)}</text>')
    return "".join(p)


p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
     f'font-family="{FONT}" role="img" aria-label="SurfaceGeoTransolver architecture">',
     f'<rect width="{W}" height="{H}" fill="{BG}"/>',
     '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7.5" markerHeight="7.5" '
     f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#475569"/></marker></defs>']

# title
p.append(f'<text x="30" y="42" font-size="21" font-weight="800" fill="{INK}">SurfaceGeoTransolver</text>')
p.append(f'<text x="30" y="64" font-size="13.5" fill="{MUTED}">Geometry-general conjugate-heat-transfer surrogate — predicts (U, T, p) at any query node</text>')

# spine (blue) row
sy, sh, sw = 108, 74, 188
xs = [30, 30 + (sw + 34), 30 + 2 * (sw + 34), 30 + 3 * (sw + 34)]   # 4 blue boxes
spine = [
    (xs[0], ["Surface", "encoder"], "self-attention"),
    (xs[1], ["Local kNN", "cross-attention"], "k = 16"),
    (xs[2], ["Concatenate", "embedding"], "per node"),
    (xs[3], ["GeoTransolver", "core"], "256 x 12, physics attn"),
]
for x, lines, sub in spine:
    p.append(box(x, sy, sw, sh, lines, BLUE, "#FFFFFF", sub=sub, fs=15.5))

# output (ember)
ox = xs[3] + sw + 46
ow = W - ox - 30
p.append(box(ox, sy, ow, sh, ["Field per node"], EMBER, "#FFFFFF", sub="U_x U_y U_z, T, p_rgh", fs=15.5))

# spine arrows + labels
midy = sy + sh / 2
lbls = ["geometry tokens", "descriptor", "embedding"]
for i in range(3):
    p.append(arrow(xs[i] + sw, midy, xs[i + 1], midy, lbls[i]))
p.append(arrow(xs[3] + sw, midy, ox, midy, "5-channel field"))

# inputs (gray) row, feeding up into the spine
iy, ih, iw = 300, 62, 210
ins = [
    (14, xs[0] + sw / 2, ["Surface point cloud", "+ outward normals"], xs[0] + sw / 2),   # -> encoder
    (300, xs[1] + sw / 2, ["Query points", "(fluid domain)"], None),                       # -> kNN and core
    (556, xs[2] + sw / 2, ["Physics conditioning", "rho mu Cp Pr u_in BC device"], xs[2] + sw / 2),  # -> concat
]
# input boxes
p.append(box(14, iy, iw, ih, ["Surface point cloud", "+ outward normals"], GFILL, GTXT, fs=13.5, weight=600))
p.append(box(300, iy, iw, ih, ["Query points", "(fluid domain)"], GFILL, GTXT, fs=13.5, weight=600))
p.append(box(556, iy, iw + 20, ih, ["Physics conditioning", "rho . mu . Cp . Pr . u_in . BC . device"], GFILL, GTXT, fs=13, weight=600))
# up-arrows
p.append(arrow(14 + iw / 2, iy, xs[0] + sw / 2, sy + sh))                    # surface -> encoder
p.append(arrow(300 + iw / 2, iy, xs[1] + sw / 2, sy + sh))                   # query -> kNN
p.append(arrow(300 + iw - 10, iy + 6, xs[3] + 24, sy + sh, color="#64748B")) # query -> core (branch)
p.append(arrow(556 + (iw + 20) / 2, iy, xs[2] + sw / 2, sy + sh))           # conditioning -> concat

# optional PDE-Refiner head (dashed) above the core->output
ry = 8
p.append(box(xs[3] + 40, ry + 8, 300, 40, ["PDE-Refiner head (optional)"], BG, MUTED, sub=None, dash=True, fs=12.5, weight=600))
p.append(arrow(xs[3] + sw + 90, ry + 48, xs[3] + sw + 90, sy, dash=True, color=MUTED))

p.append("</svg>")
open(os.path.join(OUT, "fig_architecture.svg"), "w").write("".join(p))
print("wrote fig_architecture.svg")
