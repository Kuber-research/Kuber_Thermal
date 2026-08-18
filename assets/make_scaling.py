#!/usr/bin/env python3
"""Scaling-law figure (dual panel) as self-contained SVG, matching the Kuber figure style.

Form = trend line (magnitude vs a continuous axis); one series per panel (the model), so no
legend box — the panel title names it; direct value labels on every point; recessive grid; white
surface so it reads on GitHub light AND dark. Shared y-scale across panels so the size-scaling
blow-up is honestly comparable to the data-scaling gains. Numbers from the CHT scaling sweep
(Kuber surface+ABL, heatsink corpus, held-out cross-fluid OOD).
"""
import html, math, os

OUT = os.path.dirname(os.path.abspath(__file__))
EMBER = "#1F4E79"                                    # the model (single series)
INK, MUTED, GRID, BG = "#14181F", "#5B6672", "#E6EAEF", "#FFFFFF"
GREEN, RED = "#2E7D5B", "#B4442E"                    # trend direction (with text label, not color-alone)
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"

DATA = [(273, 3.27), (546, 2.79), (1092, 2.12)]      # (train cases, OOD T-RMSE K)
SIZE = [(14, 2.12), (40, 2.45), (90, 6.55)]          # (params M, OOD T-RMSE K)
YMAX = 7.0

esc = lambda s: html.escape(str(s))
num = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")


def panel(p, ox, oy, pw, ph, pts, xlabel, title, tcol, ttxt, xticks):
    xs = [math.log10(x) for x, _ in pts]
    lo, hi = min(xs), max(xs)
    rng = (hi - lo) or 1
    X = lambda v: ox + (math.log10(v) - lo) / rng * pw
    Y = lambda v: oy + ph - (v / YMAX) * ph
    for t in range(5):                                # y grid + ticks
        gy = oy + ph - ph * t / 4
        p.append(f'<line x1="{ox}" y1="{gy:.1f}" x2="{ox+pw}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{ox-9}" y="{gy+4:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="end">{num(YMAX*t/4)}</text>')
    for xv in xticks:
        p.append(f'<text x="{X(xv):.1f}" y="{oy+ph+18:.1f}" font-size="11" fill="{MUTED}" text-anchor="middle">{xv}</text>')
    path = " ".join(f"{'M' if i==0 else 'L'}{X(x):.1f} {Y(y):.1f}" for i, (x, y) in enumerate(pts))
    p.append(f'<path d="{path}" fill="none" stroke="{EMBER}" stroke-width="2.5"/>')
    for x, y in pts:
        p.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5.5" fill="{EMBER}" stroke="{BG}" stroke-width="2"/>')
        p.append(f'<text x="{X(x):.1f}" y="{Y(y)-12:.1f}" font-size="12.5" font-weight="700" fill="{INK}" text-anchor="middle">{num(y)}K</text>')
    p.append(f'<text x="{ox}" y="{oy-16}" font-size="15" font-weight="700" fill="{INK}">{esc(title)}</text>')
    p.append(f'<text x="{ox+pw}" y="{oy-16}" font-size="12.5" font-weight="700" fill="{tcol}" text-anchor="end">{esc(ttxt)}</text>')
    p.append(f'<text x="{ox+pw/2:.1f}" y="{oy+ph+40:.1f}" font-size="12" fill="{MUTED}" text-anchor="middle">{esc(xlabel)}</text>')


W, H = 940, 392
p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
     f'font-family="{FONT}" role="img" aria-label="CHT scaling laws"><rect width="{W}" height="{H}" fill="{BG}"/>']
p.append(f'<text x="28" y="34" font-size="20" font-weight="700" fill="{INK}">CHT scaling laws — Kuber (surface + ABL) on the heatsink corpus</text>')
p.append(f'<text x="28" y="57" font-size="13" fill="{MUTED}">Out-of-distribution temperature RMSE (K), lower is better. More data helps; scaling the model at fixed data overfits.</text>')
p.append(f'<text transform="translate(20,213) rotate(-90)" font-size="12" fill="{MUTED}" text-anchor="middle">OOD T-RMSE (K)</text>')
panel(p, 88, 112, 336, 205, DATA, "training cases", "Data scaling", GREEN, "↓ more data → lower error", [273, 546, 1092])
panel(p, 566, 112, 336, 205, SIZE, "model parameters (millions)", "Model-size scaling", RED, "↑ bigger → overfits", [14, 40, 90])
p.append(f'<text x="28" y="{H-14}" font-size="10.5" fill="{MUTED}">Data panel: fixed 14M model · size panel: fixed 1092 cases · held-out cross-fluid OOD (52 cases), single seed · size-L early-stopped (val plateau).</text>')
p.append("</svg>")
open(os.path.join(OUT, "fig_scaling.svg"), "w").write("".join(p))
print("wrote fig_scaling.svg", W, "x", H)
