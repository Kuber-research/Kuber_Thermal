#!/usr/bin/env python3
"""Generate the Kuber result figures as self-contained SVG (no deps).

Design follows the dataviz method: magnitude -> bar length; identity -> fixed hue
order (ember = ours, sky = ours/pretrained, slate = published baseline); orange/blue
is the colorblind-safe pairing; legends always present for >=2 series; lower-is-better
marked; recessive grid; direct value labels. White surface so it reads on GitHub light
AND dark. All numbers are from docs/RESULTS.md.
"""
import html, os

OUT = os.path.dirname(os.path.abspath(__file__))
# Restrained, publication-style palette: deep blue = our model, light gray = baselines,
# muted green = "faithful/safe". No marketing accent.
EMBER, SKY, SLATE = "#1F4E79", "#7FA8C9", "#AEB8C2"   # names kept; hues now professional
INK, MUTED, GRID, BG = "#14181F", "#5B6672", "#E6EAEF", "#FFFFFF"
GREEN = "#2E7D5B"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


def esc(s):
    return html.escape(str(s))


def num(v, dec=2):
    s = f"{v:.{dec}f}".rstrip("0").rstrip(".") if dec else f"{v:.0f}"
    return s


def head(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}" role="img" '
            f'aria-label="{esc(title)}"><rect width="{w}" height="{h}" fill="{BG}"/>')


def titleblock(p, title, subtitle, w, note=None):
    p.append(f'<text x="26" y="34" font-size="20" font-weight="700" fill="{INK}">{esc(title)}</text>')
    p.append(f'<text x="26" y="57" font-size="13.5" fill="{MUTED}">{esc(subtitle)}</text>')
    if note:
        p.append(f'<text x="{w-26}" y="57" font-size="12" fill="{MUTED}" text-anchor="end">{esc(note)}</text>')


def legend(p, items, x, y):
    for name, color in items:
        p.append(f'<circle cx="{x+6}" cy="{y-4}" r="6" fill="{color}"/>')
        p.append(f'<text x="{x+18}" y="{y}" font-size="12.5" fill="{INK}">{esc(name)}</text>')
        x += 24 + len(name) * 7.3
    return x


def write(fname, parts):
    parts.append("</svg>")
    open(os.path.join(OUT, fname), "w").write("".join(parts))
    print("wrote", fname)


# ---------------------------------------------------------------- horizontal bars
def hbars(fname, title, subtitle, rows, maxval, unit, legend_items,
          note="lower is better", ref=None, ref_label=None, valcolor=None, tickdec=0):
    W, LW, RPAD, TOP, BOT = 860, 288, 96, 92, 44
    rowH, barH = 44, 24
    n = len(rows)
    H = TOP + n * rowH + BOT
    x0, PW = LW, W - LW - RPAD
    sc = PW / maxval
    p = [head(W, H, title)]
    titleblock(p, title, subtitle, W, "↓ " + note)
    legend(p, legend_items, 26, 80)
    # gridlines
    ticks = 5
    for t in range(ticks + 1):
        gx = x0 + PW * t / ticks
        p.append(f'<line x1="{gx:.1f}" y1="{TOP}" x2="{gx:.1f}" y2="{TOP + n*rowH}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{gx:.1f}" y="{TOP + n*rowH + 20}" font-size="11" fill="{MUTED}" text-anchor="middle">{num(maxval*t/ticks,tickdec)}</text>')
    if ref is not None:
        rx = x0 + ref * sc
        p.append(f'<line x1="{rx:.1f}" y1="{TOP-6}" x2="{rx:.1f}" y2="{TOP + n*rowH}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        p.append(f'<text x="{rx:.1f}" y="{TOP-10}" font-size="11.5" fill="{MUTED}" text-anchor="middle">{esc(ref_label)}</text>')
    for i, (label, val, color, tag) in enumerate(rows):
        cy = TOP + i * rowH
        by = cy + (rowH - barH) / 2
        p.append(f'<text x="{LW-16}" y="{cy + rowH/2 + 5}" font-size="13.5" fill="{INK}" text-anchor="end">{esc(label)}</text>')
        w = val * sc
        p.append(f'<rect x="{x0}" y="{by:.1f}" width="{max(w,1):.1f}" height="{barH}" rx="5" fill="{color}"/>')
        vc = valcolor or INK
        p.append(f'<text x="{x0 + w + 9:.1f}" y="{by + barH/2 + 5:.1f}" font-size="13.5" font-weight="700" fill="{vc}">{num(val)}{unit}</text>')
        if tag:
            p.append(f'<text x="{x0 + 10:.1f}" y="{by + barH/2 + 5:.1f}" font-size="11.5" fill="#FFFFFF" opacity="0.92">{esc(tag)}</text>')
    write(fname, p)


# ---------------------------------------------------------------- grouped v-bars
def vbars(fname, title, subtitle, groups, series, ymin, ymax, unit,
          note="lower is better", ndec=2):
    W, L, R, TOP, BOT = 820, 66, 26, 96, 60
    plotH = 300
    H = TOP + plotH + BOT
    PW = W - L - R
    ng, ns = len(groups), len(series)
    gw = PW / ng
    bw = min(46, gw / (ns + 1))
    p = [head(W, H, title)]
    titleblock(p, title, subtitle, W, "↓ " + note)
    legend(p, series, 26, 80)

    def yof(v):
        return TOP + plotH - (v - ymin) / (ymax - ymin) * plotH
    # y gridlines
    steps = 5
    for t in range(steps + 1):
        v = ymin + (ymax - ymin) * t / steps
        gy = yof(v)
        p.append(f'<line x1="{L}" y1="{gy:.1f}" x2="{W-R}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{L-10}" y="{gy+4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{num(v,0)}</text>')
    for gi, (gname, vals) in enumerate(groups):
        gx = L + gi * gw
        cx = gx + gw / 2
        total_w = ns * bw + (ns - 1) * 8
        bx = cx - total_w / 2
        for si, v in enumerate(vals):
            color = series[si][1]
            x = bx + si * (bw + 8)
            y = yof(v)
            hh = TOP + plotH - y
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{hh:.1f}" rx="4" fill="{color}"/>')
            p.append(f'<text x="{x + bw/2:.1f}" y="{y-7:.1f}" font-size="12" font-weight="700" fill="{INK}" text-anchor="middle">{num(v,ndec)}</text>')
        lines = str(gname).split("\n")
        for li, ln in enumerate(lines):
            fs = 13 if li == 0 else 11
            col = INK if li == 0 else MUTED
            p.append(f'<text x="{cx:.1f}" y="{TOP + plotH + 24 + li*16:.1f}" font-size="{fs}" fill="{col}" text-anchor="middle">{esc(ln)}</text>')
    write(fname, p)


# ---------------------------------------------------------------- log h-bars (speed)
def hbars_log(fname, title, subtitle, rows, vmin, vmax, note):
    import math
    W, LW, RPAD, TOP, BOT = 884, 278, 150, 88, 48
    rowH, barH = 46, 26
    n = len(rows)
    H = TOP + n * rowH + BOT
    x0, PW = LW, W - LW - RPAD
    lmin, lmax = math.log10(vmin), math.log10(vmax)

    def xof(v):
        return x0 + (math.log10(v) - lmin) / (lmax - lmin) * PW
    p = [head(W, H, title)]
    titleblock(p, title, subtitle, W, note)
    # decade gridlines
    d = int(math.floor(lmin))
    while d <= math.ceil(lmax):
        gx = xof(10 ** d)
        if x0 - 1 <= gx <= x0 + PW + 1:
            p.append(f'<line x1="{gx:.1f}" y1="{TOP}" x2="{gx:.1f}" y2="{TOP + n*rowH}" stroke="{GRID}" stroke-width="1"/>')
            lab = {-1: "0.1 s", 0: "1 s", 1: "10 s", 2: "100 s", 3: "1000 s", 4: "10⁴ s"}.get(d, f"1e{d}")
            p.append(f'<text x="{gx:.1f}" y="{TOP + n*rowH + 20}" font-size="11" fill="{MUTED}" text-anchor="middle">{lab}</text>')
        d += 1
    for i, (label, val, human, color) in enumerate(rows):
        cy = TOP + i * rowH
        by = cy + (rowH - barH) / 2
        p.append(f'<text x="{LW-16}" y="{cy + rowH/2 + 5}" font-size="13.5" fill="{INK}" text-anchor="end">{esc(label)}</text>')
        w = xof(val) - x0
        p.append(f'<rect x="{x0}" y="{by:.1f}" width="{max(w,2):.1f}" height="{barH}" rx="5" fill="{color}"/>')
        p.append(f'<text x="{x0 + max(w,2) + 10:.1f}" y="{by + barH/2 + 5:.1f}" font-size="13" font-weight="700" fill="{INK}">{esc(human)}</text>')
    write(fname, p)


# ---------------------------------------------------------------- stat-tile panel
def tiles(fname, title, subtitle, cells):
    W, TOP = 900, 92
    cols = 4
    rows = (len(cells) + cols - 1) // cols
    cw, ch, gap, mx = 207, 98, 12, 18
    H = TOP + rows * (ch + gap) + 20
    p = [head(W, H, title)]
    titleblock(p, title, subtitle, W)
    for i, (big, small) in enumerate(cells):
        r, c = divmod(i, cols)
        x = mx + c * (cw + gap)
        y = TOP + r * (ch + gap)
        p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="10" fill="#F8FAFC" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{x+18}" y="{y+44}" font-size="25" font-weight="800" fill="{EMBER}">{esc(big)}</text>')
        for li, ln in enumerate(str(small).split("\n")):
            p.append(f'<text x="{x+18}" y="{y+64 + li*16}" font-size="11.5" fill="{MUTED}">{esc(ln)}</text>')
    write(fname, p)


# ============================================================================ data
OURS = "Kuber (ours, no UDA)"
OURSP = "Kuber pretrained on our corpus (no UDA)"
BASE = "Published baseline (with UDA)"

hbars("fig_leaderboard.svg",
      "SIMSHIFT heatsink leaderboard — temperature RMSE",
      "Medium / out-of-distribution split (train fins 5–8 → test 10–12). Baselines include UDA; Kuber uses none.",
      rows=[
          ("Kuber — SurfaceGeoTransolver", 12.14, EMBER, ""),
          ("UPT  (prev. published best)", 12.41, SLATE, ""),
          ("Transolver", 13.43, SLATE, ""),
          ("PointNet", 17.43, SLATE, ""),
      ],
      maxval=18.5, unit=" K",
      legend_items=[("Kuber (ours, no UDA)", EMBER), ("published baseline (with UDA)", SLATE)])

vbars("fig_value_of_data.svg",
      "Value of our data — pretrain on the Kuber corpus, then fine-tune",
      "Same model, only pretraining differs. Temperature RMSE (K), target domain unless noted.",
      groups=[("easy", [8.99, 7.28]), ("medium", [12.94, 12.38]),
              ("hard", [14.42, 14.43]), ("in-dist", [4.63, 4.09])],
      series=[("from scratch", SLATE), ("pretrained on our corpus", EMBER)],
      ymin=0, ymax=16, unit="T-RMSE (K)")

vbars("fig_indist_vs_ood.svg",
      "Generalization — the SOTA result is zero-shot",
      "SurfaceGeoTransolver temperature RMSE (K). Fin counts 10–14 never appear in training.",
      groups=[("in-distribution\n(trained fin range)", [4.29]),
              ("out-of-distribution\n(unseen fin counts)", [12.14])],
      series=[("SurfaceGeoTransolver", EMBER)],
      ymin=0, ymax=14, unit="")

hbars("fig_stability.svg",
      "Stability — edge temperature-gradient fidelity",
      "SurfaceGeoTransolver — predicted ÷ CFD |∇T| near the wall (fin tips / corners).",
      rows=[
          ("In-distribution · edge band", 0.969, GREEN, ""),
          ("In-distribution · steepest peak", 0.938, GREEN, ""),
          ("Out-of-distribution · edge band", 0.746, GREEN, ""),
          ("Out-of-distribution · steepest peak", 0.725, GREEN, ""),
      ],
      maxval=1.2, unit="", legend_items=[("≤ 1.0 = faithful / no explosion", GREEN)],
      note="explosion fraction = 0 · NaN/Inf = 0", ref=1.0, ref_label="1.0  faithful", tickdec=1)

hbars_log("fig_speed.svg",
          "Speed — surrogate vs. the CFD it learns from",
          "Per-case wall-clock (log scale). Surrogate inference is a sub-second estimate; CFD times are measured.",
          rows=[
              ("OpenFOAM CFD — high fin count", 7020, "≈ 117 min", SLATE),
              ("OpenFOAM CFD — median (601 cases)", 1320, "≈ 22 min", SLATE),
              ("OpenFOAM CFD — low fin count", 162, "≈ 2.7 min", SLATE),
              ("Kuber surrogate (inference)", 0.3, "≈ 0.3 s  →  ~1,000–4,000× faster", EMBER),
          ],
          vmin=0.1, vmax=10000, note="≈ 1,000–4,000× faster than CFD")

vbars("fig_mesh_convergence.svg",
      "Data fidelity — mesh convergence of the hot spot",
      "Same geometry, three meshes. Prism layers recover the near-wall peak at ~2.7× fewer cells.",
      groups=[("snap2\n(124k cells)", [359.5]), ("snap2 + 3 layers\n(142k cells)", [378.8]),
              ("snap3 fine\n(382k cells)", [378.9])],
      series=[("hot-spot T-max (K)", EMBER)],
      ymin=350, ymax=385, unit="T-max (K)", note="within 0.1 K of fine mesh", ndec=1)

tiles("fig_corpus.svg",
      "The Kuber corpus at a glance",
      "Self-generated OpenFOAM CHT data — 0 cases from SIMSHIFT or any licensed source.",
      cells=[("4", "fluids\nair · water · oil · glycol"),
             ("2", "regimes\nnatural + forced"),
             ("4+", "shapes\nfins·plate·cube·pin-fin"),
             ("2", "device classes\nheatsink + cold plate"),
             ("1.4 M", "cells per CFD case"),
             ("16,384", "nodes sampled / case"),
             ("5", "field channels (U, T, p)"),
             ("0.1 K", "mesh-convergence tol.")])

print("done")
