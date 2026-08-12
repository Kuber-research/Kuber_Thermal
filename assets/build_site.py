#!/usr/bin/env python3
"""Build the Kuber project page: a self-contained technical/paper-style website.

Embeds the simulation PNGs as base64 and inlines the SVG result figures, so the
output is a single portable HTML file (works offline, on GitHub Pages, or as an
Artifact). Writes site/index.html (full document) and prints an artifact-body copy.
"""
import base64, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = HERE
SIM = os.path.join(HERE, "sim")


def png_uri(name):
    with open(os.path.join(SIM, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def svg_inline(name):
    s = open(os.path.join(ASSETS, name)).read()
    s = re.sub(r'\swidth="[\d.]+"', '', s, count=1)     # drop root svg fixed width
    s = re.sub(r'\sheight="[\d.]+"', '', s, count=1)     # drop root svg fixed height
    s = s.replace("<svg ", '<svg style="width:100%;height:auto;display:block" ', 1)
    return s


STYLE = r"""
*,*::before,*::after{box-sizing:border-box}
:root{
  --ground:#FCFCFD; --surface:#FFFFFF; --ink:#12161C; --muted:#5B6672;
  --accent:#1F4E79; --ember:#C2410C; --line:#E6EAEF; --code:#F5F7FA; --btn:#FFFFFF;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0E1116; --surface:#161B22; --ink:#E7ECF2; --muted:#9AA7B4;
  --accent:#7CB2E8; --ember:#F0863C; --line:#242C36; --code:#12161C; --btn:#1A212B;
}}
:root[data-theme="dark"]{
  --ground:#0E1116; --surface:#161B22; --ink:#E7ECF2; --muted:#9AA7B4;
  --accent:#7CB2E8; --ember:#F0863C; --line:#242C36; --code:#12161C; --btn:#1A212B;
}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.65; font-size:17px; letter-spacing:-0.003em;
}
.serif{font-family:Georgia,"Iowan Old Style","Times New Roman",serif}
.wrap{max-width:920px; margin:0 auto; padding:0 22px 90px}
a{color:var(--accent); text-underline-offset:2px}

/* hero */
.hero{text-align:center; padding:72px 0 40px}
.eyebrow{font-size:.78rem; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); font-weight:600}
.hero h1{font-family:Georgia,"Iowan Old Style","Times New Roman",serif; font-weight:600;
  font-size:clamp(3rem,9vw,5.2rem); line-height:1; margin:.18em 0 .06em; letter-spacing:-.02em; text-wrap:balance}
.rule{width:132px; height:3px; margin:18px auto 20px; border-radius:2px;
  background:linear-gradient(90deg,var(--ember),var(--accent))}
.sub{font-size:clamp(1.1rem,2.6vw,1.42rem); color:var(--ink); max-width:32ch; margin:0 auto; text-wrap:balance}
.authors{color:var(--muted); margin:.8em 0 1.4em; font-size:1rem}
.links{display:flex; flex-wrap:wrap; gap:10px; justify-content:center}
.btn{display:inline-flex; align-items:center; gap:.4em; padding:.52em 1.05em; border:1px solid var(--line);
  border-radius:999px; background:var(--btn); color:var(--ink); text-decoration:none; font-size:.94rem; font-weight:500;
  transition:border-color .15s, color .15s, transform .15s}
.btn:hover{border-color:var(--accent); color:var(--accent); transform:translateY(-1px)}
.btn.primary{background:var(--accent); border-color:var(--accent); color:#fff}
.btn.primary:hover{color:#fff; filter:brightness(1.08)}

/* sections */
section{margin-top:56px}
h2{font-family:Georgia,"Iowan Old Style","Times New Roman",serif; font-weight:600; font-size:1.72rem;
  letter-spacing:-.01em; margin:0 0 .5em; padding-bottom:.34em; border-bottom:1px solid var(--line)}
p{margin:0 0 1.05em; max-width:70ch}
section > p{margin-left:auto; margin-right:auto}
strong{font-weight:650}

/* figures — always light "paper" cards, readable on either theme */
figure.fig{margin:26px 0; background:#FFFFFF; border:1px solid #E6EAEF; border-radius:14px;
  padding:16px 16px 6px; box-shadow:0 1px 3px rgba(16,24,40,.05)}
figure.fig img{width:100%; height:auto; display:block; border-radius:6px}
figure.fig svg{width:100%; height:auto; display:block}
figcaption{font-size:.9rem; color:#5B6672; padding:12px 4px 8px; line-height:1.5; border-top:1px solid #EEF1F5; margin-top:10px}
.grid2{display:grid; grid-template-columns:1fr 1fr; gap:20px}
.grid2 figure.fig{margin:26px 0 0}
@media (max-width:720px){.grid2{grid-template-columns:1fr}}

/* code */
pre{background:var(--code); border:1px solid var(--line); border-radius:12px; padding:16px 18px; overflow-x:auto}
code{font-family:ui-monospace,"SF Mono","Cascadia Code",Menlo,monospace; font-size:.88rem}
pre code{color:var(--ink)}

footer{margin-top:64px; padding-top:24px; border-top:1px solid var(--line); color:var(--muted); font-size:.9rem}
footer p{max-width:none}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

CONTENT = """
<div class="wrap">
  <header class="hero">
    <div class="eyebrow">Open framework · Conjugate heat transfer</div>
    <h1>Kuber</h1>
    <div class="rule"></div>
    <p class="sub">Geometry-general neural surrogates for coupled fluid&ndash;heat simulation.</p>
    <p class="authors">Shubh Jain &nbsp;&middot;&nbsp; Kuber.ai</p>
    <nav class="links">
      <a class="btn primary" href="demo.html">Interactive demo &#8599;</a>
      <a class="btn" href="https://github.com/ShubhJain007/Kuber/blob/main/paper/kuber.pdf">Paper (PDF)</a>
      <a class="btn" href="https://github.com/ShubhJain007/Kuber">Code</a>
      <a class="btn" href="https://colab.research.google.com/github/ShubhJain007/Kuber/blob/main/notebooks/quickstart.ipynb">Colab</a>
      <a class="btn" href="https://github.com/ShubhJain007/Kuber/tree/main/results">Results (JSON)</a>
      <a class="btn" href="#bibtex">BibTeX</a>
    </nav>
  </header>

  <figure class="fig">__HERO__
    <figcaption>Coupled fluid&ndash;heat fields from a single model &mdash; a heatsink in natural-convection air and a
    liquid-cooled cold plate. OpenFOAM ground truth, colored by temperature.</figcaption>
  </figure>

  <section>
    <h2>Abstract</h2>
    <p>Conjugate heat transfer (CHT) &mdash; heat conducting through a solid while a moving fluid carries it away &mdash;
    is the shared physics behind heatsinks, cold plates, heat exchangers, power electronics, and battery packs.
    <strong>Kuber</strong> is an open framework for building neural surrogates of CHT: a data engine that generates
    OpenFOAM physics, a geometry-general model (<strong>SurfaceGeoTransolver</strong> &mdash; a GeoTransolver
    physics-attention core plus a surface-geometry encoder), and an honest evaluation harness. On the public
    <strong>SIMSHIFT</strong> heatsink benchmark it reaches <strong>12.14&nbsp;K temperature RMSE</strong>, beating the
    previous published best (UPT, 12.41&nbsp;K) <strong>with no domain adaptation</strong>, and a single model spans two
    device classes &mdash; heatsinks and cold plates. Inference is sub-second, roughly 1,000&times; faster than the CFD
    it learns from.</p>
  </section>

  <section>
    <h2>Method</h2>
    <p><strong>SurfaceGeoTransolver</strong> predicts the full steady field <em>(U, T, p)</em> at an arbitrary query
    point cloud from a surface point cloud + normals and physics conditioning (fluid properties, boundary conditions,
    inlet velocity, a device flag). Geometry enters through a surface encoder that produces a per-node descriptor via
    kNN cross-attention, so the model works on arbitrary CAD with no analytic signed-distance field. About 14&nbsp;M
    parameters, built on NVIDIA PhysicsNeMo.</p>
    <p>The framework is three shipping pillars &mdash; a <strong>data engine</strong> (parametric geometry &rarr;
    OpenFOAM <code>buoyantSimpleFoam</code> &rarr; per-node <code>.npz</code>, convergence-gated), the <strong>model</strong>,
    and an <strong>evaluation harness</strong> (per-field nRMSE, near-wall fidelity, a no-explosion stability proof) &mdash;
    with CAD connectors, Bayesian uncertainty, and agentic geometry optimization on the roadmap.</p>
  </section>

  <section>
    <h2>Results</h2>
    <p>All numbers are measured and reproducible with the released code; full tables and caveats are in the repository.
    On the SIMSHIFT heatsink split (train fin counts 5&ndash;8 &rarr; test 10&ndash;12), Kuber leads on temperature
    &mdash; the design-critical field &mdash; with no unsupervised domain adaptation (UDA), the crutch every published
    baseline relies on.</p>
    <figure class="fig">__LEADERBOARD__
      <figcaption>SIMSHIFT heatsink leaderboard (medium / out-of-distribution). Kuber (no UDA) versus published
      baselines whose numbers include UDA. Lower is better.</figcaption>
    </figure>
    <div class="grid2">
      <figure class="fig">__INDIST__
        <figcaption>The leading number is zero-shot: the test fin counts never appear in training.</figcaption>
      </figure>
      <figure class="fig">__STABILITY__
        <figcaption>Predicted &nabla;T at fin tips/corners stays at or below physical everywhere &mdash; explosion
        fraction 0, zero NaN/Inf.</figcaption>
      </figure>
    </div>
    <figure class="fig">__MULTIGEO__
      <figcaption>One model, two device classes. Held-out cold plates 3.11&nbsp;K and heatsinks 5.13&nbsp;K on the
      Kuber corpus (there is no public cold-plate benchmark; not comparable to the SIMSHIFT numbers).</figcaption>
    </figure>
    <div class="grid2">
      <figure class="fig">__VALUE__
        <figcaption>Pretraining on the self-generated corpus lowers error &mdash; the data engine is a moat.</figcaption>
      </figure>
      <figure class="fig">__SPEED__
        <figcaption>Sub-second inference versus a median 22-minute CFD solve (log scale).</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <h2>Dataset</h2>
    <p>A self-generated OpenFOAM CHT corpus &mdash; zero cases from SIMSHIFT or any licensed source. Below are the
    ground-truth fields the surrogate is trained to predict, for a heatsink and a cold plate.</p>
    <figure class="fig">__SIMHS__
      <figcaption>Heatsink &mdash; temperature and velocity magnitude (air, natural convection). OpenFOAM ground truth.</figcaption>
    </figure>
    <figure class="fig">__SIMCP__
      <figcaption>Cold plate &mdash; temperature and velocity magnitude (liquid, forced). The coolant enters cold and
      warms downstream while the duct flow drives the transport.</figcaption>
    </figure>
    <div class="grid2">
      <figure class="fig">__CORPUS__
        <figcaption>Corpus coverage &mdash; fluids, regimes, shapes, device classes.</figcaption>
      </figure>
      <figure class="fig">__MESH__
        <figcaption>Fidelity is verified: prism layers recover the hot spot within 0.1&nbsp;K of a fine mesh.</figcaption>
      </figure>
    </div>
  </section>

  <section id="bibtex">
    <h2>BibTeX</h2>
<pre><code>@software{kuber2026,
  title  = {Kuber: An Open Framework for Conjugate-Heat-Transfer AI},
  author = {Jain, Shubh},
  year   = {2026},
  url    = {https://github.com/ShubhJain007/Kuber}
}</code></pre>
  </section>

  <footer>
    <p>Released under the PolyForm Noncommercial License 1.0.0 &mdash; free for research and other noncommercial use;
    commercial use requires a separate license. Built on NVIDIA PhysicsNeMo (Apache-2.0).</p>
    <p>Built by the Kuber.ai team.</p>
  </footer>
</div>
"""

CONTENT = (CONTENT
           .replace("__HERO__", f'<img src="{png_uri("hero.png")}" alt="Heatsink and cold-plate temperature fields">')
           .replace("__SIMHS__", f'<img src="{png_uri("sim_heatsink.png")}" alt="Heatsink temperature and velocity fields">')
           .replace("__SIMCP__", f'<img src="{png_uri("sim_coldplate.png")}" alt="Cold plate temperature and velocity fields">')
           .replace("__LEADERBOARD__", svg_inline("fig_leaderboard.svg"))
           .replace("__INDIST__", svg_inline("fig_indist_vs_ood.svg"))
           .replace("__STABILITY__", svg_inline("fig_stability.svg"))
           .replace("__MULTIGEO__", svg_inline("fig_multigeo.svg"))
           .replace("__VALUE__", svg_inline("fig_value_of_data.svg"))
           .replace("__SPEED__", svg_inline("fig_speed.svg"))
           .replace("__CORPUS__", svg_inline("fig_corpus.svg"))
           .replace("__MESH__", svg_inline("fig_mesh_convergence.svg")))

TITLE = "Kuber"
full = (f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{TITLE}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n{CONTENT}\n</body>\n</html>\n")
body_only = f"<title>{TITLE}</title>\n<style>{STYLE}</style>\n{CONTENT}\n"

os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
open(os.path.join(ROOT, "site", "index.html"), "w").write(full)
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(body_only)      # artifact-body copy to given path
print("wrote site/index.html", f"({len(full)//1024} KB)")
