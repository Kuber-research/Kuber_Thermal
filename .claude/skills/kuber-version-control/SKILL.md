---
name: kuber-version-control
description: Git and release conventions for this repo — commit message style, what must never be committed (checkpoints, corpora, OpenFOAM run dirs), keeping generated artifacts (figures, site, paper PDF, deploy/hf-space/kuber) in sync with their sources, CI/Pages workflows, and how results and checkpoints are released. Use before committing, branching, opening a PR, or cutting a release.
---

# Version control in Kuber

Remote is `git@github.com:Kuber-research/Kuber_Thermal.git`; work lands on `main`. There are no tags
yet — checkpoint releases are the first thing that will need them.

## Commit messages

The convention here is a **lowercase scope prefix, a colon, and an imperative summary**, then a body
of bullets explaining what changed and why. Scopes in use: `docs`, `site`, `paper`, `demo`, `deploy`,
`README`, and combinations (`docs+site:`). Conventional-commit types appear for code changes:
`feat:`, `fix(demo):`.

```
docs+site: finish KuberNet/ABL rename in architecture text (image was already updated)
- README: architecture caption/alt, model name, benchmark row -> KuberNet (ABL), 11.84 K
- docs/MODEL.md + docs/TRAINING.md: rename to KuberNet, describe the ABL wall-normal penalty
Note anything deliberately left alone, so a reader knows it wasn't an oversight.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Write bodies that state the *reason* and call out what was intentionally not changed — the history
here is used as an audit trail for published numbers, so "left Table IV as the existing
full-conditioning model, and stated honestly that this checkpoint predates ABL" is exactly the kind
of line that belongs in a commit.

## What must never be committed

`.gitignore` encodes it; know *why*, because the tempting exception is usually the one that breaks the
repo:

- **`*.pt` / `*.ckpt` / `outputs/` / `runs/` / `wandb/`** — checkpoints are ~165 MB each and ship as
  **GitHub Release assets**, documented in `checkpoints/README.md`. Never `git add -f` one.
- **`data/`, `corpus/`, `*.vtk`, `*.vtu`** — only the curated 6-case `data_sample/` is tracked.
- **`datagen/cases/`, `datagen/out/`, `*.foam`, `processor*/`** — OpenFOAM run dirs are regenerable.
- **Anything licensed or scraped.** The corpus's clean provenance ("0 cases from SIMSHIFT or any
  licensed source") is a load-bearing claim in the README, the paper, and the license terms; a single
  externally-sourced file committed here invalidates it.

## Generated artifacts must be committed with their source

Several tracked files are build outputs. Changing the source without regenerating leaves the repo
self-contradictory — a real hazard, since the site and paper are what people read.

| generated | built by | trigger |
|---|---|---|
| `assets/fig_*.svg` | `python assets/make_figures.py` | any number in `docs/RESULTS.md` changes (they are hard-coded in the script) |
| `site/index.html` (3.8 MB, self-contained) | `python assets/build_site.py` | README/results/figure changes |
| `site/demo.html` | `python assets/build_viewer.py` (data from `assets/export_viewer.py`) | viewer data or UI changes — **currently deleted from `site/`**: the demo links point at the Hugging Face Space (`shubhoo7-kuber-live.hf.space`) instead. Rebuilding it re-adds a tracked 3D page; decide deliberately. |
| `paper/kuber.pdf` | LaTeX build of `paper/kuber.tex` | any paper edit — re-render, don't ship a stale PDF |
| `deploy/hf-space/kuber/*` | manual copy of the top-level `kuber/` package | see below |

**`deploy/hf-space/kuber/` is a duplicate of the `kuber/` package and is currently stale** — it is
missing the ABL cross-attention (`--no_abl` / `abl=` arg and the wall-normal penalty) and the
Buckingham-Pi conditioning (`--pi_features`) that landed in `kuber/`. Whenever you touch
`kuber/*.py`, decide explicitly whether the deployed Space needs the same change, and say which way
you went in the commit body. Diff before assuming:

```bash
for f in kuber/*.py; do diff -q "$f" "deploy/hf-space/$f" >/dev/null || echo "DIFF $f"; done
```

## Atomic commits for published numbers

A number lives in up to five places at once (`results/*.json`, `results/leaderboard.csv`,
`docs/RESULTS.md`, `README.md`, `paper/kuber.tex` + regenerated figures). Change them in **one**
commit. A commit that updates the README table but not `results/leaderboard.csv` leaves the
machine-readable file lying, and the machine-readable files are what reviewers check.

The same applies to renames: the KuberNet/ABL rename took several commits precisely because it was
split across text, images, and figure labels — prefer one sweep per surface.

## Branching and PRs

- Land small, focused changes; open an issue first for anything large (a new device class, a hosted
  leaderboard) — `CONTRIBUTING.md`.
- Don't commit directly to `main` on the user's behalf without being asked; branch, then propose.
- External contributions come as forks + PRs, and a leaderboard row PR must carry the four
  disclosures (UDA, external pretraining, measured params, reproduction command + environment).
  Review those for honesty, not for whether the result beats Kuber.
- CI (`.github/workflows/tests.yml`) runs `pytest -q` on every push and PR with CPU torch and **no
  PhysicsNeMo** — tests that need the GeoTransolver core must self-skip, or you break the badge.
- `.github/workflows/pages.yml` deploys `site/` to GitHub Pages on pushes to `main` that touch
  `site/**`. Pushing a regenerated `site/index.html` publishes it immediately — treat it as a
  release, not a scratch commit. `site/README.md` also carries Hugging Face Space front-matter, so
  that folder is consumed by two hosts.

## Releases

Checkpoints are distributed as tagged GitHub Release assets (`surface_simshift.pt`,
`surface_pretrained.pt`, `multigeo.pt`), each a self-describing `{"model", "meta"}` bundle so
`--eval_only` can reproduce its numbers without remembering training flags. When the first release is
cut, `checkpoints/README.md` must lose its "releases are being cut / request via an issue" note, and
`docs/RESULTS.md` §7 and the roadmap item about releasing checkpoints should be updated in the same
commit.
