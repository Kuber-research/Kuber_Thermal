---
name: kuber-training
description: Train KuberNet / GeoTransolver surrogates in this repo — launching runs with kuber.train_simshift, choosing geom_mode and conditioning flags, the WSD schedule, smoke/timing dry-runs, resume and warm-start, and what a checkpoint must carry. Use whenever starting, resuming, debugging, or configuring a training run.
---

# Training a Kuber surrogate

The single training entry point is `kuber/train_simshift.py` (`python -m kuber.train_simshift`).
`scripts/train.sh <data_dir> <splits.json> [geom_mode] [difficulty]` is a thin wrapper around it.
Never write a second trainer — add a flag to this one, because `edge_proof.py`, `deploy/hf-space/infer.py`,
and eval-only all reuse its `load_dataset` / `_build_model` / `_forward` so predictions stay identical
everywhere.

## Before launching anything long

Run these in order. Each catches a different class of failure cheaply.

```bash
# 1. laptop-scale end-to-end smoke (tiny dims, <=8 epochs so all three WSD phases fire)
python -u -m kuber.train_simshift --data <npz_dir> --splits <splits.json> --smoke --geom_mode surface

# 2. real per-epoch cost + peak VRAM on the actual pod, then exit
python -u -m kuber.train_simshift --data <npz_dir> --splits <splits.json> --time_one_epoch --geom_mode surface

# 3. the full run
python -u -m kuber.train_simshift --data <npz_dir> --splits <splits.json> \
    --difficulty medium --geom_mode surface --out outputs/medium_surface
```

Always `python -u` for long runs — the trainer prints per-epoch lines with `flush=True`, and `-u`
keeps them ordered when redirected to a log. `--time_one_epoch` prints an estimate for 100/200 epochs;
use it to decide the horizon before committing GPU hours.

## Environment

Needs the PhysicsNeMo env (`pnemo`): `torch==2.12.1`, `numpy==2.4.6`, `nvidia-physicsnemo==2.1.1`
(`requirements.txt` is pinned to the exact training environment). The GeoTransolver core is imported
lazily by `_import_geotransolver`; if PhysicsNeMo is a clone rather than a pip install, pass
`--pnemo_path <clone>`. A CUDA GPU is required for the multiscale ball-query core; the trainer picks
`cuda` automatically when available and silently falls back to CPU, so check the `[cfg] dev=` line
before assuming a run is on GPU.

## The data contract

Each case is one `.npz` with `coords[16384,3]`, `U[16384,3]`, `T[16384,1]`, `p_rgh[16384,1]`,
`conditions` (JSON object scalar), and — for surface mode — `surf_pts[2048,3]` + `surf_normals[2048,3]`
(otherwise an analytic heatsink surface/SDF is synthesised from the condition scalars).
`splits.json` maps `difficulty -> {src,tgt} -> {train,val,test}`. `docs/DATASET.md` is the contract of
record and `tests/test_data_contract.py` enforces it against `data_sample/`.

Everything is loaded into host RAM up front, and the loader materializes each npz and closes it
immediately — do not "optimize" that into lazy handles: 1.5 k+ open files exceeds `ulimit -n`.

## Preprocessing invariants — do not break these

- **Per-case `[0,1]` frame.** Volume coords are per-axis min–max normalized per case; the surface
  cloud is mapped with the *same* frame so surface and volume stay registered.
- **All normalizers are fit on `src.train` only** — targets (`ymean`/`ystd`), condition scalars
  (`cmean`/`cstd`), extents, Pi groups, SDF. This is what makes the numbers leakage-free and
  difficulty-invariant (a medium-trained model can be evaluated on easy/hard `tgt.test`).
- **`tgt.*` is evaluation-only.** Model selection uses `src.val`. Kuber publishes *no* UDA results;
  never add a loss term or a normalizer that touches the target domain without changing the
  disclosure in `docs/TRAINING.md`, `docs/BENCHMARK.md`, and the leaderboard row.
- Condition columns are auto-selected as those non-constant over `src.train`; `--cond_keys a,b,c`
  forces them, which is required for pretrain→finetune transfer (the two runs must share a
  conditioning vector or the weights will not load).

## Flags that actually change the science

| flag | effect |
|---|---|
| `--geom_mode surface` | production model: surface cloud + normals → encoder → ABL cross-attention descriptor. Reported everywhere. |
| `--geom_mode sdf \| dsdf \| none` | analytic scalar SDF (1 ch) / signed distance + unit gradient (4 ch) / conditions-only ablation floor |
| `--no_abl` | isotropic kNN cross-attention baseline (ABL is on by default) |
| `--geom_wiring deep` | feed the raw surface cloud to GeoTransolver's native per-block geometry cross-attention instead of concatenating a descriptor |
| `--extent_feats` | append log physical bbox extents; **required for multi-geometry** (per-axis framing erases the 36:1 cold-plate duct aspect) |
| `--pi_features` | append Buckingham-Pi groups (Re, Ra, Pe) for cross-fluid transfer |
| `--drop_geom_scalars` | keep only `solidTemp` in conditions — the "transfer conditioning" used by the value-of-data A/B |
| `--refine` | wrap in PDE-Refiner (denoising head); **requires `--geom_mode surface`**, and changes the loss and the inference path (`model.predict`) |
| `--no_local` | GEOT-noLS ablation (multiscale ball-query off) |
| `--track_ood` | diagnostic only: logs `tgt.test` per epoch. It must never influence checkpoint selection. |

## Schedule: Warmup–Stable–Decay

`WSDScheduler` replaces a fixed epoch budget with a convergence-driven one: linear warmup (5 ep) →
hold peak LR `1e-3` while `src.val` mean-nRMSE keeps improving → cosine decay to `1e-6` over 30 epochs
once val plateaus for `--stable_patience` (25) epochs *or* the run approaches `--epochs` (a safety cap
of 1000, not a target), then stop. Watch the `[WSD]` phase-transition lines: a run that enters DECAY
after 6 epochs plateaued instantly (bad LR or bad data), and a run that never leaves STABLE is
hitting the cap.

Checkpointing: best `src.val` mean-nRMSE only, to `<out>/geot_<difficulty>[_noLS]_<geom_mode>[_noscal].pt`.
A full `<out>/resume.pt` (model + optimizer + WSD state + epoch) is written **every** epoch.

## Resuming and warm starts

```bash
# crash recovery — continues mid-run from <out>/resume.pt if it exists
... --resume

# transfer: load weights only, fresh optimizer + fresh WSD cycle
... --init_from pretrain/geot_medium_surface.pt
```
`--resume` and `--init_from` compose: on first launch it warm-starts, on relaunch it continues.
`--init_from` requires the architecture and `--cond_keys` to match the source checkpoint.

## The checkpoint's `meta` block is the contract

Every checkpoint bundles `{"model": state_dict, "opt": ..., "meta": {cond_keys, ymean, ystd, channels,
config}}` where `config` is the *actual* dims used (smoke overrides included). Eval, `edge_proof`, and
the deployed engine all rebuild the model from it via `_dims_from`, so nobody has to remember the
training flags. If you add a hyperparameter that changes model shape, add it to `_dims_from` **and**
give it a default in `edge_proof._arg_defaults()`, or old checkpoints stop loading.

## Reporting a run

A run counts only if it ships with the exact command and environment (`CONTRIBUTING.md`,
`docs/BENCHMARK.md`). The trainer writes `<out>/results_<tag>.json` next to the checkpoint with the
in-dist and OOD metrics, best epoch, and param count — keep it; it is the machine-readable evidence
behind any number that lands in `docs/RESULTS.md` or `results/`.
