---
name: kuber-optimization
description: Improve a Kuber surrogate — running honest ablations and hyperparameter sweeps, fixing VRAM/throughput bottlenecks in the surface branch and multiscale core, tuning the WSD schedule, and cutting inference latency for the deployed engine. Use when a run is too slow, OOMs, plateaus, or when comparing architecture/conditioning variants.
---

# Optimizing Kuber runs and models

Two different jobs share this skill: making a run *better* (accuracy per parameter) and making it
*cheaper* (time, VRAM, latency). Keep them separate in any experiment — a change that improves both
usually did neither and moved something else.

## Ablation method

The repo's results are A/B claims (ABL vs isotropic kNN, surface vs SDF, pretrained vs scratch), so
the method matters as much as the number.

1. **Change one thing.** The trainer is designed for this: every variant is a flag, and the tag in the
   output filename (`geot_<difficulty>[_noLS]_<geom_mode>[_noscal].pt`) encodes it. Give each variant
   its own `--out` so results JSONs don't overwrite.
2. **Select on `src.val` mean nRMSE, always.** That is what checkpointing tracks and what SIMSHIFT uses.
   `--track_ood` may log the target trajectory for diagnosis, but choosing a variant by OOD score is
   selecting on the test set — it turns a zero-shot claim into a tuned one.
3. **Match the budget.** The leaderboard claim is "same ~14 M parameter budget, better geometry
   conditioning and recipe". A variant that wins by growing `n_hidden` is a different claim; report the
   measured param count the trainer prints.
4. **Hold the seedless parts fixed.** Point subsampling reseeds per epoch (`_subsample_idx(…, seed=ep)`)
   and evaluation uses a fixed seed 123, so eval is comparable across runs; don't change `n_points`
   between the two arms of an A/B.
5. **Run the stability proof on the winner**, not just RMSE — see the `kuber-evaluation` skill. An
   ablation that improves mean nRMSE while dropping the edge ∇T ratio bought accuracy with smoothing.

Useful existing axes: `--geom_mode {surface,dsdf,sdf,none}`, `--no_abl`, `--geom_wiring deep`,
`--no_local`, `--extent_feats`, `--pi_features`, `--drop_geom_scalars`, `--refine`,
`--init_from` (pretrain → finetune, the value-of-data A/B).

## Where the compute actually goes

Measure before tuning: `--time_one_epoch` prints wall-clock per epoch **and** peak VRAM, and estimates
the 100/200-epoch cost. Re-run it after any change that touches shapes.

The dominant costs in `--geom_mode surface`:

- **kNN in `LocalSurfaceCrossAttention`.** `knn_idx` builds a dense `torch.cdist` of
  `[B, N, Ns]` = 4 × 16384 × 2048 floats ≈ 0.5 GB before the top-k. This is usually the first thing to
  OOM and it scales linearly in `--n_points` × `--n_surf` × batch.
- **Multiscale ball-query** in the GeoTransolver core (`--radii`, `--neighbors`); `--no_local` removes
  it entirely (that is the noLS ablation, not a free speedup).
- **The transformer core**, `n_hidden` × `n_layers` × `slice_num`.

Levers, roughly in order of cost-cut per unit of accuracy lost:

| lever | effect |
|---|---|
| `--batch_size` | linear in VRAM; the cheapest OOM fix, costs schedule noise |
| `--n_points` (default 16384) | linear in both the kNN matrix and the core; points are resampled each epoch, so training on a subset still sees the whole cloud over a run |
| `--n_surf` (2048) | linear in the kNN matrix and the surface encoder's self-attention (quadratic there) |
| `--knn_k` (16) | linear in the attention over neighbours, not in the cdist |
| `--d_surf`/`--d_geo`/`--surf_layers` | the surface branch is ~0.3 M params — shrinking it saves little memory, don't start here |
| `--n_hidden`/`--n_layers`/`--slice_num` | the real parameter budget; changing these changes the claim |

Do not reach for gradient accumulation or mixed precision as a first move — neither is wired in, and
adding them changes the reproduction command for every published number.

## Schedule tuning

WSD (`WSDScheduler`) already adapts the horizon: it holds peak LR while `src.val` improves and only
spends the cosine decay once it plateaus. The largest val drop typically arrives in the decay phase, so
a run killed during STABLE is not a finished run. Read the `[WSD]` lines:

- decay triggered within a few epochs → the peak LR is wrong or the data/normalizers are broken;
  re-check `[data] cond_dim/feat_dim` before touching the schedule.
- "approaching max_epochs" as the decay reason → the run hit the `--epochs` safety cap; raise the cap
  rather than shortening `--decay`, so the anneal stays full length.
- long STABLE with noisy val → raise `--stable_patience` before lowering `--lr`; that is what the
  patience is for.

`--warmup`, `--decay`, `--stable_patience`, `--min_lr` are all asserted consistent at construction
(`0 < min_lr < peak_lr`, `max_epochs > warmup + decay`), so a bad combination fails immediately rather
than mid-run.

## Inference latency

Inference is the product's whole point (sub-second vs CFD-minutes), and it is *geometry-independent* —
a forward pass does not grow with mesh size the way a CFD solve does.

- The deployed engine (`deploy/hf-space/infer.py`) loads the checkpoint **once**, holds model +
  normalizers + cases in memory, and reuses `load_dataset` / `_build_model` / `_forward` so the demo's
  physics matches training exactly. Keep that reuse; a bespoke inference path is how demo and paper
  numbers drift apart.
- `--refine` (PDE-Refiner) costs **K+1 forward passes** per prediction (`predict` runs step 0 plus K
  denoising steps). It buys high-frequency fidelity and an uncertainty estimate, not speed; all reported
  numbers use the one-shot model.
- Query points are arbitrary — latency scales with how many you ask for, so a viewer or an optimizer
  loop can trade resolution for speed without retraining.
- `edge_proof` runs case-by-case at batch 1 on CPU by design; it is a proof harness, not a throughput
  path. Don't "optimize" it onto the GPU to save minutes.
- Published latency is still an **estimate**. If you measure it, record the GPU/CPU and update
  `docs/RESULTS.md` §5 and §7 together — the caveat and the number are one edit.
