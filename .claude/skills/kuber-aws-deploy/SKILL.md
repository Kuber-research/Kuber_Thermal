---
name: kuber-aws-deploy
description: Deploy Kuber on AWS — containerizing the FastAPI inference app for ECS/Fargate/App Runner, serving checkpoints and case bundles from S3, sizing CPU/GPU/RAM against the real boot cost, and running GPU training on EC2 spot with the trainer's resume path. Use for any AWS hosting, inference-endpoint, or cloud-training question in this repo.
---

# Deploying Kuber on AWS

**There is no AWS deployment in this repo today.** The only shipped deployment is the Hugging Face
Docker Space in `deploy/hf-space/` (FastAPI + CPU inference, assets pulled from an HF dataset repo at
boot). This skill describes how to move that same app to AWS without forking it, and what the
repo's actual runtime characteristics force you to size for.

If you build one, commit the artifacts under `deploy/aws/` with the same reproduction discipline the
rest of the repo demands (`CONTRIBUTING.md`): exact commands, exact image tag, exact asset bundle.

## Licensing gate — check before hosting

Kuber is **PolyForm Noncommercial 1.0.0**. Research, education, evaluation, and other noncommercial
use are free; **commercial use requires a separate license**. Standing up a hosted inference endpoint
that serves a commercial product is a licensing decision, not an infrastructure one — raise it rather
than quietly deploying it.

## What you are actually deploying

`deploy/hf-space/` is the unit of deployment. Do not rewrite the server for AWS:

- `boot.py` — entrypoint: fetches assets, sets `KUBER_CKPT` / `KUBER_DATA` / `KUBER_SPLITS`, runs
  uvicorn on `$PORT` (default 7860).
- `app.py` — FastAPI app; constructs `Engine` **inside the lifespan handler**, so the app does not
  serve until the model is loaded.
- `infer.py` — the `Engine`, which reuses the trainer's `load_dataset` / `_build_model` / `_forward`
  so served physics matches trained physics exactly.
- `static/` — the UI (~850 KB), served from the image.
- `Dockerfile` — `python:3.11-slim`, CPU torch wheels, code-only image.

### The stale vendored package — check this first

`deploy/hf-space/kuber/` is a **copy** of the top-level `kuber/` package and is currently behind it
(no ABL cross-attention, no `--pi_features`). The image imports that copy. An ABL-trained checkpoint
carries a `cross.gamma_raw` parameter the old code does not define, so `load_state_dict` fails at
container start with an unexpected-key error — an AWS deploy will look like an infra bug and be a
sync bug. Diff and refresh before building any image:

```bash
for f in kuber/*.py; do diff -q "$f" "deploy/hf-space/$f" >/dev/null || echo "DIFF $f"; done
```

## Sizing — driven by how `Engine` boots

`Engine.__init__` calls `load_dataset(...)` over the **whole split** and holds it in host RAM,
because normalizers (`ymean/ystd`, `cmean/cstd`, extents) are fit on the training split exactly as in
training. Consequences, all of which set AWS parameters:

| property | value | what it forces |
|---|---|---|
| asset bundle | checkpoint ~165 MB + the `.npz` cases the split names (~2.3 k cases, ~1 GB) | S3, not the image |
| cold boot | asset download + normalizer fit, **~1–2 min** | long health-check grace period; no scale-to-zero unless a 2-min first request is acceptable |
| resident RAM | roughly 1.4 MB/case of arrays → **several GB** for the demo split | task memory ≥ 8 GB; do not size from the checkpoint |
| inference | ~1.5 s/case on CPU, single-threaded-ish | vCPU count matters more than count of tasks for latency |
| GPU | optional; `Engine` picks `cuda` when available and warms it up on boot | only worth it above modest QPS |

The standing optimization noted in `deploy/hf-space/README.md` is to **ship precomputed normalizers**
so boot needs only the checkpoint plus a few preset cases. On AWS that is the single highest-value
change: it removes the ~1 GB download, most of the RAM, and most of the cold start, and it is what
makes Lambda or scale-to-zero viable at all. Consider doing it before building infrastructure around
the current boot cost.

## Choosing a target

| target | when | notes |
|---|---|---|
| **ECS on Fargate + ALB** | default for the demo/API | 2–4 vCPU / 8–16 GB task; `linux/amd64` (torch + physicsnemo wheels); ALB health check on `/api/info` with a startup grace ≥ 180 s |
| **App Runner** | simplest managed HTTP | set `PORT=8080`; same image; less control over health-check grace |
| **EC2 GPU (g5/g6) + ECS or systemd** | GPU inference or a long-lived internal endpoint | needs a CUDA-matched torch build, not the CPU wheel index in `deploy/hf-space/requirements.txt` |
| **SageMaker endpoint** | if the org already standardizes on it | requires a `/ping` + `/invocations` shim over `app.py`; don't reshape `Engine` for it |
| **Lambda (container)** | only after precomputed normalizers | today's 1–2 min init and multi-GB working set make it a poor fit |

## Assets: S3 replaces the HF dataset repo

Keep the image code-only. `prepare_assets.py` already assembles the exact bundle layout
(`multigeo.pt`, `splits.json`, `cases/<id>.npz` — only the cases the split references, not the 1.4 GB
corpus); point it at a local `--out` and sync that to S3:

```bash
python deploy/hf-space/prepare_assets.py --ckpt <ckpt.pt> --corpus <corpus_dir> \
    --splits <splits.json> --out /tmp/kuber-assets
aws s3 sync /tmp/kuber-assets s3://<bucket>/kuber-assets/<version>/ --exact-timestamps
```

Version the prefix (`.../v1/`, or the checkpoint's git SHA). A checkpoint and the split that fit its
normalizers are one atomic artifact — never let a task pull a new checkpoint against an old split.

`boot.py` currently knows only `KUBER_HF_REPO`. Add an S3 branch beside it rather than replacing it,
so the HF Space keeps working:

```python
s3 = os.environ.get("KUBER_S3_URI", "").strip()      # e.g. s3://bucket/kuber-assets/v1
if s3 and not (ASSETS / "multigeo.pt").exists():
    subprocess.run(["aws", "s3", "sync", s3, str(ASSETS)], check=True)
```

Credentials come from the **task role** (ECS) or instance profile — never bake keys into the image or
pass them as plaintext env vars. Scope the policy to `s3:GetObject`/`s3:ListBucket` on that one
prefix. Use Secrets Manager or SSM Parameter Store for anything genuinely secret (e.g. `HF_TOKEN` if a
mixed deployment still pulls from HF).

Do not put the corpus in a public bucket without checking provenance: the "0 cases from SIMSHIFT or
any licensed/scraped source" claim is only true of Kuber's own OpenFOAM data.

## Build and push

```bash
aws ecr get-login-password --region <region> \
  | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker build --platform linux/amd64 -t kuber-live:<sha> deploy/hf-space
docker tag kuber-live:<sha> <acct>.dkr.ecr.<region>.amazonaws.com/kuber-live:<sha>
docker push <acct>.dkr.ecr.<region>.amazonaws.com/kuber-live:<sha>
```

Tag images by git SHA, never `latest` — the served numbers must be traceable to a commit, same as any
published result. Confirm the container locally before pushing (it should print
`[app] ready: device=cpu cases=N`):

```bash
docker run --rm -p 7860:7860 -e KUBER_S3_URI=... -e PORT=7860 kuber-live:<sha>
```

## Training on AWS

Training is a separate job from serving. Use a GPU instance (the multiscale ball-query core needs
CUDA), install torch from the CUDA-matched index and a matching `nvidia-physicsnemo` build — the CPU
pins in `deploy/hf-space/requirements.txt` are for inference only.

Spot instances are a good fit because the trainer is already crash-safe: it writes
`<out>/resume.pt` (model + optimizer + WSD state + epoch) **every epoch**, and `--resume` continues
mid-run. Pattern:

```bash
python -u -m kuber.train_simshift --data <npz> --splits <splits> \
    --difficulty medium --geom_mode surface --out /mnt/outputs/medium_surface --resume
```

- Put `--out` on a persistent volume (EBS) and sync `resume.pt` + the best checkpoint to S3 after each
  epoch or on a timer; a spot reclaim then costs one epoch, not the run.
- `--time_one_epoch` first, on the actual instance type, before committing to an instance-hours budget.
- The corpus lives in S3; stage it to local NVMe/EBS before training — `load_dataset` reads every
  referenced `.npz` up front, so a network filesystem shows up directly as startup time.
- Trained checkpoints are released as **GitHub Release assets** (`checkpoints/README.md`), so S3 is
  the working store, not the distribution channel.

## Operational checks

- Health check `/api/info` (returns the device label). It only answers once the lifespan handler has
  built the `Engine`, which makes it a true readiness signal — give it the startup grace it needs
  instead of shortening the boot.
- Log the startup lines (`[boot] downloading assets...`, `[app] ready: device=... cases=...`) to
  CloudWatch; they are how you tell an asset-sync failure from a model-load failure.
- Watch task memory, not CPU, for OOM kills — the resident dataset is the dominant term.
- One task can serve many requests, but predictions are synchronous CPU work; scale out on
  concurrency, and keep at least one warm task if cold starts are user-visible.
