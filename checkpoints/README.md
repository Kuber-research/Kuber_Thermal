# Checkpoints

Trained weights are distributed as **GitHub Release assets** (not committed to git, to keep the repo
lean). Each checkpoint is a single `.pt` bundling `{"model": state_dict, "meta": {...}}`, where
`meta` records the geometry mode, conditioning keys, and normalizers - so evaluation is fully
self-describing:

```bash
python -m kuber.train_simshift \
    --data <npz_dir> --splits <splits.json> --difficulty medium \
    --eval_only <checkpoint.pt>
```

## Available checkpoints

| name | geom mode | trained on | headline | size |
|---|---|---|---|---|
| `surface_simshift.pt` | surface | SIMSHIFT (222 cases) | 12.14 K temp RMSE, medium/OOD | ~165 MB |
| `surface_pretrained.pt` | surface (transfer cond) | our corpus → SIMSHIFT | 12.38 K, value-of-data A/B | ~165 MB |
| `multigeo.pt` | surface | heatsinks + cold plates | one model, both device classes | ~165 MB |

> Releases are being cut. Until a tagged release is up, request weights via an issue, or retrain in
> a few GPU-hours with the commands in the top-level README. The eval/harness code here runs against
> any checkpoint produced by `kuber.train_simshift`.

The `meta` block inside each checkpoint is the source of truth for how it was trained - `eval_only`
reads `geom_mode`, `cond_keys`, and `extent_feats` straight from it, so you never have to remember
the flags a checkpoint was trained with.
