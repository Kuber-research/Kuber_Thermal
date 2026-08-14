# Sample cases

Six ready-to-load cases from the Kuber corpus - **3 heatsinks** (natural-convection air) and
**3 cold plates** (forced liquid) - plus a sample split file. Enough to inspect the data contract and
sanity-check a data loader without OpenFOAM or a GPU.

| file prefix | device | notes |
|---|---|---|
| `bsf_out_ov_air_natural_*` | heatsink | carries `sdf` + `sdf_grad` (analytic SDF available) |
| `cp__cp_out_3d_*` | cold plate | heat-flux BC, no analytic SDF (surface cloud only) |

Load one:

```python
import numpy as np, json
d = np.load("data_sample/bsf_out_ov_air_natural_s300__case_0002.npz", allow_pickle=True)
print({k: getattr(d[k], "shape", None) for k in d.files})
print(json.loads(str(d["conditions"])))
```

Full contract: [`../docs/DATASET.md`](../docs/DATASET.md). `splits_sample.json` is a sample
`difficulty → {src,tgt} → {train,val,test}` mapping in the format the trainer expects.
