"""Physics filter for single-region buoyantSimpleFoam heatsink npz files.

A case is valid iff the field is finite, temperature is bounded between ambient
and the heated wall, the air was actually heated, there is flow, and there is a
real spatial thermal gradient (a plume/wake, not a uniform field).

check_case(npz_path) -> (ok: bool, msgs: list[str])   [matches run_sweep API]
"""
from __future__ import annotations
import json
import numpy as np


def check_case(npz_path):
    msgs = []
    d = np.load(npz_path, allow_pickle=True)
    T = np.asarray(d["T"], dtype=np.float64).reshape(-1)
    U = np.asarray(d["U"], dtype=np.float64).reshape(-1, 3)
    p = json.loads(str(d["params"]))
    Tamb, Twall = float(p["T_amb"]), float(p["T_wall"])

    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(U))):
        msgs.append("non-finite field")
    if T.size < 1000:
        msgs.append(f"too few cells ({T.size})")
    tmin, tmax = float(T.min()), float(T.max())
    if tmin < Tamb - 3:
        msgs.append(f"Tmin {tmin:.1f} < ambient {Tamb}")
    if tmax > Twall + 3:
        msgs.append(f"Tmax {tmax:.1f} > wall {Twall} (overshoot)")
    if tmax < Tamb + 3:
        msgs.append(f"not heated (Tmax {tmax:.1f} ~ ambient)")
    umag = np.linalg.norm(U, axis=1)
    if float(umag.max()) < 1e-4:
        msgs.append("no flow (|U|~0)")
    if float(T.std()) < 0.05:
        msgs.append(f"no thermal gradient (T std {T.std():.3f})")
    return (len(msgs) == 0, msgs)


if __name__ == "__main__":
    import sys
    ok, m = check_case(sys.argv[1])
    print("OK" if ok else "REJECT", m)
