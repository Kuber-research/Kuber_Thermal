"""Physics filter for single-region cold-plate coolant-channel npz files.

A case is valid iff the field is finite, the coolant actually warmed up (Tmax
above inlet), temperature is physical (>= inlet - a little, and not a runaway),
there is flow, and there is a real spatial thermal gradient (development along
the channel, not a uniform field).

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
    Tin = float(p["T_in"])

    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(U))):
        msgs.append("non-finite field")
    if T.size < 1000:
        msgs.append(f"too few cells ({T.size})")
    tmin, tmax = float(T.min()), float(T.max())
    if tmin < Tin - 3:
        msgs.append(f"Tmin {tmin:.1f} < inlet {Tin} (undershoot)")
    if tmax > Tin + 490:
        msgs.append(f"Tmax {tmax:.1f} near limiter cap (runaway/unconverged)")
    if tmax < Tin + 1:
        msgs.append(f"not heated (Tmax {tmax:.1f} ~ inlet)")
    umag = np.linalg.norm(U, axis=1)
    if float(umag.max()) < 1e-4:
        msgs.append("no flow (|U|~0)")
    if float(T.std()) < 0.02:
        msgs.append(f"no thermal gradient (T std {T.std():.3f})")
    return (len(msgs) == 0, msgs)


if __name__ == "__main__":
    import sys
    ok, m = check_case(sys.argv[1])
    print("OK" if ok else "REJECT", m)
