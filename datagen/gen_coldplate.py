"""Latin-hypercube generator for single-region cold-plate coolant-channel cases.

Samples channel geometry + coolant + flow + chip heat flux, derives the flow
regime from the Reynolds number (laminar below Re~2300, else kOmegaSST), and
writes one params.json per case plus a manifest.json (consumed by run_sweep_bsf
with --runner run_coldplate.sh --checker check_coldplate).

Usage:  python gen_coldplate.py --n 200 --out cp_cases_s0 --seed 0
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from build_bsf import FLUIDS

# coolant mix (weighted toward the liquids that actually run in cold plates)
FLUID_CHOICES = ["water", "glycol", "oil", "air"]
FLUID_WEIGHTS = [0.45, 0.30, 0.15, 0.10]

# continuous design variables: (name, lo, hi, log?)
DIMS = [
    ("L",    0.03,   0.15,   False),   # channel length (m)
    ("W",    0.003,  0.030,  False),   # channel width  (m)
    ("H",    0.0008, 0.006,  False),   # channel height (m)  (thin)
    ("u_in", 0.05,   2.0,    False),   # inlet velocity (m/s)
    ("q",    1.0e4,  8.0e5,  True),    # base heat flux (W/m^2)  1-80 W/cm^2
    ("T_in", 290.0,  310.0,  False),   # coolant inlet temperature (K)
]


def lhs(n, d, rng):
    """Latin-hypercube samples in [0,1)^d (no scipy)."""
    out = np.zeros((n, d))
    for j in range(d):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True, help="cases dir to create")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    Z = lhs(a.n, len(DIMS), rng)
    fluids = rng.choice(FLUID_CHOICES, size=a.n, p=FLUID_WEIGHTS)

    manifest, regimes = [], {"laminar": 0, "kOmegaSST": 0}
    re_lo, re_hi = 1e18, 0.0
    for i in range(a.n):
        p = {}
        for j, (name, lo, hi, islog) in enumerate(DIMS):
            if islog:
                p[name] = float(10 ** (np.log10(lo) + Z[i, j] * (np.log10(hi) - np.log10(lo))))
            else:
                p[name] = float(lo + Z[i, j] * (hi - lo))
        fluid = str(fluids[i]); p["fluid"] = fluid
        f = FLUIDS[fluid]
        rho = f["rho"] if f["rho"] is not None else 1.18      # air ~ perfectGas
        mu = f["mu"]
        Dh = 2 * p["W"] * p["H"] / (p["W"] + p["H"])
        Re = rho * p["u_in"] * Dh / mu
        p["Re"] = float(Re); p["Dh"] = float(Dh)
        p["turb"] = "laminar" if Re < 2300 else "kOmegaSST"
        regimes[p["turb"]] += 1
        re_lo, re_hi = min(re_lo, Re), max(re_hi, Re)
        cid = f"case_{i:04d}"
        (out / cid).mkdir(exist_ok=True)
        (out / cid / "params.json").write_text(json.dumps(p, indent=2))
        manifest.append({"case_id": cid, "fluid": fluid, "turb": p["turb"], "Re": round(Re, 1)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    from collections import Counter
    print(f"[gen_coldplate] wrote {a.n} cases -> {out}")
    print(f"[gen_coldplate] fluids: {dict(Counter(fluids.tolist()))}")
    print(f"[gen_coldplate] regimes: {regimes}  Re range [{re_lo:.0f}, {re_hi:.0f}]")


if __name__ == "__main__":
    main()
