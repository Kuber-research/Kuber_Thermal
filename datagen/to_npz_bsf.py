"""Export a converged single-region buoyantSimpleFoam case -> compact .npz.

Fields saved: cell-centre coords (N,3) + T (N,), U (N,3), p_rgh (N,), plus the
geometry/BC params. Uses `postProcess -func writeCellCentres` then parses the
OpenFOAM ascii internalField lists (no external deps).

Usage: python to_npz_bsf.py --case <dir> --out <dir>
"""
from __future__ import annotations
import argparse, json, re, subprocess, os
from pathlib import Path
import numpy as np


def _latest_time(case: Path) -> str:
    times = [d.name for d in case.iterdir()
             if d.is_dir() and re.fullmatch(r"\d+(\.\d+)?", d.name) and d.name != "0"]
    if not times:
        return "0"
    return max(times, key=lambda t: float(t))


def _read_internal(path: Path):
    """Parse internalField from an OpenFOAM ascii field file -> np.ndarray."""
    txt = path.read_text()
    m = re.search(r"internalField\s+(uniform|nonuniform)", txt)
    if not m:
        raise ValueError(f"no internalField in {path}")
    if m.group(1) == "uniform":
        # uniform scalar or vector
        rest = txt[m.end():]
        vec = re.match(r"\s+\(([^)]+)\)", rest)
        if vec:
            return np.array([float(x) for x in vec.group(1).split()])
        val = re.match(r"\s+([-\d.eE+]+)", rest)
        return np.array([float(val.group(1))])
    # nonuniform: List<scalar> N ( ... )  or  List<vector> N ( (x y z) ... )
    body = txt[m.end():]
    n = int(re.search(r"List<\w+>\s*\n?\s*(\d+)", body).group(1))
    start = body.index("(", body.index("List<"))
    depth, i, chunk = 0, start, []
    # grab everything between the outermost ( ) after the count
    end = body.index(")", start)
    # vectors have inner parens
    if "((" in body[start:start+4] or re.search(r"\(\s*[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\)", body[start:start+200]):
        tuples = re.findall(r"\(([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\)", body[start:])
        arr = np.array(tuples[:n], dtype=float)
        return arr
    # scalars
    inner = body[start+1: body.index("\n)", start) if "\n)" in body[start:] else body.rindex(")")]
    vals = [float(x) for x in inner.split()][:n]
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    case = Path(a.case).resolve()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    cid = case.name
    # cell centres
    subprocess.run(["postProcess", "-func", "writeCellCentres", "-latestTime"],
                   cwd=case, capture_output=True, text=True)
    t = _latest_time(case)
    td = case / t
    C = _read_internal(td / "C")
    T = _read_internal(td / "T")
    U = _read_internal(td / "U")
    prgh = _read_internal(td / "p_rgh")
    params = json.loads((case / "params.json").read_text())
    np.savez_compressed(out / f"{cid}.npz",
                        coords=C.astype(np.float32), T=T.astype(np.float32),
                        U=U.astype(np.float32), p_rgh=prgh.astype(np.float32),
                        params=json.dumps(params), time=t)
    print(f"[to_npz] {cid}: N={len(T)} Trange=[{T.min():.1f},{T.max():.1f}] -> {out/(cid+'.npz')}")


if __name__ == "__main__":
    main()
