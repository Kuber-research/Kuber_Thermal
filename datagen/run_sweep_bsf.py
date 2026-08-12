"""Resumable, parallel buoyantSimpleFoam heatsink dataset generator.

Same guarantees as the CHT run_sweep: atomic per-case (npz written only on
converged solve, then raw pruned), resume by re-running the same command,
status.json rewritten after every case, physics filter -> rejected/, graceful
Ctrl-C. Calls run_bsf.sh and filters with check_bsf.check_case.

Usage:
  python -u run_sweep_bsf.py --cases <cases_dir> --out <out_dir> \
         --scripts <bsf_scripts_dir> --jobs 5 --iters 800 --timeout 5400
"""
from __future__ import annotations
import argparse, json, os, shutil, signal, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _clean_keep_params(case_dir):
    """On failure, remove heavy artifacts (mesh/time dirs/logs) but KEEP params.json
    so `--retry_failed` can regenerate the case. (Bug fix: a full rm -rf lost the input.)"""
    cd = Path(case_dir)
    if not cd.exists():
        return
    for item in cd.iterdir():
        if item.name == "params.json":
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try:
                item.unlink()
            except OSError:
                pass


def _atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _load_status(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def run_one_case(case_dir, scripts, out, iters, timeout, runner="run_bsf.sh", checker="check_bsf"):
    cid = Path(case_dir).name
    npz = Path(out) / f"{cid}.npz"
    if npz.exists():
        return {"case_id": cid, "state": "done", "note": "npz pre-existed"}
    run_one = str(Path(scripts) / runner)
    t0 = time.time()
    try:
        proc = subprocess.run(["bash", run_one, case_dir, scripts, out, str(iters)],
                              timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        _clean_keep_params(case_dir)
        return {"case_id": cid, "state": "failed", "note": f"timeout>{timeout}s",
                "secs": round(time.time() - t0)}
    if not npz.exists():
        # save the runner's output so failures are diagnosable (not silently discarded)
        try:
            (Path(out) / f"{cid}.faillog").write_text(
                f"RC={proc.returncode}\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}")
        except Exception:
            pass
        _clean_keep_params(case_dir)
        return {"case_id": cid, "state": "failed", "note": "no npz (mesh/solve/converge)",
                "secs": round(time.time() - t0)}
    try:
        import importlib
        check_case = importlib.import_module(checker).check_case
        ok, msgs = check_case(str(npz))
    except Exception as e:
        ok, msgs = False, [f"check error: {e}"]
    if not ok:
        rej = Path(out) / "rejected"; rej.mkdir(exist_ok=True)
        os.replace(npz, rej / npz.name)
        return {"case_id": cid, "state": "rejected", "secs": round(time.time() - t0),
                "note": "physics filter", "msgs": msgs}
    return {"case_id": cid, "state": "done", "secs": round(time.time() - t0)}


def main(a):
    cases_root = Path(a.cases).resolve()
    a.scripts = str(Path(a.scripts).resolve())
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    manifest = json.loads((cases_root / "manifest.json").read_text())
    all_ids = [c["case_id"] for c in manifest]

    status = _load_status(status_path)
    for cid in all_ids:
        if (out / f"{cid}.npz").exists():
            status.setdefault(cid, {"state": "done", "note": "found on disk"})

    def todo(cid):
        st = status.get(cid, {}).get("state")
        if st == "done":
            return False
        if st in ("failed", "rejected") and not a.retry_failed:
            return False
        return True
    pending = [c for c in manifest if todo(c["case_id"])]
    done0 = sum(1 for c in all_ids if status.get(c, {}).get("state") == "done")
    print(f"[sweep] total={len(all_ids)} done={done0} pending={len(pending)} "
          f"jobs={a.jobs} iters={a.iters} out={out}", flush=True)
    if not pending:
        print("[sweep] nothing to do — dataset complete."); return

    stop = {"flag": False}
    def _handler(signum, frame):
        stop["flag"] = True
        print(f"\n[sweep] signal {signum} — finishing in-flight, no new launches. "
              f"Re-run the same command to resume.", flush=True)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    t_start = time.time(); n_done = n_fail = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futures, it = {}, iter(pending)
        for _ in range(a.jobs):
            try:
                c = next(it)
            except StopIteration:
                break
            futures[ex.submit(run_one_case, str(cases_root / c["case_id"]),
                              a.scripts, str(out), a.iters, a.timeout,
                              a.runner, a.checker)] = c["case_id"]
        while futures:
            for fut in as_completed(list(futures)):
                cid = futures.pop(fut)
                res = fut.result()
                status[res["case_id"]] = {k: v for k, v in res.items() if k != "case_id"}
                _atomic_write_json(status_path, status)
                if res["state"] == "done":
                    n_done += 1
                else:
                    n_fail += 1
                nd = sum(1 for s in status.values() if s.get("state") == "done")
                print(f"[{nd}/{len(all_ids)}] {res['case_id']}: {res['state']} "
                      f"{res.get('note','')} {res.get('secs','')}s {res.get('msgs','')}", flush=True)
                if not stop["flag"]:
                    try:
                        c = next(it)
                        futures[ex.submit(run_one_case, str(cases_root / c["case_id"]),
                                          a.scripts, str(out), a.iters, a.timeout,
                                          a.runner, a.checker)] = c["case_id"]
                    except StopIteration:
                        pass
                break
            if stop["flag"] and not futures:
                break
    nd = sum(1 for s in status.values() if s.get("state") == "done")
    print(f"\n[sweep] {'STOPPED' if stop['flag'] else 'FINISHED'} | done={nd}/{len(all_ids)} "
          f"| this run: +{n_done} done, {n_fail} failed/rejected | {(time.time()-t_start)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scripts", required=True)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--timeout", type=int, default=5400)
    p.add_argument("--retry_failed", action="store_true")
    p.add_argument("--runner", default="run_bsf.sh", help="per-case run script in --scripts")
    p.add_argument("--checker", default="check_bsf", help="physics-filter module (importable)")
    main(p.parse_args())
