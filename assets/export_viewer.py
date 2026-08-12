import os, json, sys
os.chdir("/home/shubhj/cfd_thermal_mvp"); sys.path.insert(0, "/home/shubhj/cfd_thermal_mvp")
from demo.infer import Engine
e = Engine("outputs/multigeo.pt", "demo_data/multigeo_corpus", "demo_data/multigeo_splits_demo.json")
cases = e.list_cases()
hs = [c for c in cases if c["device"] == "heatsink"]
cp = [c for c in cases if c["device"] == "coldplate"]
hs_pick = max(hs, key=lambda c: c.get("solidTemp", 0))
cp_pick = max(cp, key=lambda c: c.get("heatFlux", 0))
out = {}
for key, pick in [("heatsink", hs_pick), ("coldplate", cp_pick)]:
    d = e.predict(pick["index"], max_points=8000)
    d["title"] = pick["title"]; d["sub"] = pick["sub"]
    d["fields"].pop("p_rgh", None)
    out[key] = d
    m = d["metrics"]
    print(key, "|", pick["title"], "| T_rmse", m["T_rmse"], "K | peak_T pred", m["peak_T"], "gt", m["peak_T_gt"], "| n", d["n_points"], "| infer", d["infer_ms"], "ms")
json.dump(out, open(os.path.join("/tmp/claude-1001/-home-shubhj-ucr-rl/b4bd3237-32f5-448b-b831-429d0f7f9cb1/scratchpad", "viewer_data.json"), "w"))
print("saved", os.path.getsize(os.path.join("/tmp/claude-1001/-home-shubhj-ucr-rl/b4bd3237-32f5-448b-b831-429d0f7f9cb1/scratchpad", "viewer_data.json")) // 1024, "KB")
