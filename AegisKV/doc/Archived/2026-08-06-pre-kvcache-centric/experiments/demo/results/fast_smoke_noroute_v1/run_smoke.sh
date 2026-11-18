#!/usr/bin/env bash
set -euo pipefail
cd /home/comp/zhanzhao/sglsim
OUT=AegisKV/demo/results/fast_smoke_noroute_v1
PY=.venv/bin/python
export PYTHONPATH=.
common=(
  -m AegisKV.demo.run_offline_wct
  --server-config AegisKV/demo/config/server_qwen3_32b_dp4_pp4_32k.json
  --trace AegisKV/presentation/experiments/sglsim_large/traces/weka_subagent_cdf_5w_trace_oracle_workload.jsonl
  --failure-manifest AegisKV/demo/config/failure_manifest_qwen3_32b_dp4_pp4_32k_matched_v13.json
  --workflow-concurrency 4
  --failure-count 1
  --max-workflows 5
  --slack-source trace_oracle
  --artifact-detail compact
)
echo "[$(date -Is)] start no_backup" | tee -a "$OUT/smoke.log"
"$PY" "${common[@]}" --policy no_backup --output "$OUT/c4-f1-no_backup.json" >>"$OUT/smoke.log" 2>&1
echo "[$(date -Is)] start aegiskv" | tee -a "$OUT/smoke.log"
"$PY" "${common[@]}" --policy aegiskv --output "$OUT/c4-f1-aegiskv.json" >>"$OUT/smoke.log" 2>&1
"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out=Path("AegisKV/demo/results/fast_smoke_noroute_v1")
cells={}
for policy in ("aegiskv","no_backup"):
    a=json.loads((out/f"c4-f1-{policy}.json").read_text())
    wcts=sorted(float(w["wct_s"]) for w in a["workflows"])
    routes={str(r["rid"]):int(r["execution_dp_rank"]) for w in a["workflows"] for r in w["request_routes"]}
    cells[policy]={"wcts":wcts,"mean":float(np.mean(wcts)),"routes":routes,
                   "retries":sum(len(w.get("retry_events",[])) for w in a["workflows"]),
                   "host_runtime_s":float(a.get("host_runtime_s",0)),
                   "route_violations":int((a.get("route_plan_audit") or {}).get("violation_count",0))}
ra,rb=cells["aegiskv"]["routes"],cells["no_backup"]["routes"]
shared=set(ra)&set(rb)
same=sum(1 for rid in shared if ra[rid]==rb[rid])
mean_a,mean_b=cells["aegiskv"]["mean"],cells["no_backup"]["mean"]
summary={
  "evidence_role":"fast smoke without frozen route plan; not paired formal evidence",
  "workflow_count":5,
  "workflow_concurrency":4,
  "failure_count_requested":1,
  "slack_source":"trace_oracle",
  "route_plan":None,
  "normal_routes_equal": same==len(shared) and len(shared)>0,
  "shared_route_count":len(shared),
  "identical_route_count":same,
  "mean_wct_aegiskv_s":mean_a,
  "mean_wct_no_backup_s":mean_b,
  "mean_wct_improvement_pct":100.0*(mean_b-mean_a)/mean_b if mean_b else 0.0,
  "retries_aegiskv":cells["aegiskv"]["retries"],
  "retries_no_backup":cells["no_backup"]["retries"],
  "host_runtime_aegiskv_s":cells["aegiskv"]["host_runtime_s"],
  "host_runtime_no_backup_s":cells["no_backup"]["host_runtime_s"],
}
(out/"smoke_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
fig,ax=plt.subplots(figsize=(6.2,4.2))
for policy,style in (("aegiskv","-"),("no_backup","--")):
    xs=np.array(cells[policy]["wcts"]); ys=np.arange(1,len(xs)+1)/len(xs)
    ax.step(xs,ys,where="post",linestyle=style,label=policy)
ax.set_xlabel("Workflow completion time (s)"); ax.set_ylabel("CDF")
ax.set_title("Fast smoke WCT CDF (5w, c=4, f=1, no frozen routes)")
ax.grid(True,alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig(out/"smoke_wct_cdf.png",dpi=160); fig.savefig(out/"smoke_wct_cdf.pdf")
print(json.dumps(summary,ensure_ascii=False))
PY
echo "[$(date -Is)] done" | tee -a "$OUT/smoke.log"
