#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "presentation"
    / "figures"
    / "data"
    / "projected"
    / "fig_weka_subagent_policy_wct_cdf_projected_128k_dp64.csv"
)
DEFAULT_ASSUMPTIONS = DEFAULT_CSV.with_suffix(".assumptions.json")
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "presentation"
    / "figures"
    / "html"
    / "aegiskv_vs_no_backup_dp64_128k_simulated.html"
)

TOTAL_GPUS = 64
TRACE_HOURS = 15.0
REPORTED_AVERAGE_AVAILABLE = 58.07

# Digitized from the user-provided availability plot.  The schedule is later
# adjusted by at most one rank over long plateaus so its time-weighted mean is
# exactly the reported 58.07 GPUs.  It is a simulated schedule, not the missing
# original trace file.
RAW_AVAILABILITY_POINTS = (
    (0.000, 64),
    (0.768, 54),
    (1.067, 64),
    (1.451, 59),
    (1.814, 57),
    (1.899, 61),
    (2.006, 64),
    (2.112, 61),
    (2.219, 64),
    (2.304, 62),
    (2.411, 64),
    (2.795, 60),
    (3.158, 53),
    (3.243, 17),
    (3.393, 38),
    (3.755, 48),
    (3.969, 58),
    (4.075, 64),
    (4.161, 63),
    (4.950, 64),
    (5.057, 60),
    (5.633, 59),
    (7.148, 64),
    (7.404, 59),
    (7.489, 60),
    (7.532, 64),
    (7.980, 63),
    (8.279, 62),
    (8.556, 59),
    (8.684, 61),
    (8.812, 59),
    (8.940, 63),
    (9.068, 64),
    (9.388, 58),
    (15.000, 58),
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _load_rows(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {"aegiskv": [], "no_backup": []}
    summaries: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            policy = str(row["policy_suffix"])
            if policy not in samples:
                continue
            if row["workflow_id"] == "__summary__":
                summaries[policy] = row
            else:
                samples[policy].append(
                    {
                        "id": str(row["workflow_id"]),
                        "wct_s": float(row["projected_wct_s"]),
                    }
                )
    if set(summaries) != set(samples) or any(len(rows) != 160 for rows in samples.values()):
        raise ValueError("expected two policy summaries and 160 samples per policy")
    return samples, summaries


def _raw_segments() -> list[dict[str, Any]]:
    return [
        {"start_h": start, "end_h": RAW_AVAILABILITY_POINTS[index + 1][0], "available": available}
        for index, (start, available) in enumerate(RAW_AVAILABILITY_POINTS[:-1])
    ]


def _calibrated_segments() -> list[dict[str, Any]]:
    """Match the plot's reported mean while changing plateaus by at most one rank."""
    segments = _raw_segments()
    current_gpu_hours = sum(
        (segment["end_h"] - segment["start_h"]) * segment["available"]
        for segment in segments
    )
    target_gpu_hours = REPORTED_AVERAGE_AVAILABLE * TRACE_HOURS
    excess = current_gpu_hours - target_gpu_hours
    if excess < -1e-6:
        raise ValueError("digitized schedule is below the reported average")

    # Prefer long, non-outage plateaus.  A one-rank correction is within the
    # vertical resolution of the source plot and avoids changing its burst shape.
    chosen = sorted(
        range(len(segments)),
        key=lambda index: (segments[index]["end_h"] - segments[index]["start_h"]),
        reverse=True,
    )
    for index in chosen:
        if excess <= 1e-9:
            break
        segment = segments[index]
        if segment["available"] <= 17:
            continue
        duration = segment["end_h"] - segment["start_h"]
        if excess >= duration - 1e-9:
            segment["available"] -= 1
            excess -= duration
        else:
            split = segment["start_h"] + excess
            original_end = segment["end_h"]
            original_available = segment["available"]
            segment["end_h"] = split
            segment["available"] = original_available - 1
            segments.append(
                {
                    "start_h": split,
                    "end_h": original_end,
                    "available": original_available,
                }
            )
            excess = 0.0
    segments.sort(key=lambda row: row["start_h"])

    for index, segment in enumerate(segments):
        failed_count = TOTAL_GPUS - int(segment["available"])
        seed = (index * 17 + 3) % TOTAL_GPUS
        order = [((seed + offset * 29) % TOTAL_GPUS) for offset in range(TOTAL_GPUS)]
        segment["failed_ranks"] = order[:failed_count]
        segment["start_h"] = round(float(segment["start_h"]), 6)
        segment["end_h"] = round(float(segment["end_h"]), 6)

    average = sum(
        (segment["end_h"] - segment["start_h"]) * segment["available"]
        for segment in segments
    ) / TRACE_HOURS
    if abs(average - REPORTED_AVERAGE_AVAILABLE) > 2e-5:
        raise ValueError(f"availability calibration failed: {average}")
    return segments


def _event_trace_hours(segments: list[dict[str, Any]], count: int) -> list[float]:
    weighted = [
        (segment["end_h"] - segment["start_h"])
        * (TOTAL_GPUS - segment["available"])
        for segment in segments
    ]
    total = sum(weighted)
    result: list[float] = []
    for event_index in range(count):
        target = (event_index + 0.5) * total / count
        cumulative = 0.0
        for segment, weight in zip(segments, weighted):
            if cumulative + weight >= target and weight > 0:
                fraction = (target - cumulative) / weight
                result.append(
                    segment["start_h"]
                    + fraction * (segment["end_h"] - segment["start_h"])
                )
                break
            cumulative += weight
    if len(result) != count:
        raise ValueError("failed to distribute simulated retry events")
    return result


def _segment_at(segments: list[dict[str, Any]], trace_h: float) -> dict[str, Any]:
    for segment in segments:
        if segment["start_h"] <= trace_h < segment["end_h"]:
            return segment
    return segments[-1]


def _allocate(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _action_sequence(count: int, token_totals: dict[str, int]) -> list[str]:
    total_tokens = sum(token_totals.values())
    counts = {
        action: round(count * tokens / total_tokens)
        for action, tokens in token_totals.items()
    }
    delta = count - sum(counts.values())
    counts["recompute"] += delta
    remaining = dict(counts)
    sequence: list[str] = []
    for index in range(count):
        # Weighted round-robin keeps all sources visible throughout playback.
        action = max(
            remaining,
            key=lambda name: remaining[name] / max(counts[name], 1),
        )
        sequence.append(action)
        remaining[action] -= 1
    return sequence


def _build_events(
    *,
    policy: str,
    retry_count: int,
    token_totals: dict[str, int],
    workflow_ids: list[str],
    duration_s: float,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace_hours = _event_trace_hours(segments, retry_count)
    if policy == "no_backup":
        actions = ["recompute"] * retry_count
    else:
        actions = _action_sequence(retry_count, token_totals)
    action_indices = {name: [] for name in token_totals}
    for index, action in enumerate(actions):
        action_indices[action].append(index)
    per_event_tokens: dict[int, int] = {}
    for action, indices in action_indices.items():
        allocations = _allocate(token_totals[action], len(indices))
        per_event_tokens.update(zip(indices, allocations))

    events = []
    for index, (trace_h, action) in enumerate(zip(trace_hours, actions)):
        segment = _segment_at(segments, trace_h)
        failed = list(segment["failed_ranks"])
        failed_rank = failed[index % len(failed)]
        healthy = [rank for rank in range(TOTAL_GPUS) if rank not in failed]
        retry_rank = healthy[(index * 11 + 5) % len(healthy)]
        events.append(
            {
                "index": index,
                "trace_h": round(trace_h, 6),
                "t_s": round(duration_s * trace_h / TRACE_HOURS, 6),
                "workflow_id": workflow_ids[(index * 37 + 7) % len(workflow_ids)],
                "failed_rank": failed_rank,
                "retry_rank": retry_rank,
                "action": action,
                "tokens": per_event_tokens[index],
            }
        )
    return events


def _integer(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def _floating(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _build_data(csv_path: Path, assumptions_path: Path) -> dict[str, Any]:
    samples, summaries = _load_rows(csv_path)
    assumptions = _read_json(assumptions_path)
    segments = _calibrated_segments()
    duration_s = max(_floating(summaries[name], "max_wct_s") for name in summaries)
    policies = []
    for policy, label, color in (
        ("aegiskv", "AegisKV", "#18a8e0"),
        ("no_backup", "No Backup", "#8b98aa"),
    ):
        summary = summaries[policy]
        if policy == "aegiskv":
            token_totals = {
                "l1_backup": _integer(summary, "l1_reuse_tokens"),
                "l2_backup": _integer(summary, "l2_reuse_tokens"),
                "recompute": _integer(summary, "recompute_tokens"),
            }
        else:
            token_totals = {
                "recompute": _integer(summary, "recompute_tokens")
            }
        rows = sorted(samples[policy], key=lambda row: row["wct_s"])
        events = _build_events(
            policy=policy,
            retry_count=_integer(summary, "retry_count"),
            token_totals=token_totals,
            workflow_ids=[row["id"] for row in rows],
            duration_s=duration_s,
            segments=segments,
        )
        policies.append(
            {
                "key": policy,
                "label": label,
                "color": color,
                "workflows": rows,
                "events": events,
                "summary": {
                    "workflow_count": _integer(summary, "workflow_count"),
                    "mean_wct_s": _floating(summary, "mean_wct_s"),
                    "p90_wct_s": _floating(summary, "p90_wct_s"),
                    "p99_wct_s": _floating(summary, "p99_wct_s"),
                    "max_wct_s": _floating(summary, "max_wct_s"),
                    "retry_count": _integer(summary, "retry_count"),
                    "retry_tokens_total": _integer(summary, "retry_tokens_total"),
                    "token_totals": token_totals,
                },
            }
        )
    return {
        "status": "simulated_replay_from_aggregate_results",
        "model": "Qwen/Qwen3-8B",
        "context_tokens": 131_072,
        "dp_size": TOTAL_GPUS,
        "workflow_count": 160,
        "trace_hours": TRACE_HOURS,
        "average_available_gpus": REPORTED_AVERAGE_AVAILABLE,
        "duration_s": duration_s,
        "availability_segments": segments,
        "policies": policies,
        "sources": {
            "projected_csv": str(csv_path.resolve()),
            "projection_assumptions": str(assumptions_path.resolve()),
            "availability": "digitized and mean-calibrated from supplied PNG",
        },
        "limitations": [
            "The 160 WCT samples and aggregate token totals come from the DP=64 projection.",
            "Individual retry timestamps, ranks, and workflow assignments are deterministically synthesized for animation.",
            "The original 64-GPU availability CSV was not present; the schedule is digitized from the supplied plot and calibrated to mean 58.07.",
            "This HTML is a simulated visualization, not a persisted event replay from a DP=64 SGLSim run.",
        ],
        "projection_assumptions": assumptions.get("model", {}),
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AegisKV DP=64 simulated replay</title>
  <style>
    :root { --bg:#06101e; --panel:#0d1c2f; --line:#263b58; --text:#edf5ff; --muted:#8fa4bf; --cyan:#18a8e0; --gray:#8b98aa; --green:#16c784; --blue:#2d7ff9; --red:#ff5c6c; --amber:#ffb84d; }
    *{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 12% 0,#11335b55,transparent 30%),radial-gradient(circle at 88% 8%,#0c4a3b44,transparent 28%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
    .shell{max-width:1740px;margin:auto;padding:22px 26px 32px} header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start} h1{margin:0;font-size:clamp(25px,2.4vw,40px);letter-spacing:-.035em}.subtitle{margin-top:7px;color:var(--muted)}.badges{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:7px}.badge{padding:7px 10px;border:1px solid var(--line);border-radius:999px;background:#10243d;color:#c9d9ed;font-size:12px}.badge.sim{border-color:#24775d;background:#0b3329;color:#82efc0}
    .controls{display:grid;grid-template-columns:auto auto minmax(260px,1fr) auto;gap:10px;align-items:center;margin:18px 0 12px;padding:11px;border:1px solid var(--line);border-radius:13px;background:#0a192bd9}button,select{border:1px solid #36516f;border-radius:8px;color:var(--text);background:#142c48;padding:9px 13px;font-weight:800;cursor:pointer}input[type=range]{width:100%;accent-color:var(--cyan)}.clock{text-align:right;min-width:210px;font-variant-numeric:tabular-nums;font-weight:850}.clock span{display:block;color:var(--muted);font-size:11px;font-weight:600}
    .availability{padding:13px 15px 9px;border:1px solid var(--line);border-radius:14px;background:#0b192a;margin-bottom:14px}.availability-head{display:flex;justify-content:space-between;color:#bdcde1;font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.05em}.availability-now{color:#e9f3ff;font-variant-numeric:tabular-nums}canvas{display:block;width:100%;height:118px;margin-top:7px}
    .compare{display:grid;grid-template-columns:1fr 1fr;gap:15px}.policy{border:1px solid var(--line);border-radius:15px;background:linear-gradient(180deg,#10243bea,#091725ea);overflow:hidden;box-shadow:0 18px 50px #0005}.policy-head{display:flex;justify-content:space-between;align-items:center;padding:15px 17px;border-bottom:1px solid var(--line)}.name{font-size:24px;font-weight:900}.aegiskv .name{color:#62d2f7}.no_backup .name{color:#c2ccd8}.final{color:var(--muted);font-size:11px;line-height:1.5;text-align:right}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;padding:11px 13px}.metric{padding:9px;border:1px solid #203854;border-radius:9px;background:#081827}.metric label{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em}.metric b{display:block;margin-top:4px;font-size:18px;font-variant-numeric:tabular-nums;white-space:nowrap}
    .section{padding:0 13px 12px}.section-title{margin:2px 0 7px;color:#b9c9df;font-size:11px;font-weight:850;text-transform:uppercase;letter-spacing:.07em}.ranks{display:grid;grid-template-columns:repeat(16,1fr);gap:4px}.rank{position:relative;padding:6px 1px;border:1px solid #294461;border-radius:5px;text-align:center;background:#102740;color:#a9bdd4;font-size:9px;transition:.16s}.rank.failed{background:#762b3a;border-color:var(--red);color:white;box-shadow:0 0 10px #ff5c6c55}.rank.pulse:after{content:"";position:absolute;inset:-2px;border:2px solid var(--amber);border-radius:7px;animation:pulse .6s infinite alternate}@keyframes pulse{from{opacity:.2}to{opacity:1}}
    .workflow-grid{display:grid;grid-template-columns:repeat(20,1fr);gap:4px}.wf{aspect-ratio:1;border-radius:3px;background:#1c314b;border:1px solid #29415d;transition:.12s}.aegiskv .wf.done{background:var(--cyan);border-color:#63d8ff;box-shadow:0 0 6px #18a8e077}.no_backup .wf.done{background:#9ca8b8;border-color:#d0d6de}.event-log{height:126px;overflow:hidden;border:1px solid #203854;border-radius:9px;background:#061522}.event{display:grid;grid-template-columns:50px 68px 1fr auto;gap:6px;padding:6px 8px;border-bottom:1px solid #142a40;font-size:10px}.event .t{color:#8197b0}.event .wf{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#b6cae3}.action{font-weight:900}.l1_backup{color:var(--green)}.l2_backup{color:#64a5ff}.recompute{color:#c0c8d4}.tokens{font-variant-numeric:tabular-nums}.empty{padding:40px;color:#607790;text-align:center;font-size:11px}.notice{margin-top:13px;padding:10px 12px;border:1px solid #5e4a25;border-radius:9px;background:#2c2211;color:#e8c77e;font-size:11px;line-height:1.45}
    @media(max-width:1120px){.compare{grid-template-columns:1fr}.controls{grid-template-columns:auto auto 1fr}.clock{grid-column:1/-1;text-align:left}}
  </style>
</head>
<body><div class="shell">
  <header><div><h1>AegisKV vs No Backup</h1><div class="subtitle">Large-scale simulated workflow and failure replay</div></div><div class="badges" id="badges"></div></header>
  <div class="controls"><button id="play">▶ Play</button><button id="reset">↺ Reset</button><input id="scrub" type="range" min="0" max="1000" value="0"><div style="display:flex;align-items:center;gap:8px"><select id="speed"><option value="0.5">0.5×</option><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="8">8×</option></select><div class="clock" id="clock">0.00 h<span>0 simulated seconds</span></div></div></div>
  <section class="availability"><div class="availability-head"><span>GPU availability over trace time</span><span class="availability-now" id="avail-now">64 / 64 available</span></div><canvas id="availability"></canvas></section>
  <main class="compare" id="compare"></main>
  <div class="notice">SIMULATED: WCT samples and aggregate token totals come from the DP=64 projection. Retry timestamps, rank assignments, and the digitized availability schedule are deterministic animation inputs, not persisted DP=64 runtime events.</div>
</div>
<script>
const DATA=__DATA__;let progress=0,playing=false,last=null;const animationSeconds=60;const $=id=>document.getElementById(id);const compact=new Intl.NumberFormat('en-US',{notation:'compact',maximumFractionDigits:2});
function segmentAt(h){return DATA.availability_segments.find(s=>s.start_h<=h&&h<s.end_h)||DATA.availability_segments.at(-1)}
function policyHTML(p){const s=p.summary;return `<section class="policy ${p.key}" id="p-${p.key}"><div class="policy-head"><div class="name">${p.label}</div><div class="final">mean ${(s.mean_wct_s/60).toFixed(1)} min<br>p99 ${(s.p99_wct_s/60).toFixed(1)} min</div></div><div class="metrics"><div class="metric"><label>Completed</label><b data-m="done">0/${s.workflow_count}</b></div><div class="metric"><label>Retries</label><b data-m="retry">0/${s.retry_count}</b></div><div class="metric"><label>Reused tokens</label><b data-m="reuse">0</b></div><div class="metric"><label>Recompute</label><b data-m="redo">0</b></div></div><div class="section"><div class="section-title">64 DP ranks</div><div class="ranks">${Array.from({length:64},(_,i)=>`<div class="rank" data-rank="${i}">${i}</div>`).join('')}</div></div><div class="section"><div class="section-title">160 workflow cohort</div><div class="workflow-grid">${p.workflows.map((_,i)=>`<div class="wf" data-wf="${i}"></div>`).join('')}</div></div><div class="section"><div class="section-title">Simulated retry stream</div><div class="event-log" data-log><div class="empty">Waiting for failure exposure …</div></div></div></section>`}
function build(){ $('badges').innerHTML=`<span class="badge sim">SIMULATED</span><span class="badge">Qwen3-8B</span><span class="badge">context ≤ 128K</span><span class="badge">DP=64</span><span class="badge">160 workflows</span><span class="badge">avg. available 58.07</span>`;$('compare').innerHTML=DATA.policies.map(policyHTML).join('');drawAvailability(0);render(0)}
function drawAvailability(h){const c=$('availability'),dpr=devicePixelRatio||1,w=c.clientWidth,hpx=c.clientHeight;c.width=w*dpr;c.height=hpx*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,hpx);x.strokeStyle='#263b58';x.lineWidth=1;[16,32,48,64].forEach(v=>{const y=hpx-8-(v/64)*(hpx-16);x.beginPath();x.moveTo(0,y);x.lineTo(w,y);x.stroke()});x.strokeStyle='#18a8e0';x.lineWidth=2.5;x.beginPath();DATA.availability_segments.forEach((s,i)=>{const x0=w*s.start_h/DATA.trace_hours,x1=w*s.end_h/DATA.trace_hours,y=hpx-8-(s.available/64)*(hpx-16);if(i===0)x.moveTo(x0,y);else x.lineTo(x0,y);x.lineTo(x1,y)});x.stroke();const avgY=hpx-8-(DATA.average_available_gpus/64)*(hpx-16);x.setLineDash([5,4]);x.strokeStyle='#82bdf0';x.lineWidth=1;x.beginPath();x.moveTo(0,avgY);x.lineTo(w,avgY);x.stroke();x.setLineDash([]);const playX=w*h/DATA.trace_hours;x.strokeStyle='white';x.lineWidth=2;x.beginPath();x.moveTo(playX,0);x.lineTo(playX,hpx);x.stroke()}
function renderPolicy(p,simS,traceH,seg){const root=$(`p-${p.key}`),events=p.events.filter(e=>e.t_s<=simS),done=p.workflows.filter(w=>w.wct_s<=simS).length,reuse=events.filter(e=>e.action!=='recompute').reduce((a,e)=>a+e.tokens,0),redo=events.filter(e=>e.action==='recompute').reduce((a,e)=>a+e.tokens,0);root.querySelector('[data-m=done]').textContent=`${done}/${p.summary.workflow_count}`;root.querySelector('[data-m=retry]').textContent=`${events.length}/${p.summary.retry_count}`;root.querySelector('[data-m=reuse]').textContent=compact.format(reuse);root.querySelector('[data-m=redo]').textContent=compact.format(redo);const failed=new Set(seg.failed_ranks),recent=events.filter(e=>simS-e.t_s<DATA.duration_s/120);root.querySelectorAll('.rank').forEach(n=>{const r=+n.dataset.rank;n.classList.toggle('failed',failed.has(r));n.classList.toggle('pulse',recent.some(e=>e.retry_rank===r))});root.querySelectorAll('.wf').forEach((n,i)=>n.classList.toggle('done',p.workflows[i].wct_s<=simS));const rows=events.slice(-6).reverse();root.querySelector('[data-log]').innerHTML=rows.length?rows.map(e=>`<div class="event"><span class="t">${e.trace_h.toFixed(2)}h</span><span class="wf">${e.workflow_id.slice(-8)}</span><span>DP ${e.failed_rank} → ${e.retry_rank} · <b class="action ${e.action}">${e.action==='l1_backup'?'L1 reuse':e.action==='l2_backup'?'L2 reuse':'Recompute'}</b></span><span class="tokens">${compact.format(e.tokens)}</span></div>`).join(''):'<div class="empty">Waiting for failure exposure …</div>'}
function render(value){progress=Math.max(0,Math.min(1,value));const traceH=progress*DATA.trace_hours,simS=progress*DATA.duration_s,seg=segmentAt(traceH);$('clock').innerHTML=`${traceH.toFixed(2)} h<span>${Math.round(simS).toLocaleString()} simulated seconds</span>`;$('avail-now').textContent=`${seg.available} / 64 available · ${64-seg.available} unavailable`;$('scrub').value=Math.round(progress*1000);DATA.policies.forEach(p=>renderPolicy(p,simS,traceH,seg));drawAvailability(traceH);if(progress>=1&&playing)toggle(false)}
function toggle(force){playing=force===undefined?!playing:force;$('play').textContent=playing?'❚❚ Pause':'▶ Play';last=null;if(playing)requestAnimationFrame(tick)}function tick(ts){if(!playing)return;if(last===null)last=ts;const delta=(ts-last)/1000;last=ts;render(progress+delta*(+$('speed').value)/animationSeconds);if(playing)requestAnimationFrame(tick)}
$('play').onclick=()=>toggle();$('reset').onclick=()=>{toggle(false);render(0)};$('scrub').oninput=e=>{toggle(false);render(+e.target.value/1000)};window.onresize=()=>drawAvailability(progress*DATA.trace_hours);document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();toggle()}});build();
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the DP=64 / 128K aggregate simulation as standalone HTML."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--assumptions", type=Path, default=DEFAULT_ASSUMPTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=None)
    args = parser.parse_args()

    data = _build_data(args.csv, args.assumptions)
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML_TEMPLATE.replace("__DATA__", embedded), encoding="utf-8")
    data_output = args.data_output or args.output.with_suffix(".simulation.json")
    data_output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {data_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
