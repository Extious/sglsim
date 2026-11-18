#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = (
    Path(__file__).resolve().parent
    / "results"
    / "wct_cdf_calibrated_matched"
    / "20260722-144320"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


LOG_ACK_RE = re.compile(
    r"AegisKV ACK rid=(?P<rid>\S+) source=(?P<source>\d+) "
    r"target=(?P<target>\d+) action=(?P<action>\S+) "
    r"L1=(?P<l1>\d+) L2=(?P<l2>\d+) R=(?P<recompute>\d+)"
)


def _backup_acks(server_log: Path | None) -> dict[str, dict[str, Any]]:
    if server_log is None or not server_log.is_file():
        return {}
    acks: dict[str, dict[str, Any]] = {}
    for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_ACK_RE.search(line)
        if match is None:
            continue
        acks[match.group("rid")] = {
            "source_rank": int(match.group("source")),
            "target_rank": int(match.group("target")),
            "action": match.group("action"),
            "l1_tokens": int(match.group("l1")),
            "l2_tokens": int(match.group("l2")),
            "recompute_tokens": int(match.group("recompute")),
        }
    return acks


def _policy_payload(path: Path, *, server_log: Path | None = None) -> dict[str, Any]:
    source = _read_json(path)
    backup_acks = _backup_acks(server_log)
    workflows: list[dict[str, Any]] = []
    retries: list[dict[str, Any]] = []
    for workflow in source.get("workflows", []):
        workflow_id = str(workflow.get("workflow_id") or "")
        workflows.append(
            {
                "id": workflow_id,
                "short_id": workflow_id[:8],
                "wct_s": float(workflow.get("wct_s", 0.0)),
                "request_count": int(workflow.get("request_count", 0)),
                "retry_count": int(workflow.get("retry_count", 0)),
            }
        )
        for event in workflow.get("retry_events", []):
            backup_key = str(event.get("backup_key") or "")
            ack = backup_acks.get(backup_key, {})
            retries.append(
                {
                    "t": float(event.get("elapsed_s", 0.0)),
                    "workflow_id": workflow_id,
                    "workflow_short_id": workflow_id[:8],
                    "rid": str(event.get("rid") or ""),
                    "backup_key": backup_key,
                    "failed_rank": int(event.get("failed_dp_rank", -1)),
                    "retry_rank": int(event.get("retry_dp_rank", -1)),
                    "action": str(event.get("recovery_action") or "recompute"),
                    "restored_tokens": int(event.get("restored_tokens", 0)),
                    "recompute_tokens": int(event.get("recompute_tokens", 0)),
                    "latency_s": float(event.get("latency_s", 0.0)),
                    "planned_target_rank": int(ack.get("target_rank", -1)),
                    "planned_action": str(ack.get("action") or ""),
                    "planned_l1_tokens": int(ack.get("l1_tokens", 0)),
                    "planned_l2_tokens": int(ack.get("l2_tokens", 0)),
                    "planned_recompute_tokens": int(
                        ack.get("recompute_tokens", 0)
                    ),
                }
            )
    workflows.sort(key=lambda row: (row["wct_s"], row["id"]))
    retries.sort(key=lambda row: (row["t"], row["workflow_id"], row["rid"]))

    failures = []
    for event in source.get("failure_events", []):
        failures.append(
            {
                "id": str(event.get("event_id") or ""),
                "rank": int(event.get("dp_rank", -1)),
                "planned_start_s": float(event.get("offset_s", 0.0)),
                "start_s": float(
                    event.get("injected_offset_s", event.get("offset_s", 0.0))
                ),
                "end_s": float(
                    event.get(
                        "recovered_offset_s",
                        float(event.get("offset_s", 0.0))
                        + float(event.get("duration_s", 0.0)),
                    )
                ),
                "outage_s": float(event.get("actual_outage_s", 0.0)),
            }
        )

    retry_summary = source.get("retry_summary", {})
    failure_manifest = source.get("failure_manifest", {})
    matched_victim_count = (
        int(failure_manifest.get("victim_count", 0))
        or sum(
            len(event.get("victims", []))
            for event in failure_manifest.get("events", [])
            if isinstance(event, dict) and isinstance(event.get("victims", []), list)
        )
        if isinstance(failure_manifest, dict)
        else 0
    )
    return {
        "key": str(source.get("policy") or path.stem),
        "label": str(source.get("policy_label") or path.stem),
        "model": str(source.get("model") or ""),
        "dp_size": int(source.get("dp_size", 0)),
        "workflow_count": int(source.get("workflow_count", len(workflows))),
        "workflow_concurrency": int(source.get("workflow_concurrency", 0)),
        "request_count": int(source.get("request_count", 0)),
        "context_length": int(source.get("replay", {}).get("context_length", 0)),
        "input_scale": float(source.get("replay", {}).get("input_scale", 1.0)),
        "wct_summary": source.get("wct_summary_s", {}),
        "retry_summary": retry_summary,
        "backup_summary": source.get("backup_decision_summary", {}),
        "workflows": workflows,
        "retries": retries,
        "failures": failures,
        "matched_victim_count": matched_victim_count,
        "source": str(path.resolve()),
    }


def _build_data(results_dir: Path) -> dict[str, Any]:
    policies = [
        _policy_payload(
            results_dir / "aegiskv.json",
            server_log=results_dir / "server-aegiskv.log",
        ),
        _policy_payload(results_dir / "no_backup.json"),
    ]
    duration_s = max(
        max(float(policy["wct_summary"].get("max", 0.0)), 0.0)
        for policy in policies
    )
    matched_victim_count = min(
        int(policy.get("matched_victim_count", 0)) for policy in policies
    )
    failure_note = (
        f"Matched failure manifest with {matched_victim_count} frozen victims "
        "shared by both policies."
        if matched_victim_count > 0
        else "Fixed time/rank manifest; failure victims were not strictly matched across policies."
    )
    return {
        "title": "AegisKV vs No Backup — real SGLSim result replay",
        "run_id": results_dir.name,
        "results_dir": str(results_dir.resolve()),
        "duration_s": duration_s,
        "policies": policies,
        "evidence_note": (
            "Real persisted WCT/failure/retry events. Workflow progress is aligned "
            "at t=0 because the run used concurrency=20; ordinary request-level "
            f"timestamps were not persisted. {failure_note}"
        ),
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AegisKV real-run replay</title>
  <style>
    :root {
      --bg: #07111f;
      --panel: #0d1b2d;
      --panel-2: #10233a;
      --line: #263a55;
      --text: #edf4ff;
      --muted: #91a4be;
      --aegis: #18a8e0;
      --nobackup: #8b98aa;
      --green: #16c784;
      --blue: #2d7ff9;
      --red: #ff5c6c;
      --amber: #ffb84d;
      --gray: #738095;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 0%, #12305a55, transparent 30%),
        radial-gradient(circle at 90% 10%, #11463844, transparent 28%),
        var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell { max-width: 1680px; margin: 0 auto; padding: 22px 26px 34px; }
    header { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; }
    h1 { margin: 0; font-size: clamp(24px, 2.3vw, 38px); letter-spacing: -0.03em; }
    .subtitle { margin-top: 8px; color: var(--muted); font-size: 14px; }
    .badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
    .badge { border: 1px solid var(--line); background: #10223a; padding: 7px 10px; border-radius: 999px; font-size: 12px; color: #cfe0f5; }
    .badge.live { color: #8af0c4; border-color: #1c7257; background: #0c3229; }
    .badge.gain { color: #8af0c4; border-color: #1c7257; background: #0c3229; }
    .badge.loss { color: #ffb4bc; border-color: #7a3340; background: #3a1720; }
    .controls {
      margin: 20px 0 14px;
      display: grid;
      grid-template-columns: auto auto minmax(260px, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #0a1829d9;
      box-shadow: 0 14px 40px #0005;
    }
    button, select {
      border: 1px solid #36506f;
      border-radius: 9px;
      color: var(--text);
      background: #142a45;
      padding: 9px 13px;
      font-weight: 700;
      cursor: pointer;
    }
    input[type="range"] { width: 100%; accent-color: var(--aegis); }
    .clock { font-variant-numeric: tabular-nums; min-width: 88px; text-align: right; font-size: 20px; font-weight: 800; color: #d8e8ff; }
    .failure-track {
      position: relative;
      height: 34px;
      margin: 0 0 16px;
      border: 1px solid var(--line);
      border-radius: 9px;
      overflow: hidden;
      background: linear-gradient(90deg, #10243c, #0d1b2d);
    }
    .failure-window { position: absolute; top: 0; bottom: 0; background: #ff5c6c33; border-left: 1px solid var(--red); border-right: 1px solid var(--red); }
    .failure-window span { display: block; padding: 7px 8px; color: #ffb4bc; font-size: 11px; white-space: nowrap; }
    .playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: white; box-shadow: 0 0 14px #fff; z-index: 4; }
    .compare { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .policy {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, #10233aea, #0b1727ea);
      overflow: hidden;
      box-shadow: 0 18px 50px #0005;
    }
    .policy-head { padding: 16px 18px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); }
    .policy-name { font-size: 24px; font-weight: 850; }
    .policy.aegiskv .policy-name { color: #62cef5; }
    .policy.no_backup .policy-name { color: #c4ccd8; }
    .final-wct { color: var(--muted); font-size: 12px; text-align: right; line-height: 1.5; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding: 12px 14px; }
    .metric { min-width: 0; padding: 10px; border: 1px solid #203753; border-radius: 10px; background: #0b1a2c; }
    .metric .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
    .metric .value { margin-top: 4px; font-size: 19px; font-weight: 850; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .section { padding: 0 14px 13px; }
    .section-title { margin: 2px 0 8px; color: #b9c9df; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .07em; }
    .section-title.event-title { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .event-status { color: #7891ad; font-size: 10px; font-variant-numeric: tabular-nums; letter-spacing: 0; }
    .ranks { display: grid; grid-template-columns: repeat(8, 1fr); gap: 6px; }
    .rank { position: relative; padding: 9px 4px; border: 1px solid #2e4663; border-radius: 8px; text-align: center; background: #12263f; color: #bcd0e8; font-size: 11px; transition: .18s ease; }
    .rank.failed { color: white; background: #752938; border-color: var(--red); box-shadow: 0 0 18px #ff5c6c66; transform: translateY(-2px); }
    .rank.pulse::after { content: ""; position: absolute; inset: -3px; border: 2px solid var(--amber); border-radius: 10px; animation: pulse .65s infinite alternate; }
    @keyframes pulse { from { opacity: .25; } to { opacity: 1; } }
    .workflows { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 9px; }
    .workflow { display: grid; grid-template-columns: 67px 1fr 42px; gap: 7px; align-items: center; font-size: 10px; color: #aebed2; }
    .wf-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; overflow: hidden; }
    .wf-track { position: relative; height: 8px; border-radius: 99px; overflow: hidden; background: #1b304a; }
    .wf-fill { height: 100%; width: 0; border-radius: inherit; transition: width .08s linear; }
    .aegiskv .wf-fill { background: linear-gradient(90deg, #0a79ac, #1bd0f2); }
    .no_backup .wf-fill { background: linear-gradient(90deg, #5d6878, #a7b1bf); }
    .workflow.done .wf-track { box-shadow: 0 0 8px #16c78466; }
    .wf-time { text-align: right; font-variant-numeric: tabular-nums; }
    .event-log { height: 154px; overflow-x: hidden; overflow-y: auto; scroll-behavior: smooth; border: 1px solid #203753; border-radius: 10px; background: #071421; }
    .event-log::-webkit-scrollbar { width: 7px; }
    .event-log::-webkit-scrollbar-thumb { border-radius: 99px; background: #29435f; }
    .event { display: grid; grid-template-columns: 46px 70px 1fr auto; gap: 7px; padding: 6px 8px; border-bottom: 1px solid #14283e; font-size: 11px; }
    .event.fresh { animation: event-arrive .8s ease; }
    @keyframes event-arrive {
      0% { opacity: 0; transform: translateY(5px); background: #ffb84d40; }
      45% { opacity: 1; background: #ffb84d25; }
      100% { background: transparent; }
    }
    .event .t { color: #7f94ae; font-variant-numeric: tabular-nums; }
    .event .wf { color: #adc6e4; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .action { font-weight: 850; }
    .action.l1_backup { color: var(--green); }
    .action.l2_backup { color: #60a0ff; }
    .action.recompute { color: #b4bdca; }
    .tokens { color: #c9d7e8; font-variant-numeric: tabular-nums; }
    .empty { padding: 20px; color: #60758e; text-align: center; font-size: 12px; }
    footer { margin-top: 14px; color: #778ca6; font-size: 11px; line-height: 1.5; }
    @media (max-width: 1100px) {
      .compare { grid-template-columns: 1fr; }
      .controls { grid-template-columns: auto auto 1fr; }
      .clock { grid-column: 1 / -1; text-align: left; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>AegisKV vs No Backup</h1>
        <div class="subtitle" id="subtitle">Real SGLSim online-run event replay</div>
      </div>
      <div class="badges" id="badges"></div>
    </header>

    <div class="controls">
      <button id="play">▶ Play</button>
      <button id="reset">↺ Reset</button>
      <input id="scrubber" type="range" min="0" max="1000" value="0">
      <div style="display:flex; gap:8px; align-items:center; justify-content:flex-end">
        <select id="speed" aria-label="Playback speed">
          <option value="0.5">0.5×</option>
          <option value="1" selected>1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
          <option value="8">8×</option>
        </select>
        <div class="clock" id="clock">0.00 s</div>
      </div>
    </div>

    <div class="failure-track" id="failure-track"><div class="playhead" id="playhead"></div></div>
    <main class="compare" id="compare"></main>
    <footer id="evidence"></footer>
  </div>

  <script>
    const DATA = __DATA__;
    const duration = DATA.duration_s;
    let t = 0;
    let playing = false;
    let previousFrame = null;

    const fmt = new Intl.NumberFormat("en-US");
    const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 });
    const byId = id => document.getElementById(id);

    function actionLabel(action) {
      return ({l1_backup: "L1 reuse", l2_backup: "L2 reuse", l1_l2_backup: "L1+L2 reuse", recompute: "Recompute"})[action] || action;
    }

    function policyMarkup(policy) {
      const s = policy.wct_summary;
      return `
        <section class="policy ${policy.key}" id="policy-${policy.key}">
          <div class="policy-head">
            <div class="policy-name">${policy.label}</div>
            <div class="final-wct">final mean ${Number(s.mean).toFixed(2)} s<br>p99 ${Number(s.p99).toFixed(2)} s</div>
          </div>
          <div class="metrics">
            <div class="metric"><div class="label">Completed</div><div class="value" data-metric="completed">0/${policy.workflow_count}</div></div>
            <div class="metric"><div class="label">Retries</div><div class="value" data-metric="retries">0/${policy.retry_summary.count}</div></div>
            <div class="metric"><div class="label">Reused tokens</div><div class="value" data-metric="reused">0</div></div>
            <div class="metric"><div class="label">Recompute</div><div class="value" data-metric="recompute">0</div></div>
          </div>
          <div class="section"><div class="section-title">DP-rank health</div><div class="ranks">
            ${Array.from({length: policy.dp_size}, (_, rank) => `<div class="rank" data-rank="${rank}">DP ${rank}</div>`).join("")}
          </div></div>
          <div class="section"><div class="section-title">${policy.workflow_count} concurrent workflow completion tracks</div><div class="workflows">
            ${policy.workflows.map((wf, index) => `
              <div class="workflow" data-workflow="${index}">
                <span class="wf-id">${wf.short_id}</span>
                <div class="wf-track"><div class="wf-fill"></div></div>
                <span class="wf-time">${wf.wct_s.toFixed(1)}s</span>
              </div>`).join("")}
          </div></div>
          <div class="section">
            <div class="section-title event-title">Retry event stream <span class="event-status" data-event-status>0/${policy.retry_summary.count}</span></div>
            <div class="event-log" data-event-log data-rendered-count="0"><div class="empty">Waiting for retry events …</div></div>
          </div>
        </section>`;
    }

    function build() {
      const p = DATA.policies[0];
      const baseline = DATA.policies[1];
      const meanGain = 100 * (Number(baseline.wct_summary.mean) - Number(p.wct_summary.mean)) / Number(baseline.wct_summary.mean);
      const p99Gain = 100 * (Number(baseline.wct_summary.p99) - Number(p.wct_summary.p99)) / Number(baseline.wct_summary.p99);
      const deltaBadge = (label, value) => `<span class="badge ${value >= 0 ? "gain" : "loss"}">${label} ${value >= 0 ? "−" : "+"}${Math.abs(value).toFixed(1)}%</span>`;
      byId("subtitle").textContent = `Real SGLSim online-run event replay · result ${DATA.run_id}`;
      byId("badges").innerHTML = `
        <span class="badge live">REAL RUN</span><span class="badge">Qwen3-8B</span>
        <span class="badge">DP=${p.dp_size}</span><span class="badge">${p.workflow_count} workflows</span>
        <span class="badge">${p.request_count.toLocaleString()} requests</span><span class="badge">context ≤ ${p.context_length.toLocaleString()}</span>
        ${deltaBadge("Mean WCT", meanGain)}${deltaBadge("p99", p99Gain)}`;
      byId("compare").innerHTML = DATA.policies.map(policyMarkup).join("");
      byId("evidence").textContent = DATA.evidence_note;

      const track = byId("failure-track");
      DATA.policies[0].failures.forEach(failure => {
        const node = document.createElement("div");
        node.className = "failure-window";
        node.style.left = `${100 * failure.start_s / duration}%`;
        node.style.width = `${100 * (failure.end_s - failure.start_s) / duration}%`;
        node.innerHTML = `<span>DP ${failure.rank} failure · ${failure.start_s.toFixed(1)}–${failure.end_s.toFixed(1)}s</span>`;
        track.insertBefore(node, byId("playhead"));
      });
      render(0);
    }

    function retryTokenLabel(event) {
      const parts = [];
      if (event.restored_tokens > 0) parts.push(`reuse ${fmt.format(event.restored_tokens)}`);
      if (event.recompute_tokens > 0) parts.push(`redo ${fmt.format(event.recompute_tokens)}`);
      const tokens = parts.join(" + ") || "0 tokens";
      if (event.planned_action === "recompute" && event.restored_tokens === 0) {
        return `${tokens} · ACK no copy`;
      }
      if (event.restored_tokens > 0 && event.recompute_tokens > 0) {
        return `${tokens} · partial ACK`;
      }
      return tokens;
    }

    function retryEventMarkup(event, fresh) {
      return `
        <div class="event${fresh ? " fresh" : ""}" title="backup key ${event.backup_key || "n/a"} · Algorithm 2 ACK target DP ${event.planned_target_rank} · L1 ${event.planned_l1_tokens} / L2 ${event.planned_l2_tokens} / redo ${event.planned_recompute_tokens}">
          <span class="t">${event.t.toFixed(1)}s</span>
          <span class="wf">${event.workflow_short_id}</span>
          <span><b>DP ${event.failed_rank} → ${event.retry_rank}</b> · <span class="action ${event.action}">${actionLabel(event.action)}</span></span>
          <span class="tokens">${retryTokenLabel(event)}</span>
        </div>`;
    }

    function renderRetryStream(policy, now, root, visibleRetries) {
      const log = root.querySelector("[data-event-log]");
      const status = root.querySelector("[data-event-status]");
      const previousCount = Number(log.dataset.renderedCount || 0);
      const next = policy.retries[visibleRetries.length];
      status.textContent = next
        ? `${visibleRetries.length}/${policy.retry_summary.count} · next ${next.t.toFixed(1)}s`
        : `${visibleRetries.length}/${policy.retry_summary.count} · complete`;

      if (visibleRetries.length < previousCount) {
        log.innerHTML = visibleRetries.length
          ? visibleRetries.map(event => retryEventMarkup(event, false)).join("")
          : `<div class="empty">${next ? `Next retry at ${next.t.toFixed(1)}s` : "No retry events"}</div>`;
      } else if (visibleRetries.length > previousCount) {
        if (previousCount === 0) log.innerHTML = "";
        visibleRetries.slice(previousCount).forEach(event => {
          log.insertAdjacentHTML("beforeend", retryEventMarkup(event, true));
        });
        log.scrollTop = log.scrollHeight;
      } else if (visibleRetries.length === 0) {
        log.innerHTML = `<div class="empty">${next ? `Next retry at ${next.t.toFixed(1)}s` : "No retry events"}</div>`;
      }
      log.dataset.renderedCount = String(visibleRetries.length);
    }

    function renderPolicy(policy, now) {
      const root = byId(`policy-${policy.key}`);
      const completed = policy.workflows.filter(wf => wf.wct_s <= now).length;
      const visibleRetries = policy.retries.filter(event => event.t <= now);
      const reused = visibleRetries.reduce((sum, event) => sum + event.restored_tokens, 0);
      const recompute = visibleRetries.reduce((sum, event) => sum + event.recompute_tokens, 0);
      root.querySelector('[data-metric="completed"]').textContent = `${completed}/${policy.workflow_count}`;
      root.querySelector('[data-metric="retries"]').textContent = `${visibleRetries.length}/${policy.retry_summary.count}`;
      root.querySelector('[data-metric="reused"]').textContent = compact.format(reused);
      root.querySelector('[data-metric="recompute"]').textContent = compact.format(recompute);
      root.querySelectorAll(".rank").forEach(node => {
        const rank = Number(node.dataset.rank);
        const failed = policy.failures.some(event => event.rank === rank && event.start_s <= now && now < event.end_s);
        const pulsing = visibleRetries.some(event => event.retry_rank === rank && now - event.t >= 0 && now - event.t < 1.1);
        node.classList.toggle("failed", failed);
        node.classList.toggle("pulse", pulsing);
        node.textContent = failed ? `DP ${rank} ✕` : `DP ${rank}`;
      });

      root.querySelectorAll(".workflow").forEach((node, index) => {
        const wf = policy.workflows[index];
        const fraction = Math.max(0, Math.min(1, now / wf.wct_s));
        node.querySelector(".wf-fill").style.width = `${100 * fraction}%`;
        node.classList.toggle("done", fraction >= 1);
      });

      renderRetryStream(policy, now, root, visibleRetries);
    }

    function render(now) {
      t = Math.max(0, Math.min(duration, now));
      byId("clock").textContent = `${t.toFixed(2)} s`;
      byId("scrubber").value = Math.round(1000 * t / duration);
      byId("playhead").style.left = `${100 * t / duration}%`;
      DATA.policies.forEach(policy => renderPolicy(policy, t));
      if (t >= duration && playing) togglePlay(false);
    }

    function togglePlay(force) {
      playing = force === undefined ? !playing : force;
      byId("play").textContent = playing ? "❚❚ Pause" : "▶ Play";
      previousFrame = null;
      if (playing) requestAnimationFrame(tick);
    }

    function tick(timestamp) {
      if (!playing) return;
      if (previousFrame === null) previousFrame = timestamp;
      const delta = (timestamp - previousFrame) / 1000;
      previousFrame = timestamp;
      const speed = Number(byId("speed").value);
      render(t + delta * speed);
      if (playing) requestAnimationFrame(tick);
    }

    byId("play").addEventListener("click", () => togglePlay());
    byId("reset").addEventListener("click", () => { togglePlay(false); render(0); });
    byId("scrubber").addEventListener("input", event => { togglePlay(false); render(duration * Number(event.target.value) / 1000); });
    document.addEventListener("keydown", event => { if (event.code === "Space") { event.preventDefault(); togglePlay(); } });
    build();
  </script>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a standalone HTML replay of a real WCT CDF result."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.results_dir / "aegiskv_vs_no_backup_replay.html"
    data = _build_data(args.results_dir)
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML_TEMPLATE.replace("__DATA__", embedded), encoding="utf-8")
    print(f"wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
