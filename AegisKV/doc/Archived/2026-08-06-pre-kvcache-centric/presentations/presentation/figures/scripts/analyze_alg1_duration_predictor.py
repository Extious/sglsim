#!/usr/bin/env python3
"""Build workflow-disjoint remaining-time evidence from the Real CC trace.

The predictor is evaluated at the end of each completed model turn.  It uses
only information observable at that point; future turn counts and future token
counts are deliberately excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import orjson
except ModuleNotFoundError:
    orjson = None


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TRACE = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "cc-traces-weka-062126"
    / "traces.jsonl"
)
DEFAULT_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_alg1_duration_predictor.csv"
)
DEFAULT_FEATURE_PNG = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_alg1_gbdt_feature_correlation.png"
)
DEFAULT_PROGRESS_PNG = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_alg1_gbdt_accuracy_progress.png"
)
DEFAULT_JSON = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "alg1_duration_predictor_summary.json"
)
FANDOL_FONT_DIR = (
    ROOT.parent
    / "miniforge3"
    / "envs"
    / "latex"
    / "tinytex"
    / "texmf-dist"
    / "fonts"
    / "opentype"
    / "public"
    / "fandol"
)

NUMERIC_FEATURES = [
    "turns_completed",
    "elapsed_s",
    "last_api_s",
    "mean_api_s",
    "last_think_s",
    "mean_think_s",
    "last_input_tokens",
    "last_output_tokens",
    "cum_input_tokens",
    "cum_output_tokens",
    "max_input_tokens",
]
CATEGORICAL_FEATURES = ["role", "agent_type", "model", "request_type"]
FEATURE_LABELS = {
    "turns_completed": "已完成轮次数",
    "elapsed_s": "已运行时间",
    "last_api_s": "最近一次 API 时长",
    "mean_api_s": "平均 API 时长",
    "last_think_s": "最近一次工具/思考时长",
    "mean_think_s": "平均工具/思考时长",
    "last_input_tokens": "最近输入词元数",
    "last_output_tokens": "最近输出词元数",
    "cum_input_tokens": "累计输入词元数",
    "cum_output_tokens": "累计输出词元数",
    "max_input_tokens": "历史最大输入词元数",
}
CATEGORICAL_FEATURE_LABELS = {
    "role": "角色",
    "agent_type": "智能体类型",
    "model": "模型",
    "request_type": "请求类型",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _request_end(request: dict[str, Any]) -> float:
    return _as_float(request.get("t")) + max(0.0, _as_float(request.get("api_time")))


def _trace_end(row: dict[str, Any]) -> float:
    end = 0.0
    for request in row.get("requests") or []:
        start = _as_float(request.get("t"))
        if request.get("type") == "subagent":
            end = max(end, start + _as_float(request.get("duration_ms")) / 1000.0)
            for inner in request.get("requests") or []:
                end = max(end, _request_end(inner))
        else:
            end = max(end, _request_end(request))
    return end


def _split(trace_id: str) -> str:
    bucket = int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _agent_rows(
    *,
    trace_id: str,
    agent_id: str,
    role: str,
    agent_type: str,
    requests: list[dict[str, Any]],
    agent_start: float,
    agent_end: float,
) -> Iterable[dict[str, Any]]:
    ordered = sorted(
        (request for request in requests if isinstance(request.get("t"), (int, float))),
        key=lambda request: _as_float(request.get("t")),
    )
    api_times: list[float] = []
    think_times: list[float] = []
    inputs: list[float] = []
    outputs: list[float] = []
    for turn_index, request in enumerate(ordered, start=1):
        api_times.append(max(0.0, _as_float(request.get("api_time"))))
        think_times.append(max(0.0, _as_float(request.get("think_time"))))
        inputs.append(max(0.0, _as_float(request.get("in"))))
        outputs.append(max(0.0, _as_float(request.get("out"))))
        prediction_time = _request_end(request)
        remaining_s = max(0.0, agent_end - prediction_time)
        if remaining_s <= 0.01:
            continue
        yield {
            "trace_id": trace_id,
            "agent_id": agent_id,
            "split": _split(trace_id),
            "role": role,
            "agent_type": agent_type or "unknown",
            "model": str(request.get("model") or "unknown"),
            "request_type": str(request.get("type") or "unknown"),
            "turns_completed": turn_index,
            "elapsed_s": max(0.0, prediction_time - agent_start),
            "last_api_s": api_times[-1],
            "mean_api_s": float(np.mean(api_times)),
            "last_think_s": think_times[-1],
            "mean_think_s": float(np.mean(think_times)),
            "last_input_tokens": inputs[-1],
            "last_output_tokens": outputs[-1],
            "cum_input_tokens": float(np.sum(inputs)),
            "cum_output_tokens": float(np.sum(outputs)),
            "max_input_tokens": float(np.max(inputs)),
            "remaining_s": remaining_s,
        }


def load_examples(path: Path) -> pd.DataFrame:
    examples: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = orjson.loads(line) if orjson is not None else json.loads(line)
            trace_id = str(row["id"])
            requests = list(row.get("requests") or [])
            main_phases: list[list[dict[str, Any]]] = []
            current_phase: list[dict[str, Any]] = []
            for request in requests:
                if request.get("type") == "subagent":
                    if current_phase:
                        main_phases.append(current_phase)
                        current_phase = []
                    continue
                current_phase.append(request)
            if current_phase:
                main_phases.append(current_phase)
            for phase_index, main_requests in enumerate(main_phases):
                main_start = min(_as_float(request.get("t")) for request in main_requests)
                main_end = max(_request_end(request) for request in main_requests)
                examples.extend(
                    _agent_rows(
                        trace_id=trace_id,
                        agent_id=f"{trace_id}:main_phase:{phase_index}",
                        role="main_phase",
                        agent_type="main_phase",
                        requests=main_requests,
                        agent_start=main_start,
                        agent_end=main_end,
                    )
                )
            for request in requests:
                if request.get("type") != "subagent":
                    continue
                inner = list(request.get("requests") or [])
                if not inner or request.get("status") != "completed":
                    continue
                start = _as_float(request.get("t"))
                end = max(
                    start + _as_float(request.get("duration_ms")) / 1000.0,
                    max((_request_end(item) for item in inner), default=start),
                )
                examples.extend(
                    _agent_rows(
                        trace_id=trace_id,
                        agent_id=str(request.get("agent_id") or "unknown"),
                        role="subagent",
                        agent_type=str(request.get("subagent_type") or "unknown"),
                        requests=inner,
                        agent_start=start,
                        agent_end=end,
                    )
                )
    frame = pd.DataFrame(examples)
    if frame.empty:
        raise RuntimeError("No predictor examples were derived from the trace")
    return frame


def _preprocessor() -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )


def _target_transform(regressor: Any) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def _gbdt(loss: str = "absolute_error", quantile: float | None = None) -> Pipeline:
    kwargs: dict[str, Any] = {
        "loss": loss,
        "learning_rate": 0.06,
        "max_iter": 220,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "random_state": 7,
    }
    if quantile is not None:
        kwargs["quantile"] = quantile
    return Pipeline(
        [
            ("features", _preprocessor()),
            (
                "model",
                _target_transform(HistGradientBoostingRegressor(**kwargs)),
            ),
        ]
    )


def _metric_rows(name: str, y: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    absolute = np.abs(y - pred)
    return [
        {"row_type": "model_metric", "model": name, "metric": "MAE", "value": mean_absolute_error(y, pred), "unit": "seconds"},
        {"row_type": "model_metric", "model": name, "metric": "MedianAE", "value": median_absolute_error(y, pred), "unit": "seconds"},
        {"row_type": "model_metric", "model": name, "metric": "P90AE", "value": float(np.quantile(absolute, 0.90)), "unit": "seconds"},
        {"row_type": "model_metric", "model": name, "metric": "WAPE", "value": float(np.sum(absolute) / np.sum(np.abs(y))), "unit": "ratio"},
        {"row_type": "model_metric", "model": name, "metric": "R2", "value": r2_score(y, pred), "unit": "score"},
    ]


def evaluate(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    x_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train["remaining_s"].to_numpy()
    x_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test["remaining_s"].to_numpy()

    rows: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        coefficient, p_value = spearmanr(
            frame[feature].to_numpy(),
            frame["remaining_s"].to_numpy(),
            nan_policy="omit",
        )
        rows.append(
            {
                "row_type": "feature_correlation",
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "metric": "spearman_rho",
                "value": float(coefficient),
                "p_value": float(p_value),
                "unit": "correlation",
            }
        )
    target = frame["remaining_s"].to_numpy()
    target_mean = float(np.mean(target))
    total_ss = float(np.sum((target - target_mean) ** 2))
    for feature in CATEGORICAL_FEATURES:
        grouped = frame.groupby(feature, dropna=False)["remaining_s"].agg(
            ["count", "mean"]
        )
        between_ss = float(
            np.sum(
                grouped["count"]
                * (grouped["mean"] - target_mean) ** 2
            )
        )
        correlation_ratio = (
            math.sqrt(max(0.0, between_ss / total_ss))
            if total_ss > 0
            else 0.0
        )
        rows.append(
            {
                "row_type": "feature_correlation",
                "feature": feature,
                "label": CATEGORICAL_FEATURE_LABELS[feature],
                "metric": "correlation_ratio_eta",
                "value": correlation_ratio,
                "p_value": "",
                "unit": "association",
            }
        )

    q10 = _gbdt(loss="quantile", quantile=0.10)
    q50 = _gbdt(loss="quantile", quantile=0.50)
    q90 = _gbdt(loss="quantile", quantile=0.90)
    for model in (q10, q50, q90):
        model.fit(x_train, y_train)
    p10 = np.maximum(0.0, q10.predict(x_test))
    p50 = np.maximum(0.0, q50.predict(x_test))
    p90 = np.maximum(0.0, q90.predict(x_test))
    lo = np.minimum(p10, p90)
    hi = np.maximum(p10, p90)
    raw_coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    rows.extend(_metric_rows("Quantile GBDT P50", y_test, p50))
    rows.append(
        {
            "row_type": "interval_metric",
            "model": "Quantile GBDT",
            "metric": "P10-P90 coverage",
            "value": raw_coverage,
            "unit": "ratio",
        }
    )
    prediction_rows: list[dict[str, Any]] = []
    for index, test_row in enumerate(test.itertuples()):
        prediction_rows.append(
            {
                "row_type": "test_prediction",
                "trace_id": test_row.trace_id,
                "agent_id": test_row.agent_id,
                "role": test_row.role,
                "turns_completed": int(test_row.turns_completed),
                "actual_remaining_s": float(y_test[index]),
                "q10_pred_s": float(lo[index]),
                "q50_pred_s": float(p50[index]),
                "q90_pred_s": float(hi[index]),
            }
        )

    for split, split_frame in frame.groupby("split"):
        rows.append(
            {
                "row_type": "dataset",
                "split": split,
                "metric": "examples",
                "value": len(split_frame),
                "unit": "count",
            }
        )
        rows.append(
            {
                "row_type": "dataset",
                "split": split,
                "metric": "workflows",
                "value": split_frame["trace_id"].nunique(),
                "unit": "count",
            }
        )

    metric_lookup = {
        (row.get("model"), row.get("metric")): row["value"]
        for row in rows
        if row["row_type"] == "model_metric"
    }
    summary = {
        "examples": int(len(frame)),
        "agents": int(frame["agent_id"].nunique()),
        "workflows": int(frame["trace_id"].nunique()),
        "roles": {str(key): int(value) for key, value in frame["role"].value_counts().items()},
        "test_workflows": int(test["trace_id"].nunique()),
        "test_examples": int(len(test)),
        "gbdt_mae_s": metric_lookup[("Quantile GBDT P50", "MAE")],
        "gbdt_median_ae_s": metric_lookup[
            ("Quantile GBDT P50", "MedianAE")
        ],
        "gbdt_p90ae_s": metric_lookup[
            ("Quantile GBDT P50", "P90AE")
        ],
        "gbdt_wape": metric_lookup[("Quantile GBDT P50", "WAPE")],
        "gbdt_r2": metric_lookup[("Quantile GBDT P50", "R2")],
        "gbdt_p10_p90_coverage": raw_coverage,
    }
    return rows, prediction_rows, summary


def _configure_chinese_font() -> None:
    regular_font = FANDOL_FONT_DIR / "FandolHei-Regular.otf"
    bold_font = FANDOL_FONT_DIR / "FandolHei-Bold.otf"
    for font_path in (regular_font, bold_font):
        font_manager.fontManager.addfont(str(font_path))
    font_family = font_manager.FontProperties(fname=regular_font).get_name()
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": font_family,
            "axes.unicode_minus": False,
        }
    )


def render_feature_correlations(
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    _configure_chinese_font()
    correlations = pd.DataFrame(
        row
        for row in rows
        if row["row_type"] == "feature_correlation"
    )
    numeric = (
        correlations[correlations["metric"] == "spearman_rho"]
        .sort_values("value")
        .reset_index(drop=True)
    )
    categorical = (
        correlations[correlations["metric"] == "correlation_ratio_eta"]
        .sort_values("value")
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.3),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    numeric_bars = axes[0].barh(
        numeric["label"],
        numeric["value"],
        color="#3979a8",
        alpha=0.92,
    )
    axes[0].bar_label(
        numeric_bars,
        labels=[f"{value:.2f}" for value in numeric["value"]],
        padding=3,
        fontsize=9.5,
    )
    axes[0].set_xlim(0.0, 0.68)
    axes[0].set_xlabel(r"Spearman 相关系数 $\rho$")
    axes[0].set_title("(a) 数值特征与真实剩余时间的秩相关")
    axes[0].grid(axis="x", alpha=0.18)

    categorical_bars = axes[1].barh(
        categorical["label"],
        categorical["value"],
        color="#4f8f6b",
        alpha=0.92,
    )
    axes[1].bar_label(
        categorical_bars,
        labels=[f"{value:.2f}" for value in categorical["value"]],
        padding=3,
        fontsize=9.5,
    )
    axes[1].set_xlim(0.0, 0.35)
    axes[1].set_xlabel(r"相关比 $\eta$")
    axes[1].set_title("(b) 类别特征与真实剩余时间的关联")
    axes[1].grid(axis="x", alpha=0.18)

    fig.suptitle(
        "Real CC：GBDT 输入特征与真实剩余时间的单变量关联",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "相关性用于说明输入信号，不等同于 GBDT 特征重要性，也不表示因果关系。",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _progress_statistics(
    prediction_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["agent_id", "turns_completed"]
    )
    predictions["initial_remaining_s"] = (
        predictions.groupby("agent_id")["actual_remaining_s"]
        .transform("first")
        .clip(lower=1.0)
    )
    predictions["normalized_abs_error"] = (
        (
            predictions["q50_pred_s"]
            - predictions["actual_remaining_s"]
        ).abs()
        / predictions["initial_remaining_s"]
    )
    predictions["turn_bucket"] = np.where(
        predictions["turns_completed"] < 10,
        predictions["turns_completed"],
        10,
    ).astype(int)
    per_agent = (
        predictions.groupby(
            ["agent_id", "turn_bucket"],
            as_index=False,
        )["normalized_abs_error"]
        .median()
    )
    quantiles = (
        per_agent.groupby("turn_bucket")["normalized_abs_error"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .rename(
            columns={
                0.10: "p10",
                0.25: "p25",
                0.50: "median",
                0.75: "p75",
                0.90: "p90",
            }
        )
    )
    quantiles["agents"] = per_agent.groupby("turn_bucket")[
        "agent_id"
    ].nunique()
    return quantiles.reset_index()


def render_gbdt_accuracy_progress(
    prediction_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    _configure_chinese_font()
    stats = _progress_statistics(prediction_rows)
    x = stats["turn_bucket"].to_numpy()
    median = stats["median"].to_numpy()
    p10 = np.maximum(stats["p10"].to_numpy(), 1e-3)
    p25 = np.maximum(stats["p25"].to_numpy(), 1e-3)
    p75 = np.maximum(stats["p75"].to_numpy(), 1e-3)
    p90 = np.maximum(stats["p90"].to_numpy(), 1e-3)

    fig, axis = plt.subplots(figsize=(11.8, 5.6))
    axis.fill_between(
        x,
        p10,
        p90,
        color="#a9c7df",
        alpha=0.26,
        label="P10--P90",
    )
    axis.fill_between(
        x,
        p25,
        p75,
        color="#5f9bc4",
        alpha=0.34,
        label="P25--P75",
    )
    axis.plot(
        x,
        median,
        color="#1f5f8b",
        marker="o",
        markersize=6,
        linewidth=2.4,
        label="中位误差",
    )
    axis.annotate(
        f"{median[0]:.2f}",
        (x[0], median[0]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color="#1f5f8b",
    )
    axis.annotate(
        f"{median[-1]:.2f}",
        (x[-1], median[-1]),
        xytext=(-8, 11),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        fontweight="bold",
        color="#1f5f8b",
    )
    axis.annotate(
        "整体中位误差下降 48%",
        xy=(x[-1], median[-1]),
        xytext=(6.8, 1.55),
        arrowprops={"arrowstyle": "->", "color": "#d05a45"},
        fontsize=10.5,
        color="#b64a39",
        fontweight="bold",
    )
    axis.set_yscale("log")
    axis.set_ylim(0.05, 12)
    axis.set_yticks([0.1, 1.0, 10.0])
    axis.set_yticklabels(["0.1", "1", "10"])
    axis.set_xlim(0.7, 10.3)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [str(value) if value < 10 else "10+" for value in x]
    )
    axis.set_xlabel("已完成轮次数（每轮结束后重新预测）")
    axis.set_ylabel(
        r"归一化绝对误差 $|\widehat{T}^{rem}_{i,t}-T^{rem}_{i,t}|/T^{rem}_{i,1}$"
    )
    axis.set_title(
        "Real CC 工作流独立测试集：Quantile GBDT 的 P50 预测随轮次更新",
        fontsize=13.5,
        fontweight="bold",
    )
    axis.grid(which="both", axis="y", alpha=0.20)
    axis.legend(frameon=False, loc="upper right", ncol=3)
    axis.text(
        0.015,
        0.025,
        (
            f"P90：第 1 轮 {stats.iloc[0]['p90']:.2f}"
            f" → 第 10 轮及以后 {stats.iloc[-1]['p90']:.2f}；"
            "单个智能体的误差不保证逐轮单调下降。"
        ),
        transform=axis.transAxes,
        fontsize=9.8,
        color="#555555",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--feature-png",
        type=Path,
        default=DEFAULT_FEATURE_PNG,
    )
    parser.add_argument(
        "--progress-png",
        type=Path,
        default=DEFAULT_PROGRESS_PNG,
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    frame = load_examples(args.trace)
    rows, prediction_rows, summary = evaluate(frame)
    write_csv(rows + prediction_rows, args.csv)
    render_feature_correlations(rows, args.feature_png)
    render_gbdt_accuracy_progress(prediction_rows, args.progress_png)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
