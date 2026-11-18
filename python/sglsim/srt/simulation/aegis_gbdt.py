"""Small deterministic GBDT used by the AegisKV trace simulator.

The production design calls for a GBDT remaining-time predictor, but SGLSim's
runtime dependency set intentionally does not include scikit-learn.  This file
implements the narrow regression functionality that the simulator needs:

* squared-error gradient boosting;
* shallow histogram regression trees;
* deterministic quantile binning; and
* a log1p target transform for heavy-tailed request durations.

It is not a general machine-learning library.  Keeping the implementation here
also makes the exact predictor used by an experiment auditable and avoids a
silent fallback to random or oracle Slack.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class AegisGBDTConfig:
    estimators: int = 48
    learning_rate: float = 0.06
    max_depth: int = 3
    min_leaf: int = 48
    max_bins: int = 32
    min_gain: float = 1e-9

    def __post_init__(self) -> None:
        if int(self.estimators) <= 0:
            raise ValueError("estimators must be positive")
        if not math.isfinite(float(self.learning_rate)) or not 0.0 < float(
            self.learning_rate
        ) <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if int(self.max_depth) <= 0:
            raise ValueError("max_depth must be positive")
        if int(self.min_leaf) <= 0:
            raise ValueError("min_leaf must be positive")
        if int(self.max_bins) < 2:
            raise ValueError("max_bins must be at least two")
        if not math.isfinite(float(self.min_gain)) or float(self.min_gain) < 0.0:
            raise ValueError("min_gain must be finite and nonnegative")


@dataclass(frozen=True)
class _TreeNode:
    value: float
    feature: int = -1
    threshold_bin: int = -1
    left: int = -1
    right: int = -1

    @property
    def leaf(self) -> bool:
        return self.feature < 0


class _HistogramRegressionTree:
    def __init__(self, *, max_depth: int, min_leaf: int, min_gain: float) -> None:
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        self.min_gain = float(min_gain)
        self.nodes: list[_TreeNode] = []

    @staticmethod
    def _squared_error(count: np.ndarray, total: np.ndarray, square: np.ndarray) -> np.ndarray:
        return square - np.divide(
            total * total,
            count,
            out=np.zeros_like(total, dtype=float),
            where=count > 0,
        )

    def fit(self, binned: np.ndarray, residual: np.ndarray, bin_counts: Sequence[int]) -> None:
        if binned.ndim != 2 or residual.ndim != 1:
            raise ValueError("tree fit expects a 2-D feature matrix and 1-D target")
        if len(binned) != len(residual) or not len(residual):
            raise ValueError("tree fit received mismatched or empty samples")
        self.nodes = []

        def build(indices: np.ndarray, depth: int) -> int:
            values = residual[indices]
            value = float(values.mean())
            node_index = len(self.nodes)
            self.nodes.append(_TreeNode(value=value))
            if depth >= self.max_depth or len(indices) < 2 * self.min_leaf:
                return node_index

            parent_sum = float(values.sum())
            parent_square = float(np.dot(values, values))
            parent_error = parent_square - parent_sum * parent_sum / len(indices)
            best_gain = self.min_gain
            best_feature = -1
            best_threshold = -1

            for feature, feature_bins in enumerate(bin_counts):
                if feature_bins <= 1:
                    continue
                bins = binned[indices, feature]
                count = np.bincount(bins, minlength=feature_bins).astype(float)
                total = np.bincount(
                    bins, weights=values, minlength=feature_bins
                ).astype(float)
                square = np.bincount(
                    bins, weights=values * values, minlength=feature_bins
                ).astype(float)
                left_count = np.cumsum(count)[:-1]
                right_count = len(indices) - left_count
                valid = (left_count >= self.min_leaf) & (
                    right_count >= self.min_leaf
                )
                if not np.any(valid):
                    continue
                left_total = np.cumsum(total)[:-1]
                right_total = parent_sum - left_total
                left_square = np.cumsum(square)[:-1]
                right_square = parent_square - left_square
                split_error = self._squared_error(
                    left_count, left_total, left_square
                ) + self._squared_error(right_count, right_total, right_square)
                gains = parent_error - split_error
                gains[~valid] = -np.inf
                threshold = int(np.argmax(gains))
                gain = float(gains[threshold])
                if gain > best_gain + 1e-15 or (
                    math.isclose(gain, best_gain, rel_tol=0.0, abs_tol=1e-15)
                    and (feature, threshold) < (best_feature, best_threshold)
                ):
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

            if best_feature < 0:
                return node_index
            goes_left = binned[indices, best_feature] <= best_threshold
            left_indices = indices[goes_left]
            right_indices = indices[~goes_left]
            if len(left_indices) < self.min_leaf or len(right_indices) < self.min_leaf:
                return node_index
            left = build(left_indices, depth + 1)
            right = build(right_indices, depth + 1)
            self.nodes[node_index] = _TreeNode(
                value=value,
                feature=best_feature,
                threshold_bin=best_threshold,
                left=left,
                right=right,
            )
            return node_index

        build(np.arange(len(residual), dtype=np.int64), 0)

    def predict_binned(self, binned: np.ndarray) -> np.ndarray:
        if not self.nodes:
            raise RuntimeError("tree has not been fitted")
        output = np.empty(len(binned), dtype=float)
        stack: list[tuple[int, np.ndarray]] = [
            (0, np.arange(len(binned), dtype=np.int64))
        ]
        while stack:
            node_index, indices = stack.pop()
            if not len(indices):
                continue
            node = self.nodes[node_index]
            if node.leaf:
                output[indices] = node.value
                continue
            goes_left = binned[indices, node.feature] <= node.threshold_bin
            stack.append((node.right, indices[~goes_left]))
            stack.append((node.left, indices[goes_left]))
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "value": node.value,
                    "feature": node.feature,
                    "threshold_bin": node.threshold_bin,
                    "left": node.left,
                    "right": node.right,
                }
                for node in self.nodes
            ]
        }


class AegisGBDTRegressor:
    """Deterministic histogram GBDT for causal remaining-time prediction."""

    def __init__(self, config: AegisGBDTConfig | None = None) -> None:
        self.config = config or AegisGBDTConfig()
        self._edges: list[np.ndarray] = []
        self._trees: list[_HistogramRegressionTree] = []
        self._initial = 0.0
        self._feature_count = 0
        self._fitted_samples = 0

    def _fit_edges(self, features: np.ndarray) -> None:
        quantiles = np.linspace(0.0, 1.0, self.config.max_bins + 1)[1:-1]
        self._edges = []
        for feature in range(features.shape[1]):
            column = features[:, feature]
            raw = np.quantile(column, quantiles, method="linear")
            self._edges.append(np.unique(raw[np.isfinite(raw)]).astype(float))

    def _bin(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != self._feature_count:
            raise ValueError("feature matrix shape differs from fitted GBDT schema")
        binned = np.empty(features.shape, dtype=np.int16)
        for feature, edges in enumerate(self._edges):
            binned[:, feature] = np.searchsorted(
                edges, features[:, feature], side="right"
            )
        return binned

    def fit(self, features: np.ndarray, target_ms: np.ndarray) -> "AegisGBDTRegressor":
        features = np.asarray(features, dtype=float)
        target_ms = np.asarray(target_ms, dtype=float)
        if features.ndim != 2 or target_ms.ndim != 1:
            raise ValueError("GBDT fit expects 2-D features and a 1-D target")
        if len(features) != len(target_ms) or not len(target_ms):
            raise ValueError("GBDT fit received mismatched or empty samples")
        if not np.all(np.isfinite(features)):
            raise ValueError("GBDT features must be finite")
        if not np.all(np.isfinite(target_ms)) or np.any(target_ms < 0.0):
            raise ValueError("remaining-time targets must be finite and nonnegative")
        self._feature_count = features.shape[1]
        self._fitted_samples = len(features)
        self._fit_edges(features)
        binned = self._bin(features)
        transformed = np.log1p(target_ms)
        self._initial = float(transformed.mean())
        prediction = np.full(len(transformed), self._initial, dtype=float)
        self._trees = []
        bin_counts = [len(edges) + 1 for edges in self._edges]
        for _ in range(self.config.estimators):
            residual = transformed - prediction
            tree = _HistogramRegressionTree(
                max_depth=self.config.max_depth,
                min_leaf=self.config.min_leaf,
                min_gain=self.config.min_gain,
            )
            tree.fit(binned, residual, bin_counts)
            prediction += self.config.learning_rate * tree.predict_binned(binned)
            self._trees.append(tree)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._trees:
            raise RuntimeError("GBDT has not been fitted")
        features = np.asarray(features, dtype=float)
        if not np.all(np.isfinite(features)):
            raise ValueError("GBDT features must be finite")
        binned = self._bin(features)
        prediction = np.full(len(features), self._initial, dtype=float)
        for tree in self._trees:
            prediction += self.config.learning_rate * tree.predict_binned(binned)
        return np.maximum(np.expm1(prediction), 0.0)

    def manifest(self) -> dict[str, Any]:
        if not self._trees:
            raise RuntimeError("GBDT has not been fitted")
        payload = {
            "schema": "aegis_histogram_gbdt_v1",
            "config": {
                "estimators": self.config.estimators,
                "learning_rate": self.config.learning_rate,
                "max_depth": self.config.max_depth,
                "min_leaf": self.config.min_leaf,
                "max_bins": self.config.max_bins,
                "min_gain": self.config.min_gain,
            },
            "feature_count": self._feature_count,
            "fitted_samples": self._fitted_samples,
            "initial_log_target": self._initial,
            "edges": [edge.tolist() for edge in self._edges],
            "trees": [tree.to_dict() for tree in self._trees],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return {
            "schema": payload["schema"],
            "config": payload["config"],
            "feature_count": self._feature_count,
            "fitted_samples": self._fitted_samples,
            "tree_count": len(self._trees),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }


__all__ = ["AegisGBDTConfig", "AegisGBDTRegressor"]
