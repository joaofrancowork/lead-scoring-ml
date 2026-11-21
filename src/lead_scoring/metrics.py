"""Model metrics plus the ones a sales team actually cares about.

With ~11% positives, accuracy is meaningless. PR-AUC measures ranking quality on
the rare class, Brier score measures calibration, and lift / capture at the top
decile answer the business question: "if we only call 10% of leads, how many of
the conversions do we reach?"
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

FloatArray = npt.NDArray[np.float64]


def top_fraction_mask(scores: FloatArray, fraction: float) -> npt.NDArray[np.bool_]:
    k = max(1, round(len(scores) * fraction))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def lift_at(y: npt.ArrayLike, scores: FloatArray, fraction: float = 0.1) -> float:
    y_arr = np.asarray(y)
    base_rate = y_arr.mean()
    if base_rate == 0:
        return 0.0
    return float(y_arr[top_fraction_mask(scores, fraction)].mean() / base_rate)


def capture_at(y: npt.ArrayLike, scores: FloatArray, fraction: float = 0.1) -> float:
    y_arr = np.asarray(y, dtype=np.float64)
    total = float(y_arr.sum())
    return float(y_arr[top_fraction_mask(scores, fraction)].sum()) / total if total else 0.0


def evaluate(y: npt.ArrayLike, scores: FloatArray) -> dict[str, float]:
    y_arr = np.asarray(y)
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y_arr, scores)),
        "pr_auc": float(average_precision_score(y_arr, scores)),
        "brier": float(brier_score_loss(y_arr, clipped)),
        "log_loss": float(log_loss(y_arr, clipped)),
        "lift_at_10": lift_at(y_arr, scores, 0.1),
        "capture_at_10": capture_at(y_arr, scores, 0.1),
        "capture_at_30": capture_at(y_arr, scores, 0.3),
        "base_rate": float(y_arr.mean()),
    }
