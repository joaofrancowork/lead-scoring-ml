"""Drift monitoring with the Population Stability Index (PSI).

The dataset shows why this matters: the conversion rate climbs from ~3% to ~46%
over the collection period. A model trained on the early months is miscalibrated
for the late ones, and nothing in the request payload says so. Comparing each
incoming batch with the training data catches the shift before the sales team does.

Rule of thumb: PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 investigate / retrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WATCH, ALERT = 0.1, 0.25
MAX_DISCRETE_VALUES = 30
EPS = 1e-4


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference.dropna(), edges)[0] / max(1, reference.notna().sum())
    cur = np.histogram(current.dropna(), edges)[0] / max(1, current.notna().sum())
    return _psi(ref, cur)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    ref = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi(ref, cur)


def _psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


@dataclass
class DriftReport:
    feature_psi: dict[str, float]

    @property
    def alerts(self) -> list[str]:
        return [f for f, v in self.feature_psi.items() if v > ALERT]

    @property
    def watch(self) -> list[str]:
        return [f for f, v in self.feature_psi.items() if WATCH < v <= ALERT]

    def to_markdown(self) -> str:
        rows = sorted(self.feature_psi.items(), key=lambda kv: -kv[1])
        lines = ["| Feature | PSI | Status |", "|---|---|---|"]
        for feature, value in rows:
            status = "alert" if value > ALERT else "watch" if value > WATCH else "stable"
            lines.append(f"| {feature} | {value:.3f} | {status} |")
        return "\n".join(lines)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> DriftReport:
    scores = {}
    for column in columns:
        # Low-cardinality numerics (pdays = 999 for most rows, previous = 0..7) collapse
        # quantile bins into one, which hides real shifts; compare them as categories.
        is_discrete = reference[column].nunique() <= MAX_DISCRETE_VALUES
        if pd.api.types.is_numeric_dtype(reference[column]) and not is_discrete:
            scores[column] = psi_numeric(reference[column], current[column])
        else:
            scores[column] = psi_categorical(reference[column], current[column])
    return DriftReport(scores)
