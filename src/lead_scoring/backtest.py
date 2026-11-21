"""Rolling-origin (expanding window) backtest.

A single time split gives one noisy number: on this dataset the selected model
flips between logistic regression and gradient boosting depending on the cut.
The backtest trains on everything before each cut-off and scores the next block,
for several consecutive blocks, and reports the mean and spread per candidate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from lead_scoring.data import TARGET
from lead_scoring.features import input_columns
from lead_scoring.metrics import evaluate
from lead_scoring.train import candidates

log = logging.getLogger(__name__)
REPORTED = ("roc_auc", "pr_auc", "lift_at_10", "base_rate")


@dataclass
class FoldResult:
    model: str
    fold: int
    train_rows: int
    test_rows: int
    metrics: dict[str, float]


def folds(n_rows: int, n_folds: int = 5, min_train_frac: float = 0.5) -> list[tuple[int, int]]:
    """(train_end, test_end) row positions; test blocks are consecutive and equal-sized."""
    if not 0 < min_train_frac < 1 or n_folds < 1:
        raise ValueError("need 0 < min_train_frac < 1 and n_folds >= 1")
    start = int(n_rows * min_train_frac)
    block = (n_rows - start) // n_folds
    if block == 0:
        raise ValueError("not enough rows for this many folds")
    return [(start + i * block, start + (i + 1) * block) for i in range(n_folds)]


def backtest(
    df: pd.DataFrame,
    feature_set: str = "client",
    n_folds: int = 5,
    fast: bool = False,
    track: bool = True,
) -> list[FoldResult]:
    df = df.sort_values("contact_order").reset_index(drop=True)
    columns = input_columns(feature_set)
    results: list[FoldResult] = []
    for i, (train_end, test_end) in enumerate(folds(len(df), n_folds), start=1):
        train, test = df.iloc[:train_end], df.iloc[train_end:test_end]
        for name, make in candidates(fast, feature_set).items():
            model = make().fit(train[columns], train[TARGET])
            scores = np.asarray(model.predict_proba(test[columns])[:, 1], dtype=np.float64)
            metrics = evaluate(test[TARGET], scores)
            results.append(FoldResult(name, i, len(train), len(test), metrics))
            log.info("fold %d %s ROC-AUC %.3f", i, name, metrics["roc_auc"])
    if track:
        _log_to_mlflow(results, feature_set, n_folds)
    return results


def summarise(results: list[FoldResult]) -> pd.DataFrame:
    rows = [{"model": r.model, "fold": r.fold, **r.metrics} for r in results]
    frame = pd.DataFrame(rows)
    summary = frame.groupby("model")[list(REPORTED)].agg(["mean", "std"])
    return summary.sort_values(("pr_auc", "mean"), ascending=False)


def to_markdown(results: list[FoldResult]) -> str:
    summary = summarise(results)
    lines = [
        "| Model | ROC-AUC | PR-AUC | Lift@10% | Base rate (mean) |",
        "|---|---|---|---|---|",
    ]
    for model, row in summary.iterrows():
        lines.append(
            f"| {model} | {row[('roc_auc', 'mean')]:.3f} ± {row[('roc_auc', 'std')]:.3f} | "
            f"{row[('pr_auc', 'mean')]:.3f} ± {row[('pr_auc', 'std')]:.3f} | "
            f"{row[('lift_at_10', 'mean')]:.2f} ± {row[('lift_at_10', 'std')]:.2f} | "
            f"{row[('base_rate', 'mean')] * 100:.1f}% |"
        )
    return "\n".join(lines)


def _log_to_mlflow(results: list[FoldResult], feature_set: str, n_folds: int) -> None:
    from lead_scoring.train import _ensure_tracking

    _ensure_tracking()
    summary = summarise(results)
    for model in summary.index:
        with mlflow.start_run(run_name=f"{model}-backtest-{feature_set}"):
            mlflow.log_params(
                {"model": model, "split": "rolling", "feature_set": feature_set, "folds": n_folds}
            )
            for metric in REPORTED:
                mlflow.log_metric(f"bt_{metric}_mean", float(summary.loc[model, (metric, "mean")]))
                mlflow.log_metric(f"bt_{metric}_std", float(summary.loc[model, (metric, "std")]))
            for r in results:
                if r.model == model:
                    mlflow.log_metric("fold_roc_auc", r.metrics["roc_auc"], step=r.fold)
