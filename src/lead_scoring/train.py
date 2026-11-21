"""Training pipeline: candidates -> model selection on validation -> calibration -> test.

Every candidate is an MLflow run with its parameters and validation metrics; the
selected model is calibrated on the validation set, evaluated once on the
held-out test set, and saved with a model card the API serves.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from lead_scoring.data import TARGET
from lead_scoring.features import build_preprocessor, input_columns
from lead_scoring.metrics import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
EXPERIMENT = "lead-scoring"


def candidates(fast: bool = False, feature_set: str = "all") -> dict[str, Callable[[], Pipeline]]:
    iterations = 30 if fast else 400
    return {
        "prior_baseline": lambda: Pipeline(
            [
                ("prep", build_preprocessor(feature_set=feature_set)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": lambda: Pipeline(
            [
                ("prep", build_preprocessor(feature_set=feature_set)),
                ("model", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
            ]
        ),
        "hist_gradient_boosting": lambda: Pipeline(
            [
                ("prep", build_preprocessor(scale=False, feature_set=feature_set)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=iterations,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


@dataclass
class TrainResult:
    best: str
    validation: dict[str, dict[str, float]]
    test: dict[str, float]
    model_dir: Path


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _scores(model: ClassifierMixin, X: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def _calibration_plot(y: pd.Series, scores: np.ndarray[Any, Any], path: Path) -> Path:
    frac_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999", label="perfect")
    ax.plot(mean_pred, frac_pos, marker="o", color="#111", label="model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed conversion rate")
    ax.set_title("Calibration on the test period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _ensure_tracking() -> None:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        # Local SQLite store by default: no tracking server needed to reproduce a run.
        mlflow.set_tracking_uri(f"sqlite:///{Path.cwd() / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    split_name: str = "time",
    fast: bool = False,
    feature_set: str = "all",
) -> TrainResult:
    _ensure_tracking()
    columns = input_columns(feature_set)
    X_train, y_train = train_df[columns], train_df[TARGET]
    X_val, y_val = val_df[columns], val_df[TARGET]
    X_test, y_test = test_df[columns], test_df[TARGET]
    tag = f"{split_name}-{feature_set}"

    fitted: dict[str, Pipeline] = {}
    validation: dict[str, dict[str, float]] = {}
    for name, make in candidates(fast, feature_set).items():
        with mlflow.start_run(run_name=f"{name}-{tag}"):
            started = time.perf_counter()
            model = make().fit(X_train, y_train)
            metrics = evaluate(y_val, _scores(model, X_val))
            mlflow.log_params(
                {
                    "model": name,
                    "split": split_name,
                    "n_train": len(train_df),
                    "git_sha": _git_sha(),
                }
            )
            mlflow.log_params(
                {
                    f"model__{k}": v
                    for k, v in model.named_steps["model"].get_params().items()
                    if isinstance(v, int | float | str | bool)
                }
            )
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
            mlflow.log_metric("fit_seconds", time.perf_counter() - started)
        fitted[name], validation[name] = model, metrics
        log.info("%s: val PR-AUC %.3f ROC-AUC %.3f", name, metrics["pr_auc"], metrics["roc_auc"])

    best = max(validation, key=lambda n: validation[n]["pr_auc"])
    # Isotonic calibration on the validation period: scores become probabilities the
    # sales team can reason about ("30% likely"), not just a ranking.
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted[best]), method="isotonic")
    calibrated.fit(X_val, y_val)
    test_scores = _scores(calibrated, X_test)
    test_metrics = evaluate(y_test, test_scores)
    val_scores = _scores(calibrated, X_val)
    thresholds = {
        "top_10": float(np.quantile(val_scores, 0.9)),
        "top_30": float(np.quantile(val_scores, 0.7)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, out_dir / "model.joblib")
    card = {
        "model": best,
        "calibration": "isotonic (validation period)",
        "split": split_name,
        "feature_set": feature_set,
        "trained_rows": len(train_df),
        "input_columns": columns,
        "excluded_columns": ["duration (known only after the call)"],
        "priority_thresholds": thresholds,
        "validation": validation[best],
        "test": test_metrics,
        "git_sha": _git_sha(),
        "data": "UCI Bank Marketing (bank-additional-full), CC BY 4.0",
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    plot = _calibration_plot(y_test, test_scores, out_dir / "calibration.png")

    with mlflow.start_run(run_name=f"{best}-calibrated-{tag}"):
        mlflow.log_params(
            {
                "model": best,
                "calibration": "isotonic",
                "split": split_name,
                "feature_set": feature_set,
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(str(out_dir / "model_card.json"))
        mlflow.log_artifact(str(plot))
        mlflow.log_artifact(str(out_dir / "model.joblib"))

    return TrainResult(best=best, validation=validation, test=test_metrics, model_dir=out_dir)
