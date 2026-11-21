from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from lead_scoring import cli
from lead_scoring.backtest import backtest, folds, summarise, to_markdown
from lead_scoring.data import TARGET

from .conftest import synthetic_contacts


def test_folds_are_consecutive_and_never_look_ahead() -> None:
    cuts = folds(1000, n_folds=4, min_train_frac=0.6)
    assert cuts == [(600, 700), (700, 800), (800, 900), (900, 1000)]
    for (train_end, test_end), (next_train_end, _) in itertools.pairwise(cuts):
        assert train_end < test_end == next_train_end
    with pytest.raises(ValueError):
        folds(1000, n_folds=0)
    with pytest.raises(ValueError):
        folds(3, n_folds=5)


def test_backtest_reports_every_model_and_fold() -> None:
    results = backtest(synthetic_contacts(1500), n_folds=3, fast=True, track=False)
    assert len(results) == 3 * 3  # three candidates x three folds
    assert all(r.train_rows < 1500 and r.test_rows > 0 for r in results)
    summary = summarise(results)
    assert summary.loc["prior_baseline", ("roc_auc", "mean")] == 0.5
    assert summary.index[0] != "prior_baseline"  # a real model ranks first on PR-AUC
    assert "| logistic_regression |" in to_markdown(results) and "±" in to_markdown(results)


def test_cli_backtest_tracks_to_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    data = tmp_path / "contacts.csv"
    raw = synthetic_contacts(900).drop(columns=["contact_order"])
    raw["y"] = raw.pop(TARGET).map({1: "yes", 0: "no"})
    raw.to_csv(data, sep=";", index=False)
    assert cli.main(["backtest", "--data", str(data), "--folds", "2", "--fast"]) == 0
    assert "| Model |" in capsys.readouterr().out
    assert (tmp_path / "mlflow.db").exists()
