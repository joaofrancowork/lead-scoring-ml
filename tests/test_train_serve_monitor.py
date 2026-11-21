from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lead_scoring import cli
from lead_scoring.data import TARGET, time_split
from lead_scoring.monitoring import drift_report, psi_categorical, psi_numeric
from lead_scoring.serving import create_app, get_model
from lead_scoring.train import train

from .conftest import synthetic_contacts


@pytest.fixture
def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    out = tmp_path / "model"
    result = train(
        *time_split(synthetic_contacts(1500)), out_dir=out, fast=True, feature_set="client"
    )
    assert result.best in {"logistic_regression", "hist_gradient_boosting"}
    assert result.validation["prior_baseline"]["roc_auc"] == 0.5
    assert result.test["roc_auc"] > 0.6  # the synthetic signal is learnable
    return out


def test_training_writes_model_card_plot_and_mlflow_runs(trained: Path) -> None:
    card = json.loads((trained / "model_card.json").read_text())
    assert card["feature_set"] == "client" and "euribor3m" not in card["input_columns"]
    assert card["priority_thresholds"]["top_10"] >= card["priority_thresholds"]["top_30"]
    assert (trained / "calibration.png").stat().st_size > 1000
    assert (trained.parent / "mlflow.db").exists()


LEAD = {
    "age": 67, "job": "retired", "marital": "married", "education": "basic.9y",
    "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
    "month": "nov", "day_of_week": "tue", "campaign": 1, "pdays": 3, "previous": 1,
    "poutcome": "success", "emp_var_rate": -1.8, "cons_price_idx": 92.9,
    "cons_conf_idx": -46.2, "euribor3m": 1.3, "nr_employed": 5099.1,
}  # fmt: skip


def test_api_scores_and_prioritises(trained: Path) -> None:
    get_model.cache_clear()
    client = TestClient(create_app(trained))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model").json()["feature_set"] == "client"
    cold = {**LEAD, "poutcome": "nonexistent", "pdays": 999, "previous": 0, "campaign": 9,
            "contact": "telephone", "job": "services"}  # fmt: skip
    res = client.post("/score", json={"leads": [LEAD, cold]}).json()
    hot, not_hot = res["scores"]
    assert hot["probability"] > not_hot["probability"]
    assert hot["priority"] in {"call_first", "call_this_week"}
    assert client.post("/score", json={"leads": []}).status_code == 422
    assert client.post("/score", json={"leads": [{**LEAD, "age": 5}]}).status_code == 422


def test_api_without_model_returns_503(tmp_path: Path) -> None:
    get_model.cache_clear()
    assert TestClient(create_app(tmp_path / "missing")).get("/health").status_code == 503


def test_psi_detects_shift() -> None:
    ref = pd.Series(range(1000), dtype=float)
    assert psi_numeric(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi_numeric(ref, ref + 800) > 0.25
    assert psi_numeric(pd.Series([1.0] * 10), pd.Series([2.0] * 10)) == 0.0
    cats = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi_categorical(cats, cats) == pytest.approx(0.0, abs=1e-9)
    assert psi_categorical(cats, pd.Series(["a"] * 95 + ["c"] * 5)) > 0.25


def test_drift_report_flags_shifted_columns(contacts: pd.DataFrame) -> None:
    shifted = contacts.assign(euribor3m=contacts["euribor3m"] + 10, month="dec")
    report = drift_report(contacts, shifted, ["euribor3m", "month", "age"])
    assert set(report.alerts) == {"euribor3m", "month"} and report.watch == []
    assert "| euribor3m |" in report.to_markdown() and "stable" in report.to_markdown()


def test_low_cardinality_numeric_shift_is_not_hidden() -> None:
    ref = pd.DataFrame({"pdays": [999] * 95 + [3] * 5})
    cur = pd.DataFrame({"pdays": [999] * 60 + [3] * 40})
    assert drift_report(ref, cur, ["pdays"]).alerts == ["pdays"]


def test_cli_train_and_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    data = tmp_path / "contacts.csv"
    raw = synthetic_contacts(900).drop(columns=["contact_order"])
    raw["y"] = raw.pop(TARGET).map({1: "yes", 0: "no"})
    raw.columns = [
        c.replace("_", ".")
        if c in {"emp_var_rate", "cons_price_idx", "cons_conf_idx", "nr_employed"}
        else c
        for c in raw.columns
    ]
    raw.to_csv(data, sep=";", index=False)

    assert cli.main(["download", "--path", str(data)]) == 0  # already present: no network
    assert cli.main(["train", "--data", str(data), "--out", str(tmp_path / "m"), "--fast"]) == 0
    assert '"best"' in capsys.readouterr().out
    assert cli.main(["drift", "--data", str(data)]) == 0
    assert "alerts:" in capsys.readouterr().out
