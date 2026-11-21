from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lead_scoring.data import TARGET, download, load, random_split, time_split
from lead_scoring.features import add_features, build_preprocessor, input_columns
from lead_scoring.metrics import capture_at, evaluate, lift_at, top_fraction_mask

CSV = '"age";"job";"duration";"pdays";"y"\n30;"admin.";120;999;"no"\n45;"retired";300;3;"yes"\n'


def _nested_zip() -> bytes:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("bank-additional/bank-additional-full.csv", CSV)
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("bank-additional.zip", inner.getvalue())
    return outer.getvalue()


def test_download_extracts_nested_zip_once(tmp_path: Path) -> None:
    target = tmp_path / "raw" / "data.csv"
    fake = io.BytesIO(_nested_zip())
    with patch("urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = fake
        assert download(target) == target
        assert download(target) == target  # cached, no second request
    assert urlopen.call_count == 1

    df = load(target)
    assert "duration" not in df.columns  # leakage removed
    assert df[TARGET].tolist() == [0, 1]
    assert df["contact_order"].tolist() == [0, 1]


def test_time_split_is_chronological(contacts: pd.DataFrame) -> None:
    train, val, test = time_split(contacts.sample(frac=1, random_state=0))
    assert len(train) + len(val) + len(test) == len(contacts)
    assert train["contact_order"].max() < val["contact_order"].min()
    assert val["contact_order"].max() < test["contact_order"].min()


def test_random_split_is_stratified(contacts: pd.DataFrame) -> None:
    train, val, test = random_split(contacts)
    rates = [part[TARGET].mean() for part in (train, val, test)]
    assert max(rates) - min(rates) < 0.03


def test_add_features_treats_999_as_never_contacted() -> None:
    df = pd.DataFrame({"pdays": [999, 4], "campaign": [1, 3]})
    out = add_features(df)
    assert out["previously_contacted"].tolist() == [0, 1]
    assert np.isnan(out.loc[0, "days_since_contact"]) and out.loc[1, "days_since_contact"] == 4
    assert out["log_campaign"].iloc[1] == pytest.approx(np.log(4))


def test_client_feature_set_excludes_context(contacts: pd.DataFrame) -> None:
    client = input_columns("client")
    assert "euribor3m" not in client and "month" not in client and "job" in client
    assert set(client) < set(input_columns("all"))
    matrix = build_preprocessor(feature_set="client").fit_transform(contacts[client])
    assert matrix.shape[0] == len(contacts) and not np.isnan(matrix).any()


def test_preprocessor_ignores_unseen_categories(contacts: pd.DataFrame) -> None:
    prep = build_preprocessor().fit(contacts)
    unseen = contacts.head(2).assign(job="astronaut")
    assert prep.transform(unseen).shape[0] == 2


def test_ranking_metrics() -> None:
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    perfect = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8])
    assert top_fraction_mask(perfect, 0.2).sum() == 2
    assert lift_at(y, perfect, 0.2) == pytest.approx(5.0)
    assert capture_at(y, perfect, 0.2) == 1.0
    assert lift_at(np.zeros(4), np.ones(4)) == 0.0 and capture_at(np.zeros(4), np.ones(4)) == 0.0
    metrics = evaluate(y, perfect)
    assert metrics["roc_auc"] == 1.0 and metrics["pr_auc"] == 1.0
    assert metrics["base_rate"] == pytest.approx(0.2)
