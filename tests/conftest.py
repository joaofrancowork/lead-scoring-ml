from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_scoring.data import TARGET

RNG_SEED = 7


def synthetic_contacts(n: int = 1200, seed: int = RNG_SEED) -> pd.DataFrame:
    """Small dataset with the same schema as bank-additional, and a learnable signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "contact_order": np.arange(n),
            "age": rng.integers(18, 90, n),
            "job": rng.choice(["admin.", "technician", "student", "retired", "services"], n),
            "marital": rng.choice(["married", "single", "divorced"], n),
            "education": rng.choice(["basic.9y", "high.school", "university.degree"], n),
            "default": rng.choice(["no", "unknown"], n),
            "housing": rng.choice(["yes", "no"], n),
            "loan": rng.choice(["yes", "no"], n),
            "contact": rng.choice(["cellular", "telephone"], n),
            "month": rng.choice(["may", "jun", "jul", "nov"], n),
            "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], n),
            "campaign": rng.integers(1, 10, n),
            "pdays": np.where(rng.random(n) < 0.85, 999, rng.integers(0, 20, n)),
            "previous": rng.integers(0, 3, n),
            "poutcome": rng.choice(["nonexistent", "failure", "success"], n, p=[0.85, 0.1, 0.05]),
            "emp_var_rate": rng.normal(0, 1.5, n),
            "cons_price_idx": rng.normal(93.5, 0.5, n),
            "cons_conf_idx": rng.normal(-40, 4, n),
            "euribor3m": rng.uniform(0.6, 5.0, n),
            "nr_employed": rng.normal(5150, 60, n),
        }
    )
    logit = (
        -2.4
        + 1.8 * (df["poutcome"] == "success")
        + 0.9 * (df["contact"] == "cellular")
        + 0.8 * df["job"].isin(["student", "retired"])
        - 0.15 * df["campaign"]
    )
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def contacts() -> pd.DataFrame:
    return synthetic_contacts()
