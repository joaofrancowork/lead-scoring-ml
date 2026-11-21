"""Feature engineering as a scikit-learn pipeline, so training and serving share it."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
NUMERIC = [
    "age",
    "campaign",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
]
ENGINEERED = ["previously_contacted", "days_since_contact", "log_campaign"]
INPUT_COLUMNS = CATEGORICAL + NUMERIC + ["pdays"]
NEVER_CONTACTED = 999

# Macro indicators and the calendar month describe *when* a contact happened, not *who*
# the lead is. Under a time split they mostly encode the period, and the period changes.
CONTEXT_FEATURES = [
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
    "month",
]
FEATURE_SETS = {
    "all": (CATEGORICAL, NUMERIC),
    "client": (
        [c for c in CATEGORICAL if c not in CONTEXT_FEATURES],
        [c for c in NUMERIC if c not in CONTEXT_FEATURES],
    ),
}


def input_columns(feature_set: str = "all") -> list[str]:
    categorical, numeric = FEATURE_SETS[feature_set]
    return [*categorical, *numeric, "pdays"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features. ``pdays == 999`` is a sentinel for "never contacted", not a number."""
    out = df.copy()
    out["previously_contacted"] = (out["pdays"] != NEVER_CONTACTED).astype(int)
    out["days_since_contact"] = out["pdays"].where(out["previously_contacted"] == 1, np.nan)
    out["log_campaign"] = np.log1p(out["campaign"])
    return out


def build_preprocessor(scale: bool = True, feature_set: str = "all") -> Pipeline:
    categorical, numeric = FEATURE_SETS[feature_set]
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    columns = ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric + ENGINEERED),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [("engineer", FunctionTransformer(add_features, validate=False)), ("columns", columns)]
    )
