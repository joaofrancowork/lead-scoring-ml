"""Download, load and split the UCI Bank Marketing dataset.

Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5K306 (CC BY 4.0).

Each row is a phone contact in a direct-marketing campaign; the target is whether
the client subscribed. It is a lead-scoring problem in all but name: decide who
to call first with a limited sales team.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
INNER_ZIP = "bank-additional.zip"
CSV_NAME = "bank-additional/bank-additional-full.csv"
DEFAULT_PATH = Path("data/raw/bank-additional-full.csv")
TARGET = "converted"

# Known only after the call ends, so it cannot be used to decide whom to call.
LEAKY_COLUMNS = ["duration"]


def download(path: Path = DEFAULT_PATH, url: str = URL) -> Path:
    if path.exists():
        return path
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(inner.read(CSV_NAME))
    return path


def load(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df[TARGET] = (df.pop("y") == "yes").astype(int)
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    # Rows are in chronological order (May 2008 - Nov 2010); keep that as an explicit column.
    df.insert(0, "contact_order", range(len(df)))
    return df


def time_split(
    df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on the past, validate on the near future, test on the latest contacts.

    A random split leaks future economic conditions into training and overstates
    how well the model will do next quarter.
    """
    df = df.sort_values("contact_order")
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]


def random_split(
    df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df[TARGET], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_frac / (1 - test_frac),
        stratify=train_val[TARGET],
        random_state=seed,
    )
    return train, val, test
