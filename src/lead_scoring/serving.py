"""FastAPI scoring service over the saved model and its model card."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

Priority = Literal["call_first", "call_this_week", "nurture"]


class Lead(BaseModel):
    """One prospect, described with the information available *before* the call."""

    age: int = Field(ge=16, le=110)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: Literal["cellular", "telephone"]
    month: str
    day_of_week: str
    campaign: int = Field(ge=1, description="Contacts in this campaign, including this one")
    pdays: int = Field(ge=0, description="Days since last contact; 999 = never contacted")
    previous: int = Field(ge=0)
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


class Score(BaseModel):
    probability: float
    priority: Priority


class ScoreRequest(BaseModel):
    leads: list[Lead] = Field(min_length=1, max_length=1000)


class ScoreResponse(BaseModel):
    model: str
    scores: list[Score]


class LoadedModel:
    def __init__(self, model_dir: Path) -> None:
        if not (model_dir / "model.joblib").exists():
            raise FileNotFoundError(f"No model in {model_dir}; run `lead-scoring train` first")
        self.model = joblib.load(model_dir / "model.joblib")
        self.card: dict[str, Any] = json.loads((model_dir / "model_card.json").read_text())

    def score(self, leads: list[Lead]) -> list[Score]:
        frame = pd.DataFrame([lead.model_dump() for lead in leads])[self.card["input_columns"]]
        probabilities = self.model.predict_proba(frame)[:, 1]
        thresholds = self.card["priority_thresholds"]
        return [
            Score(probability=round(float(p), 4), priority=_priority(p, thresholds))
            for p in probabilities
        ]


def _priority(p: float, thresholds: dict[str, float]) -> Priority:
    if p >= thresholds["top_10"]:
        return "call_first"
    if p >= thresholds["top_30"]:
        return "call_this_week"
    return "nurture"


@lru_cache
def get_model(model_dir: str) -> LoadedModel:
    return LoadedModel(Path(model_dir))


def create_app(model_dir: Path | None = None) -> FastAPI:
    directory = (
        str(model_dir) if model_dir else os.getenv("LEAD_SCORING_MODEL_DIR", "artifacts/model")
    )
    app = FastAPI(title="lead-scoring-ml", version="0.1.0")

    def model() -> LoadedModel:
        try:
            return get_model(directory)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model().card["model"]}

    @app.get("/model")
    def model_card() -> dict[str, Any]:
        return model().card

    @app.post("/score")
    def score(body: ScoreRequest) -> ScoreResponse:
        loaded = model()
        return ScoreResponse(model=loaded.card["model"], scores=loaded.score(body.leads))

    return app
