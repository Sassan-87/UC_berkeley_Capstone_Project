"""SECOM early-warning API: scores production runs with the alarm model saved by the notebook.

Start it from the repository folder:   python -m fastapi dev app/api.py
Interactive documentation:             http://127.0.0.1:8000/docs
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# load the model once, when the server starts (random forest + 0.17 alarm threshold, notebook Section 11)
model = joblib.load(Path(__file__).parent / "secom_alarm_model.joblib")
FEATURES = list(model.feature_names_in_)   # the 274 inputs, in the order the model was trained on

app = FastAPI(title="SECOM early-warning API",
              description="Gives each production run a fail probability and raises an alarm above the threshold.")


class Runs(BaseModel):
    """Request body: a list of runs, each one a dictionary of feature name -> reading."""
    runs: list[dict[str, float | None]]                 # None (null in JSON) = missing reading
    threshold: float | None = Field(None, ge=0, le=1)  # optional, default is the model's own threshold


@app.get("/")
def model_info():
    """Health check: is the model loaded, and which inputs does it expect?"""
    return {"status": "ok",
            "model": "random forest with a fixed alarm threshold",
            "threshold": float(model.threshold),
            "features": FEATURES}


@app.post("/predict")
def predict(request: Runs):
    """Score every run: fail probability and alarm yes/no."""
    runs = pd.DataFrame(request.runs)
    missing = [f for f in FEATURES if f not in runs.columns]
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"{len(missing)} features are missing, for example {missing[:5]}")

    threshold = float(model.threshold) if request.threshold is None else request.threshold
    # missing readings become NaN and are filled by the model's own median imputer, as in the notebook
    probabilities = model.predict_proba(runs[FEATURES].astype(float))[:, 1]
    return {"threshold": threshold,
            "results": [{"fail_probability": round(float(p), 4), "alarm": bool(p >= threshold)}
                        for p in probabilities]}
