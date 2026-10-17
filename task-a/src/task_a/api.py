from __future__ import annotations

import os
import time
from datetime import timedelta

from .dataio import default_data_path, load_series_csv
from .models.baseline import BaselineForecaster
from .schemas import parse_timestamp

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    FastAPI = None
    BaseModel = object
    def Field(default=None, **_):
        return default


class ForecastRequest(BaseModel):
    series_ids: list[str] = Field(...)
    reference_timestamp_utc: str
    horizons: list[str] = ["1h", "24h"]


def _horizon_steps(label: str) -> int:
    if label == "1h":
        return 4
    if label == "24h":
        return 96
    if label.endswith("m"):
        return int(label[:-1]) // 15
    return int(label)


def create_app(model: BaselineForecaster | None = None):
    if FastAPI is None:
        raise ImportError("FastAPI is required for the Task A forecast server")
    if model is None:
        model_path = os.environ.get("TASK_A_MODEL")
        if model_path:
            model = BaselineForecaster.load(model_path)
        else:
            model = BaselineForecaster.fit(load_series_csv(default_data_path()))

    app = FastAPI(title="Task A Forecast API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True, "version": "0.1.0"}

    @app.post("/forecast")
    def forecast(req: ForecastRequest):
        started = time.perf_counter()
        origin = parse_timestamp(req.reference_timestamp_utc)
        records = []
        for series_id in req.series_ids:
            for label in req.horizons:
                steps = _horizon_steps(label)
                forecast_ts = origin + timedelta(minutes=15 * steps)
                energy, cfp = model.predict_one(series_id, forecast_ts)
                records.append({
                    "series_id": series_id,
                    "forecast_timestamp_utc": forecast_ts.isoformat(),
                    "horizon_steps_15m": steps,
                    "energy_wh_pred": energy,
                    "cfp_g_pred": cfp,
                })
        return {
            "reference_timestamp_utc": origin.isoformat(),
            "forecasts": records,
            "model_version": "task-a-baseline-0.1.0",
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    return app


app = create_app()
