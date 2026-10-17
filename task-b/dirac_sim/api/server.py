"""
dirac_sim.api.server
====================
FastAPI application that wraps a participant's forecasting model and exposes
it as a REST service.  This is the *reference server* provided with the
starter kit.  Participants replace `_load_model()` and `_predict()` with
their own Task A model.

Endpoints
---------
GET  /health          – liveness probe
POST /forecast        – produce energy_wh / cfp_g forecasts
POST /schedule        – (optional) combined forecast+schedule in one call
GET  /metrics         – Prometheus-compatible counters

Running the server
------------------
    uvicorn dirac_sim.api.server:app --host 0.0.0.0 --port 8000 --reload

Environment variables
---------------------
MODEL_PATH   : Path to serialised model artefact (pickle, ONNX, etc.)
SITES_JSON   : Path to site_config.json
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
DEFAULT_SITES_JSON = (
    Path(__file__).resolve().parents[3] / "data" / "site_config.json"
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    # Provide a stub so the module can be imported without FastAPI
    class FastAPI:
        def get(self, *a, **k): return lambda f: f
        def post(self, *a, **k): return lambda f: f
        def add_middleware(self, *a, **k): pass

from dirac_sim.api.schemas import (
    ForecastRequest, ForecastResponse, ForecastRecord,
    ScheduleRequest, ScheduleResponse, DispatchDecisionOut,
    HealthResponse,
)


# ------------------------------------------------------------------ #
# App factory                                                          #
# ------------------------------------------------------------------ #
def create_app(model=None) -> "FastAPI":
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI not installed. "
                          "Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="DiracSim Forecast & Scheduling API",
        description=(
            "Reference REST server for ECML PKDD Challenge Task B. "
            "Replace _predict() with your own Task A model."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State bag
    state: Dict[str, Any] = {
        "model": model,
        "model_loaded": model is not None,
        "last_forecast_at": None,
        "forecast_count": 0,
        "schedule_count": 0,
    }

    # ---------------------------------------------------------------- #
    # Model loading (participant replaces this)                         #
    # ---------------------------------------------------------------- #
    def _load_model() -> Any:
        """
        Load participant's Task A model.

        Replace this with your own loading logic:
            import pickle
            with open(os.environ["MODEL_PATH"], "rb") as f:
                return pickle.load(f)
        """
        logger.warning("No model loaded — returning dummy forecasts.")
        return None

    def _predict(model: Any, series_id: str,
                 reference_ts: datetime, horizon_steps: int) -> tuple:
        """
        Call participant's model to produce (energy_wh_pred, cfp_g_pred).

        Replace the dummy below with real model inference:
            features = build_features(series_id, reference_ts, horizon_steps)
            energy, carbon = model.predict(features)
            return float(energy), float(carbon)
        """
        # Dummy: flat persistence with slight horizon decay
        decay = 1.0 - 0.01 * horizon_steps
        return 500.0 * decay, 120.0 * decay

    if state["model"] is None:
        model_path = os.environ.get("MODEL_PATH")
        if model_path:
            state["model"] = _load_model()
            state["model_loaded"] = state["model"] is not None

    # ---------------------------------------------------------------- #
    # Routes                                                            #
    # ---------------------------------------------------------------- #
    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model_loaded=state["model_loaded"],
            last_forecast_at=state["last_forecast_at"],
        )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(req: ForecastRequest):
        t0 = time.perf_counter()
        records = []

        for series_id in req.series_ids:
            for horizon_label in req.horizons:
                steps = 4 if horizon_label == "1h" else 96
                forecast_ts = req.reference_timestamp_utc + timedelta(
                    minutes=15 * steps)

                energy_pred, cfp_pred = _predict(
                    state["model"], series_id,
                    req.reference_timestamp_utc, steps)

                records.append(ForecastRecord(
                    series_id=series_id,
                    forecast_timestamp_utc=forecast_ts,
                    horizon_steps_15m=steps,
                    energy_wh_pred=energy_pred,
                    cfp_g_pred=cfp_pred,
                ))

        state["last_forecast_at"] = datetime.now(timezone.utc)
        state["forecast_count"] += 1
        latency_ms = (time.perf_counter() - t0) * 1000

        return ForecastResponse(
            reference_timestamp_utc=req.reference_timestamp_utc,
            forecasts=records,
            model_version=os.environ.get("MODEL_VERSION", "0.1.0"),
            latency_ms=latency_ms,
        )

    @app.post("/schedule", response_model=ScheduleResponse)
    def schedule(req: ScheduleRequest):
        """
        Combined forecast + schedule endpoint.

        This convenience endpoint is useful for real DIRAC/SLURM integrations
        where the infrastructure sends jobs directly to the REST API for
        immediate scheduling decisions, without running the full simulator.
        """
        state["schedule_count"] += 1
        now = req.current_time_utc

        # Lazy import to avoid circular dependency
        from dirac_sim.core.job_queue import Job, JobQueue
        from dirac_sim.core.site_model import SiteRegistry
        from dirac_sim.core.scheduler import ForecastBundle
        from dirac_sim.baselines.greedy_carbon import GreedyCarbonScheduler

        queue = JobQueue()
        for js in req.jobs:
            queue.add(Job.from_dict(js.dict()))

        sites_path = os.environ.get("SITES_JSON", str(DEFAULT_SITES_JSON))
        try:
            registry = SiteRegistry.from_json(sites_path)
        except FileNotFoundError:
            raise HTTPException(status_code=500,
                                detail=f"Site config not found: {sites_path}")

        # Build a mini forecast bundle from the model
        bundle_records = []
        for series_id in {j.series_id for j in queue.all_jobs()}:
            for steps in (4, 96):
                forecast_ts = now + timedelta(minutes=15 * steps)
                energy_pred, cfp_pred = _predict(
                    state["model"], series_id, now, steps)
                bundle_records.append({
                    "series_id": series_id,
                    "forecast_timestamp_utc": forecast_ts.isoformat(),
                    "horizon_steps_15m": steps,
                    "energy_wh_pred": energy_pred,
                    "cfp_g_pred": cfp_pred,
                })

        h1 = [r for r in bundle_records if r["horizon_steps_15m"] == 4]
        h24 = [r for r in bundle_records if r["horizon_steps_15m"] == 96]
        bundle = ForecastBundle(tick_time=now, horizon_1h=h1, horizon_24h=h24)

        scheduler = GreedyCarbonScheduler(
            declared_objective=req.declared_objective)
        plan = scheduler.schedule(queue, registry, bundle, now)

        decisions_out = [
            DispatchDecisionOut(
                job_id=d.job_id,
                site_id=d.site_id,
                dispatch_at=d.dispatch_at,
                rationale=d.rationale,
            )
            for d in plan.decisions
        ]
        n_held = sum(1 for d in plan.decisions if d.dispatch_at > now)
        return ScheduleResponse(
            current_time_utc=now,
            decisions=decisions_out,
            declared_objective=req.declared_objective,
            n_held=n_held,
        )

    @app.get("/metrics")
    def metrics():
        """Prometheus-compatible plain-text metrics."""
        lines = [
            "# HELP dirac_sim_forecast_total Total forecast requests",
            "# TYPE dirac_sim_forecast_total counter",
            f"dirac_sim_forecast_total {state['forecast_count']}",
            "# HELP dirac_sim_schedule_total Total schedule requests",
            "# TYPE dirac_sim_schedule_total counter",
            f"dirac_sim_schedule_total {state['schedule_count']}",
        ]
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("\n".join(lines))

    return app


# Module-level app instance (used by uvicorn)
app = create_app()


if __name__ == "__main__":
    if not _HAS_FASTAPI:
        raise SystemExit("Install FastAPI: pip install fastapi uvicorn")
    uvicorn.run("dirac_sim.api.server:app",
                host="0.0.0.0", port=8000, reload=True)
