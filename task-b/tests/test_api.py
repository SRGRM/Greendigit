"""
tests/test_api.py
=================
Tests for the REST API schemas and forecast client (offline mode).
Run with: pytest tests/ -v
"""
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dirac_sim.api.schemas import (
    ForecastRequest, ForecastRecord, ForecastResponse,
    ScheduleRequest, ScheduleResponse, HealthResponse, JobSpec,
)
from dirac_sim.api.forecast_client import ForecastClient
from dirac_sim.core.scheduler import ForecastBundle

T0 = datetime(2025, 11, 20, 0, 0, tzinfo=timezone.utc)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ------------------------------------------------------------------ #
# Schema tests                                                         #
# ------------------------------------------------------------------ #
class TestSchemas:
    def test_forecast_request_valid(self):
        req = ForecastRequest(
            series_ids=["CE1", "CE2"],
            reference_timestamp_utc=T0,
            horizons=["1h", "24h"],
        )
        assert len(req.series_ids) == 2

    def test_forecast_record_roundtrip(self):
        rec = ForecastRecord(
            series_id="CE1",
            forecast_timestamp_utc=T0 + timedelta(hours=1),
            horizon_steps_15m=4,
            energy_wh_pred=450.5,
            cfp_g_pred=123.0,
        )
        d = rec.model_dump()
        rec2 = ForecastRecord(**d)
        assert rec2.energy_wh_pred == pytest.approx(450.5)

    def test_forecast_response_serialise(self):
        resp = ForecastResponse(
            reference_timestamp_utc=T0,
            forecasts=[
                ForecastRecord(
                    series_id="CE1",
                    forecast_timestamp_utc=T0 + timedelta(hours=1),
                    horizon_steps_15m=4,
                    energy_wh_pred=400.0,
                    cfp_g_pred=80.0,
                )
            ],
        )
        j = resp.model_dump_json()
        assert "CE1" in j

    def test_schedule_request_valid(self):
        req = ScheduleRequest(
            jobs=[JobSpec(
                job_id="j1",
                series_id="CE1",
                arrival_time=T0,
                deadline=T0 + timedelta(hours=12),
                cpu_minutes=60.0,
            )],
            current_time_utc=T0,
            declared_objective="carbon",
        )
        assert req.jobs[0].job_id == "j1"

    def test_health_response(self):
        h = HealthResponse(status="ok", model_loaded=True)
        assert h.status == "ok"


# ------------------------------------------------------------------ #
# Forecast client (offline mode)                                      #
# ------------------------------------------------------------------ #
class TestForecastClientOffline:
    @pytest.fixture
    def client(self):
        csv_path = DATA_DIR / "forecast_baseline.csv"
        if not csv_path.exists():
            pytest.skip("forecast_baseline.csv not prepared; run "
                        "python3 -m dirac_sim prepare-data first")
        return ForecastClient(
            series_ids=[
                "site_ad3433ffac",
                "site_e726c7cce5",
                "site_f5bcf8a88a",
            ],
            offline_csv=str(csv_path),
        )

    def test_fetch_bundle_returns_bundle(self, client):
        bundle = client.fetch_bundle(T0)
        assert isinstance(bundle, ForecastBundle)
        assert bundle.tick_time == T0

    def test_1h_records_present(self, client):
        bundle = client.fetch_bundle(T0)
        assert len(bundle.horizon_1h) > 0

    def test_24h_records_present(self, client):
        bundle = client.fetch_bundle(T0)
        assert len(bundle.horizon_24h) > 0

    def test_records_have_required_keys(self, client):
        bundle = client.fetch_bundle(T0)
        required = {"series_id", "forecast_timestamp_utc",
                    "energy_wh_pred", "cfp_g_pred"}
        for rec in bundle.horizon_1h:
            assert required.issubset(rec.keys())

    def test_by_series_all_sites(self, client):
        bundle = client.fetch_bundle(T0)
        by_series = bundle.by_series("1h")
        # Should have records for at least one site
        assert len(by_series) >= 1

    def test_cache_reuse(self, client):
        """Second fetch should use cache (no file re-read)."""
        bundle1 = client.fetch_bundle(T0)
        bundle2 = client.fetch_bundle(T0)
        # Both return same number of records
        assert len(bundle1.horizon_1h) == len(bundle2.horizon_1h)

    def test_different_ticks_different_bundles(self, client):
        bundle1 = client.fetch_bundle(T0)
        bundle2 = client.fetch_bundle(T0 + timedelta(hours=6))
        # Forecast timestamps should differ
        if bundle1.horizon_24h and bundle2.horizon_24h:
            ts1 = bundle1.horizon_24h[0]["forecast_timestamp_utc"]
            ts2 = bundle2.horizon_24h[0]["forecast_timestamp_utc"]
            assert ts1 != ts2


# ------------------------------------------------------------------ #
# FastAPI app tests (optional — requires fastapi + httpx)             #
# ------------------------------------------------------------------ #
try:
    from fastapi.testclient import TestClient
    from dirac_sim.api.server import create_app
    _HAS_TESTCLIENT = True
except ImportError:
    _HAS_TESTCLIENT = False


@pytest.mark.skipif(not _HAS_TESTCLIENT,
                    reason="fastapi or httpx not installed")
class TestFastAPIServer:
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_forecast_endpoint(self, client):
        payload = {
            "series_ids": ["CE1", "CE2"],
            "reference_timestamp_utc": T0.isoformat(),
            "horizons": ["1h", "24h"],
        }
        resp = client.post("/forecast", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "forecasts" in data
        assert len(data["forecasts"]) == 4  # 2 series x 2 horizons

    def test_forecast_response_schema(self, client):
        payload = {
            "series_ids": ["CE1"],
            "reference_timestamp_utc": T0.isoformat(),
            "horizons": ["1h"],
        }
        resp = client.post("/forecast", json=payload)
        assert resp.status_code == 200
        rec = resp.json()["forecasts"][0]
        assert "energy_wh_pred" in rec
        assert "cfp_g_pred" in rec
        assert rec["horizon_steps_15m"] == 4

    def test_metrics_endpoint(self, client):
        # Hit forecast first to increment counter
        client.post("/forecast", json={
            "series_ids": ["CE1"],
            "reference_timestamp_utc": T0.isoformat(),
            "horizons": ["1h"],
        })
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "dirac_sim_forecast_total" in resp.text
