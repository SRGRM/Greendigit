"""
dirac_sim.api.forecast_client
==============================
HTTP client that fetches forecasts from a participant's model server.

The client speaks the Task A REST schema, so any model that can answer
POST /forecast is compatible — no special integration needed.

Supports:
  * Synchronous blocking fetch (default)
  * Async fetch (asyncio) for integration with async WMS backends
  * Retry with exponential back-off
  * Local offline mode (reads a pre-generated CSV instead of calling the API)
  * Request/response logging for debugging

Environment variables
---------------------
FORECAST_API_BASE_URL   Override base_url at runtime (useful in containers).
FORECAST_API_KEY        Bearer token if the server requires auth.
FORECAST_API_TIMEOUT    Per-request timeout in seconds (default 10).
"""
from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from dirac_sim.core.scheduler import ForecastBundle
from dirac_sim.api.schemas import ForecastRecord, ForecastResponse

logger = logging.getLogger(__name__)


class ForecastClient:
    """
    Thin HTTP wrapper around the participant's forecast REST server.

    The server must expose at minimum:

        POST /forecast
        Body : ForecastRequest  (see dirac_sim.api.schemas)
        Reply: ForecastResponse

    Parameters
    ----------
    base_url       : Root URL of the forecast server, e.g.
                     "http://localhost:8000".
    series_ids     : Series IDs to request forecasts for (challenge series).
    api_key        : Optional Bearer token.
    timeout        : Per-request timeout in seconds.
    max_retries    : Number of retry attempts on transient errors.
    offline_csv    : Path to a CSV fallback (see data/ directory).
                     If set, the client reads from CSV instead of HTTP.
    """

    FORECAST_ENDPOINT = "/forecast"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        series_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        offline_csv: Optional[str] = None,
    ) -> None:
        self.base_url = os.environ.get("FORECAST_API_BASE_URL", base_url).rstrip("/")
        self.series_ids = series_ids or []
        self.api_key = api_key or os.environ.get("FORECAST_API_KEY")
        self.timeout = float(os.environ.get("FORECAST_API_TIMEOUT", timeout))
        self.max_retries = max_retries
        self.offline_csv = offline_csv
        self._offline_cache: Optional[Dict] = None   # lazy-loaded

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def fetch_bundle(self, now: datetime) -> ForecastBundle:
        """
        Fetch 1h-ahead and 24h-ahead forecasts for *now*.

        Falls back to offline CSV if configured or on HTTP error.
        """
        if self.offline_csv:
            return self._fetch_offline(now)

        if not _HAS_REQUESTS:
            logger.warning("requests not installed; using empty bundle")
            return ForecastBundle(tick_time=now)

        return self._fetch_http(now)

    # ------------------------------------------------------------------ #
    # HTTP fetch                                                           #
    # ------------------------------------------------------------------ #
    def _fetch_http(self, now: datetime) -> ForecastBundle:
        import requests as req

        payload = {
            "series_ids": self.series_ids,
            "reference_timestamp_utc": now.isoformat(),
            "horizons": ["1h", "24h"],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.base_url + self.FORECAST_ENDPOINT
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = req.post(url, json=payload, headers=headers,
                                timeout=self.timeout)
                latency_ms = (time.perf_counter() - t0) * 1000
                resp.raise_for_status()
                data = resp.json()
                logger.debug("Forecast fetched in %.1f ms", latency_ms)
                return self._parse_response(now, data)
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("Forecast API attempt %d/%d failed: %s "
                               "(retry in %ds)", attempt, self.max_retries,
                               exc, wait)
                if attempt < self.max_retries:
                    time.sleep(wait)

        logger.error("All forecast API attempts failed: %s", last_exc)
        return ForecastBundle(tick_time=now)

    # ------------------------------------------------------------------ #
    # Offline CSV fetch                                                    #
    # ------------------------------------------------------------------ #
    def _fetch_offline(self, now: datetime) -> ForecastBundle:
        if self._offline_cache is None:
            self._offline_cache = self._load_csv(self.offline_csv)
        return self._slice_cache(now)

    def _load_csv(self, path: str) -> Dict[str, List[Dict]]:
        """Load full forecast CSV into memory keyed by reference_ts bucket."""
        cache: Dict[str, List[Dict]] = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                # key = forecast_timestamp_utc rounded to 15-min bucket
                key = row.get("forecast_timestamp_utc", "")[:16]
                cache.setdefault(key, []).append(row)
        logger.info("Loaded offline forecast CSV: %d buckets", len(cache))
        return cache

    def _slice_cache(self, now: datetime) -> ForecastBundle:
        """Return local records for current, 1h, and 24h buckets."""
        from datetime import timedelta
        ts_now = now
        ts_1h = now + timedelta(hours=1)
        ts_24h = now + timedelta(hours=24)

        key_now = ts_now.strftime("%Y-%m-%dT%H:%M")
        key_1h = ts_1h.strftime("%Y-%m-%dT%H:%M")
        key_24h = ts_24h.strftime("%Y-%m-%dT%H:%M")

        h1 = [
            *self._offline_cache.get(key_now, []),
            *self._offline_cache.get(key_1h, []),
        ]
        h24 = self._offline_cache.get(key_24h, [])
        return ForecastBundle(tick_time=now, horizon_1h=h1, horizon_24h=h24)

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(now: datetime, data: dict) -> ForecastBundle:
        forecasts = data.get("forecasts", [])
        h1 = [f for f in forecasts if f.get("horizon_steps_15m", 0) == 4]
        h24 = [f for f in forecasts if f.get("horizon_steps_15m", 0) == 96]
        return ForecastBundle(
            tick_time=now,
            horizon_1h=h1,
            horizon_24h=h24,
            raw_response=data,
        )

    # ------------------------------------------------------------------ #
    # Async variant (optional, requires httpx)                            #
    # ------------------------------------------------------------------ #
    async def fetch_bundle_async(self, now: datetime) -> ForecastBundle:
        """
        Async version for use with async WMS backends.
        Requires `httpx` (pip install httpx).
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for async mode: "
                              "pip install httpx")

        payload = {
            "series_ids": self.series_ids,
            "reference_timestamp_utc": now.isoformat(),
            "horizons": ["1h", "24h"],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.base_url + self.FORECAST_ENDPOINT
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return self._parse_response(now, resp.json())
