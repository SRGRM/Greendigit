"""
dirac_sim.core.site_model
=========================
Computing site definitions with time-varying energy and carbon signals.

Signals are expected at 15-minute resolution, matching the challenge dataset.
Sites can be loaded from a JSON config and augmented with live forecast values
retrieved via the REST forecast client.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SiteCapacity:
    """Resource limits for a computing site."""
    max_cores: int = 1000
    max_memory_gb: float = 4000.0
    max_concurrent_jobs: int = 500
    current_jobs: int = 0
    current_cores: int = 0
    current_memory_gb: float = 0.0

    @property
    def available_slots(self) -> int:
        return max(0, self.max_concurrent_jobs - self.current_jobs)

    @property
    def available_cores(self) -> int:
        return max(0, self.max_cores - self.current_cores)

    @property
    def available_memory_gb(self) -> float:
        return max(0.0, self.max_memory_gb - self.current_memory_gb)


@dataclass
class Site:
    """
    One computing site (e.g. a WLCG Tier-1/Tier-2 or an HPC cluster).

    Energy and carbon timeseries are stored as dicts keyed by UTC bucket
    timestamps rounded to 15-minute boundaries.

    Parameters
    ----------
    site_id         : Unique identifier (matches challenge series_id convention).
    name            : Human-readable name.
    location        : ISO country or region code (e.g. "DE", "FR", "US-WEST").
    capacity        : Resource limits.
    energy_wh       : Dict[timestamp_str -> Wh per 15-min bucket].
    cfp_g           : Dict[timestamp_str -> gCO2eq per 15-min bucket].
    pue             : Power Usage Effectiveness multiplier (default 1.4).
    dispatch_latency: Minutes between dispatch decision and job start.
    """
    site_id: str
    name: str
    location: str = "UNKNOWN"
    capacity: SiteCapacity = field(default_factory=SiteCapacity)
    energy_wh: Dict[str, float] = field(default_factory=dict)
    cfp_g: Dict[str, float] = field(default_factory=dict)
    pue: float = 1.4
    dispatch_latency: float = 2.0   # minutes
    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Signal access                                                        #
    # ------------------------------------------------------------------ #
    def get_energy(self, t: datetime, fallback: float = 1000.0) -> float:
        """Return Wh for the 15-min bucket containing *t*."""
        key = _bucket_key(t)
        return self.energy_wh.get(key, fallback) * self.pue

    def get_carbon(self, t: datetime, fallback: float = 200.0) -> float:
        """Return gCO2eq for the 15-min bucket containing *t*."""
        key = _bucket_key(t)
        return self.cfp_g.get(key, fallback)

    def update_signals(self, bucket_ts: datetime,
                       energy_wh: float, cfp_g: float) -> None:
        """Ingest one 15-min signal bucket (called by ForecastClient)."""
        key = _bucket_key(bucket_ts)
        self.energy_wh[key] = energy_wh
        self.cfp_g[key] = cfp_g

    def bulk_update(self, records: List[Dict]) -> None:
        """
        Bulk-ingest forecast records.

        Each record must have keys:
          forecast_timestamp_utc, energy_wh_pred, cfp_g_pred
        """
        for r in records:
            ts = _parse_dt(r["forecast_timestamp_utc"])
            self.update_signals(ts,
                                float(r["energy_wh_pred"]),
                                float(r["cfp_g_pred"]))

    # ------------------------------------------------------------------ #
    # Window helpers (used by carbon-aware schedulers)                    #
    # ------------------------------------------------------------------ #
    def green_windows(self, start: datetime, horizon_minutes: int,
                      carbon_threshold: float) -> List[Tuple[datetime, datetime]]:
        """
        Return contiguous time windows within *horizon_minutes* where
        carbon intensity is below *carbon_threshold* gCO2eq.
        """
        windows: List[Tuple[datetime, datetime]] = []
        current_start: Optional[datetime] = None
        t = _bucket_start(start)
        end = start + timedelta(minutes=horizon_minutes)

        while t < end:
            carbon = self.get_carbon(t)
            if carbon < carbon_threshold:
                if current_start is None:
                    current_start = t
            else:
                if current_start is not None:
                    windows.append((current_start, t))
                    current_start = None
            t += timedelta(minutes=15)

        if current_start is not None:
            windows.append((current_start, t))
        return windows

    def to_dict(self) -> Dict:
        return {
            "site_id": self.site_id,
            "name": self.name,
            "location": self.location,
            "pue": self.pue,
            "dispatch_latency": self.dispatch_latency,
            "capacity": {
                "max_cores": self.capacity.max_cores,
                "max_memory_gb": self.capacity.max_memory_gb,
                "max_concurrent_jobs": self.capacity.max_concurrent_jobs,
            },
            "metadata": self.metadata,
        }


class SiteRegistry:
    """Collection of all known computing sites."""

    def __init__(self) -> None:
        self._sites: Dict[str, Site] = {}

    def register(self, site: Site) -> None:
        self._sites[site.site_id] = site

    def get(self, site_id: str) -> Optional[Site]:
        return self._sites.get(site_id)

    def all_sites(self) -> List[Site]:
        return list(self._sites.values())

    def available_sites(self, job_whitelist: List[str]) -> List[Site]:
        """Sites that are in the job's whitelist (or all if whitelist empty)."""
        if not job_whitelist:
            return self.all_sites()
        return [s for s in self._sites.values() if s.site_id in job_whitelist]

    def cheapest_carbon_site(self, t: datetime,
                             candidates: List[Site]) -> Optional[Site]:
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.get_carbon(t))

    def cheapest_energy_site(self, t: datetime,
                             candidates: List[Site]) -> Optional[Site]:
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.get_energy(t))

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, path: str | Path) -> "SiteRegistry":
        registry = cls()
        with open(path) as fh:
            data = json.load(fh)
        for entry in data["sites"]:
            cap = SiteCapacity(**entry.pop("capacity", {}))
            site = Site(**entry, capacity=cap)
            registry.register(site)
        return registry

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            json.dump({"sites": [s.to_dict() for s in self.all_sites()]},
                      fh, indent=2)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #
def _bucket_start(t: datetime) -> datetime:
    """Floor t to 15-minute boundary."""
    minutes = (t.minute // 15) * 15
    return t.replace(minute=minutes, second=0, microsecond=0)


def _bucket_key(t: datetime) -> str:
    return _bucket_start(t).strftime("%Y-%m-%dT%H:%M:00+00:00")


def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
