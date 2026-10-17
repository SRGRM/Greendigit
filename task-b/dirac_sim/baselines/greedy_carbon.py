"""
dirac_sim.baselines.greedy_carbon
==================================
Greedy carbon-aware scheduler — provided as a non-trivial baseline and
as an example of how to consume the forecast REST API inside a scheduler.

Strategy
--------
For each ready job at tick *t*:
  1. Fetch 1h-ahead forecast to find the lowest-carbon site *right now*.
  2. Fetch 24h-ahead forecast to check if a significantly greener window
     exists within `max_hold_hours`.
  3. If a future window saves ≥ `carbon_saving_threshold` %  AND the job
     has enough deadline slack, HOLD the job until that window.
  4. Otherwise dispatch immediately to the lowest-carbon site.

This implements the minimal forecast-to-dispatch loop described in the
challenge spec.  Participants are expected to go beyond this with learned
models, multi-objective optimisation, etc.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from dirac_sim.core.job_queue import Job, JobQueue
from dirac_sim.core.scheduler import (
    DispatchDecision, DispatchPlan, ForecastBundle, Objective, Scheduler,
)
from dirac_sim.core.site_model import Site, SiteRegistry

logger = logging.getLogger(__name__)


class GreedyCarbonScheduler(Scheduler):
    """
    Greedy carbon-aware scheduler.

    Parameters
    ----------
    declared_objective     : Primary objective (default: CARBON).
    max_hold_hours         : Max hours to defer a job for a greener window.
    carbon_saving_threshold: Minimum fractional carbon saving to justify hold.
    min_deadline_slack_h   : Minimum deadline slack (hours) required to hold.
    """

    def __init__(
        self,
        declared_objective: str = "carbon",
        max_hold_hours: float = 6.0,
        carbon_saving_threshold: float = 0.15,
        min_deadline_slack_h: float = 2.0,
    ) -> None:
        obj = Objective(declared_objective) if isinstance(
            declared_objective, str) else declared_objective
        super().__init__(declared_objective=obj)
        self.max_hold_hours = max_hold_hours
        self.carbon_saving_threshold = carbon_saving_threshold
        self.min_deadline_slack_h = min_deadline_slack_h

        # Cache current carbon forecasts by series_id
        self._carbon_now: Dict[str, float] = {}
        self._carbon_future: Dict[str, List[tuple]] = {}  # series_id -> [(ts, cfp)]

    # ------------------------------------------------------------------ #
    # Hooks                                                                #
    # ------------------------------------------------------------------ #
    def on_forecast_received(self, bundle: ForecastBundle) -> None:
        """Pre-process forecast bundle into carbon lookup tables."""
        # 1h-ahead carbon by series_id
        for rec in bundle.horizon_1h:
            sid = rec.get("series_id", "")
            self._carbon_now[sid] = float(rec.get("cfp_g_pred", 9999))

        # 24h-ahead sorted timeline by series_id
        by_series: Dict[str, List] = {}
        for rec in bundle.horizon_24h:
            sid = rec.get("series_id", "")
            ts_str = rec.get("forecast_timestamp_utc", "")
            try:
                ts = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            cfp = float(rec.get("cfp_g_pred", 9999))
            by_series.setdefault(sid, []).append((ts, cfp))

        for sid, entries in by_series.items():
            self._carbon_future[sid] = sorted(entries, key=lambda x: x[0])

    # ------------------------------------------------------------------ #
    # Core schedule()                                                      #
    # ------------------------------------------------------------------ #
    def schedule(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        forecast: ForecastBundle,
        now: datetime,
    ) -> DispatchPlan:
        plan = DispatchPlan(declared_objective=self.declared_objective)

        for job in queue.ready_jobs(now):
            decision = self._decide(job, registry, now)
            if decision:
                plan.add(decision)

        return plan

    # ------------------------------------------------------------------ #
    # Per-job decision logic                                               #
    # ------------------------------------------------------------------ #
    def _decide(self, job: Job, registry: SiteRegistry,
                now: datetime) -> Optional[DispatchDecision]:
        candidates = [s for s in registry.available_sites(job.site_whitelist)
                      if s.capacity.available_slots > 0]
        if not candidates:
            return None

        # Best site right now (lowest carbon)
        best_now = min(candidates,
                       key=lambda s: self._carbon_now.get(s.site_id, 9999))
        carbon_now = self._carbon_now.get(best_now.site_id, 9999)

        # Check for greener future window (24h forecast)
        slack_h = job.deadline_slack_minutes(now) / 60.0
        if slack_h >= self.min_deadline_slack_h:
            future = self._find_green_window(
                job, candidates, now, carbon_now)
            if future is not None:
                future_ts, future_site, future_carbon = future
                saving = (carbon_now - future_carbon) / (carbon_now + 1e-9)
                if saving >= self.carbon_saving_threshold:
                    logger.debug(
                        "Job %s: holding until %s (save %.1f%% carbon)",
                        job.job_id, future_ts.isoformat(), saving * 100)
                    return DispatchDecision(
                        job_id=job.job_id,
                        site_id=future_site.site_id,
                        dispatch_at=future_ts,
                        rationale=(
                            f"greedy_carbon: deferred to green window "
                            f"(save {saving:.1%} carbon vs now; "
                            f"cfp_now={carbon_now:.1f} "
                            f"cfp_future={future_carbon:.1f})"),
                    )

        # Dispatch now to cheapest-carbon site
        return DispatchDecision(
            job_id=job.job_id,
            site_id=best_now.site_id,
            dispatch_at=now,
            rationale=(
                f"greedy_carbon: immediate dispatch "
                f"(cfp={carbon_now:.1f} gCO2eq)"),
        )

    def _find_green_window(
        self,
        job: Job,
        candidates: List[Site],
        now: datetime,
        carbon_now: float,
    ) -> Optional[tuple]:
        """
        Return (timestamp, site, carbon) for the best future window, or None.
        Only looks within max_hold_hours.
        """
        max_hold = timedelta(hours=self.max_hold_hours)
        best_future: Optional[tuple] = None
        best_carbon = carbon_now

        for site in candidates:
            future_entries = self._carbon_future.get(site.site_id, [])
            for ts, cfp in future_entries:
                if ts <= now:
                    continue
                if ts > now + max_hold:
                    break
                if ts > job.deadline:
                    break
                if cfp < best_carbon:
                    best_carbon = cfp
                    best_future = (ts, site, cfp)

        return best_future
