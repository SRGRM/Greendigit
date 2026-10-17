"""
Forecast-shift probe scheduler.

This is a local debugging policy for checking that Task A forecasts are being
consumed by Task B. It respects each job's site whitelist, but delays a job by
one hour when the same site's one-hour-ahead forecast has lower carbon than the
current bucket and the job has enough deadline slack.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from dirac_sim.core.job_queue import JobQueue
from dirac_sim.core.scheduler import (
    DispatchDecision,
    DispatchPlan,
    ForecastBundle,
    Objective,
    Scheduler,
)
from dirac_sim.core.site_model import Site, SiteRegistry


class ForecastShiftProbe(Scheduler):
    """Delay jobs by one hour when the forecast says the same site gets cleaner."""

    def __init__(self, declared_objective: str = "carbon") -> None:
        obj = Objective(declared_objective) if isinstance(
            declared_objective, str) else declared_objective
        super().__init__(declared_objective=obj)
        self.probe_end = _parse_optional_utc(os.environ.get("FORECAST_SHIFT_PROBE_END"))

    def schedule(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        forecast: ForecastBundle,
        now: datetime,
    ) -> DispatchPlan:
        plan = DispatchPlan(declared_objective=self.declared_objective)
        one_hour = now + timedelta(hours=1)

        for job in queue.ready_jobs(now):
            candidates = [
                site for site in registry.available_sites(job.site_whitelist)
                if site.capacity.available_slots > 0
            ]
            if not candidates:
                continue

            site = self._pick_job_site(job.series_id, candidates)
            dispatch_at = now
            current_cfp = site.get_carbon(now)
            future_cfp = site.get_carbon(one_hour)
            rationale = (
                "forecast_shift_probe: immediate dispatch "
                f"(cfp_now={current_cfp:.1f}, cfp_1h={future_cfp:.1f})"
            )

            can_hold_within_probe = self.probe_end is None or one_hour <= self.probe_end
            if can_hold_within_probe and one_hour <= job.deadline and future_cfp < current_cfp:
                dispatch_at = one_hour
                rationale = (
                    "forecast_shift_probe: held for cleaner same-site bucket "
                    f"(cfp_now={current_cfp:.1f}, cfp_1h={future_cfp:.1f})"
                )

            plan.add(DispatchDecision(
                job_id=job.job_id,
                site_id=site.site_id,
                dispatch_at=dispatch_at,
                rationale=rationale,
            ))

        return plan

    @staticmethod
    def _pick_job_site(series_id: str, candidates: list[Site]) -> Site:
        for site in candidates:
            if site.site_id == series_id:
                return site
        return candidates[0]


def _parse_optional_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
