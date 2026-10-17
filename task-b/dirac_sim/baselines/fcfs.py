"""
dirac_sim.baselines.fcfs
========================
First-Come-First-Served (FCFS) baseline scheduler.

Dispatches every ready job immediately to the site with the most available
slots, with no regard for energy or carbon.  This is the reference baseline
against which all Task B submissions are scored.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from dirac_sim.core.job_queue import Job, JobQueue
from dirac_sim.core.scheduler import (
    DispatchDecision, DispatchPlan, ForecastBundle, Objective, Scheduler,
)
from dirac_sim.core.site_model import Site, SiteRegistry


class FCFSScheduler(Scheduler):
    """
    Blind FCFS: dispatch every ready job immediately to the least-loaded site.
    No use of forecast data whatsoever.
    """

    def __init__(self) -> None:
        super().__init__(declared_objective=Objective.MAKESPAN)

    def schedule(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        forecast: ForecastBundle,
        now: datetime,
    ) -> DispatchPlan:
        plan = DispatchPlan(declared_objective=Objective.MAKESPAN)

        for job in queue.ready_jobs(now):
            site = self._pick_site(job, registry, now)
            if site is None:
                continue
            plan.add(DispatchDecision(
                job_id=job.job_id,
                site_id=site.site_id,
                dispatch_at=now,
                rationale="fcfs: first available slot",
            ))
        return plan

    @staticmethod
    def _pick_site(job: Job, registry: SiteRegistry,
                   now: datetime) -> Optional[Site]:
        candidates = registry.available_sites(job.site_whitelist)
        candidates = [s for s in candidates
                      if s.capacity.available_slots > 0]
        if not candidates:
            return None
        # Most available slots
        return max(candidates, key=lambda s: s.capacity.available_slots)
