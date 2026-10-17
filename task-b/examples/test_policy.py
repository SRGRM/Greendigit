"""
Test-only policy used to verify that scoring responds to changed dispatches.

This scheduler deliberately ignores each job's site whitelist and should not be
used as a participant submission.  It exists only as a smoke-test policy that
can produce non-zero score deltas on the pinned aggregate trace.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from dirac_sim.core.job_queue import Job, JobQueue
from dirac_sim.core.scheduler import (
    DispatchDecision,
    DispatchPlan,
    ForecastBundle,
    Objective,
    Scheduler,
)
from dirac_sim.core.site_model import Site, SiteRegistry


class LowestCarbonTestPolicy(Scheduler):
    """Dispatch ready jobs to the currently lowest-carbon site."""

    def __init__(self, declared_objective: str = "carbon") -> None:
        obj = Objective(declared_objective) if isinstance(
            declared_objective, str) else declared_objective
        super().__init__(declared_objective=obj)

    def schedule(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        forecast: ForecastBundle,
        now: datetime,
    ) -> DispatchPlan:
        plan = DispatchPlan(declared_objective=self.declared_objective)
        for job in queue.ready_jobs(now):
            site = self._pick_site(job, registry, now)
            if site is None:
                continue
            plan.add(DispatchDecision(
                job_id=job.job_id,
                site_id=site.site_id,
                dispatch_at=now,
                rationale=(
                    "test_policy: lowest current carbon, intentionally "
                    "ignores site whitelist"),
            ))
        return plan

    @staticmethod
    def _pick_site(job: Job, registry: SiteRegistry,
                   now: datetime) -> Optional[Site]:
        candidates = [
            site for site in registry.all_sites()
            if site.capacity.available_slots > 0
            and site.capacity.available_cores >= _job_cores(job)
            and site.capacity.available_memory_gb >= job.memory_gb
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda site: site.get_carbon(now))


def _job_cores(job: Job) -> int:
    try:
        return max(1, int(float(job.metadata.get("ncores", 1))))
    except (TypeError, ValueError):
        return 1
