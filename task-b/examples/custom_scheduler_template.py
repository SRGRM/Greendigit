"""
examples/custom_scheduler_template.py
======================================
Template for participants to implement their own Task B scheduler.

Copy this file to your own package, implement the TODO sections, then
run it with:

    python examples/run_simulation.py \
        --offline \
        --scheduler examples.custom_scheduler_template.MyScheduler

The scheduler has access to:
  * queue    : all pending jobs with priorities and deadlines
  * registry : all sites with live energy/carbon signals (updated from API)
  * forecast : 1h-ahead and 24h-ahead bundles from your Task A model
  * now      : current simulation clock

Your goal: beat the GreedyCarbonScheduler baseline on the Pareto score.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from dirac_sim.core.job_queue import Job, JobQueue
from dirac_sim.core.scheduler import (
    DispatchDecision, DispatchPlan, ForecastBundle, Objective, Scheduler,
)
from dirac_sim.core.site_model import Site, SiteRegistry

logger = logging.getLogger(__name__)


class MyScheduler(Scheduler):
    """
    Your Task B scheduler.

    Declared objective: carbon  (change to 'energy' or 'makespan' if desired)

    Constraints (enforced by evaluator):
      - delta_energy_budget   = 10%  (energy may increase by at most 10% vs FCFS)
      - delta_carbon_budget   = 10%  (likewise for carbon)
      - delta_makespan_budget = 10%  (likewise for makespan)

    Recommended approach:
      1. Parse the 1h-ahead forecast for current site carbon intensities.
      2. Parse the 24h-ahead forecast to identify future green windows.
      3. For each ready job, decide: dispatch now vs defer.
      4. For deferred jobs, choose the optimal future timestamp and site.
      5. Consider multi-objective tradeoffs (e.g. defer only if deadline allows).
    """

    def __init__(
        self,
        declared_objective: str = "carbon",
        delta_energy_budget: float = 0.10,
        delta_carbon_budget: float = 0.10,
        delta_makespan_budget: float = 0.10,
    ) -> None:
        super().__init__(
            declared_objective=Objective(declared_objective),
            delta_energy_budget=delta_energy_budget,
            delta_carbon_budget=delta_carbon_budget,
            delta_makespan_budget=delta_makespan_budget,
        )
        # TODO: initialise your model / state here
        self._carbon_forecast: Dict[str, List[Tuple[datetime, float]]] = {}
        self._energy_forecast: Dict[str, List[Tuple[datetime, float]]] = {}

    # ------------------------------------------------------------------ #
    # Hook: called every tick BEFORE schedule()                           #
    # ------------------------------------------------------------------ #
    def on_forecast_received(self, bundle: ForecastBundle) -> None:
        """
        Pre-process the forecast bundle.
        Called automatically by WMSSimulator after each REST API fetch.

        TODO: extract and store the signals you need.
        """
        # Example: build carbon timeline per site from 24h-ahead forecast
        by_series = bundle.by_series("24h")
        for series_id, records in by_series.items():
            timeline = []
            for rec in sorted(records,
                               key=lambda r: r["forecast_timestamp_utc"]):
                ts_str = rec["forecast_timestamp_utc"]
                try:
                    ts = datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                cfp = float(rec.get("cfp_g_pred", 9999.0))
                energy = float(rec.get("energy_wh_pred", 9999.0))
                timeline.append((ts, cfp))
            self._carbon_forecast[series_id] = timeline

    # ------------------------------------------------------------------ #
    # Core: called every tick to produce the dispatch plan                #
    # ------------------------------------------------------------------ #
    def schedule(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        forecast: ForecastBundle,
        now: datetime,
    ) -> DispatchPlan:
        """
        Produce a DispatchPlan for the current tick.

        Parameters
        ----------
        queue    : Job queue — call queue.ready_jobs(now) to get pending jobs.
        registry : Site registry — call registry.available_sites(whitelist)
                   to get compatible sites.
        forecast : Forecast bundle — 1h and 24h horizon records.
        now      : Current simulation clock (UTC).

        Returns
        -------
        DispatchPlan with DispatchDecision for each job you want to dispatch
        or defer. Jobs with no decision remain pending for the next tick.
        """
        plan = DispatchPlan(declared_objective=self.declared_objective)

        for job in queue.ready_jobs(now):
            decision = self._make_decision(job, registry, now)
            if decision is not None:
                plan.add(decision)

        return plan

    # ------------------------------------------------------------------ #
    # Per-job decision                                                     #
    # ------------------------------------------------------------------ #
    def _make_decision(
        self,
        job: Job,
        registry: SiteRegistry,
        now: datetime,
    ) -> Optional[DispatchDecision]:
        """
        Decide what to do with one job.

        Return a DispatchDecision with:
          - dispatch_at == now          → dispatch immediately
          - dispatch_at >  now          → hold until that future timestamp
          - None                        → skip this job this tick (retry next)

        TODO: Replace the stub below with your actual logic.
        """
        candidates = [
            s for s in registry.available_sites(job.site_whitelist)
            if s.capacity.available_slots > 0
        ]
        if not candidates:
            return None  # No slots available; retry next tick

        # ---- STUB: dispatch immediately to lowest-carbon site ---- #
        # Replace this with your learned / optimised policy.
        best_site = self._pick_site(candidates, job, now)
        if best_site is None:
            return None

        return DispatchDecision(
            job_id=job.job_id,
            site_id=best_site.site_id,
            dispatch_at=now,    # TODO: consider temporal shifting
            rationale="my_scheduler: stub dispatch",
        )

    def _pick_site(
        self,
        candidates: List[Site],
        job: Job,
        now: datetime,
    ) -> Optional[Site]:
        """
        Pick the best site for this job.

        TODO: implement your multi-objective site selection.
        Currently: lowest carbon intensity at current time.
        """
        return min(
            candidates,
            key=lambda s: self._current_carbon(s.site_id, now),
        )

    def _current_carbon(self, series_id: str, now: datetime) -> float:
        """Return forecast carbon for *now* (or large sentinel if unknown)."""
        timeline = self._carbon_forecast.get(series_id, [])
        if not timeline:
            return 9999.0
        # Find closest future entry
        for ts, cfp in timeline:
            if ts >= now:
                return cfp
        return timeline[-1][1] if timeline else 9999.0

    # ------------------------------------------------------------------ #
    # Optional hook: called after each job completes                     #
    # ------------------------------------------------------------------ #
    def on_job_done(self, job: Job, energy_wh: float, cfp_g: float) -> None:
        """
        Update any online model state when a job finishes.

        TODO: use this for online learning / adaptive policies.
        """
        logger.debug("Job %s done: energy=%.1f Wh, carbon=%.1f g",
                     job.job_id, energy_wh, cfp_g)


# ------------------------------------------------------------------ #
# Quick smoke test                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from dirac_sim.core.job_queue import JobQueue
    from dirac_sim.core.site_model import SiteRegistry
    from dirac_sim.core.wms import WMSSimulator
    from dirac_sim.api.forecast_client import ForecastClient
    from datetime import timezone

    data_dir = Path(__file__).resolve().parents[2] / "data"
    queue    = JobQueue.from_csv(data_dir / "job_trace.csv")
    registry = SiteRegistry.from_json(data_dir / "site_config.json")
    series_ids = [site.site_id for site in registry.all_sites()]
    client = ForecastClient(
        offline_csv=str(data_dir / "forecast_baseline.csv"),
        series_ids=series_ids,
    )

    sim = WMSSimulator(
        queue=queue, registry=registry,
        scheduler=MyScheduler(),
        forecast_client=client,
        start_time=datetime(2025, 11, 19, 23, 0, tzinfo=timezone.utc),
        end_time=datetime(2025, 11, 20, 23, 0, tzinfo=timezone.utc),
    )
    report = sim.run()
    print(report.summary())
