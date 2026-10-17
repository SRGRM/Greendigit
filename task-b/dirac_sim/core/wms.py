"""
dirac_sim.core.wms
==================
Workload Management System simulator.

Replays job dispatching decisions over the challenge test period using a
discrete-event tick loop at 15-minute resolution (matching the dataset).

The simulator is intentionally backend-agnostic: it calls
`Scheduler.schedule()` then either:
  * Records the plan for offline evaluation (simulation mode), or
  * Delegates to a real backend (DIRAC / SLURM) via the Backend interface.

Design mirrors DIRAC WMS architecture:
  Task Queue  ->  Matchmaker  ->  Pilot Agent  ->  Worker Node
  (JobQueue)      (Scheduler)     (Backend)        (Site)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from dirac_sim.core.job_queue import Job, JobQueue, JobStatus
from dirac_sim.core.scheduler import (
    DispatchDecision, DispatchPlan, ForecastBundle, Scheduler,
)
from dirac_sim.core.site_model import SiteRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Execution record (one per completed job)                            #
# ------------------------------------------------------------------ #
@dataclass
class ExecutionRecord:
    job_id: str
    series_id: str
    site_id: str
    dispatch_time: datetime
    start_time: datetime
    end_time: datetime
    energy_wh: float
    cfp_g: float
    deadline_met: bool
    rationale: str = ""
    status: str = "running"

    @property
    def makespan_minutes(self) -> float:
        return (self.end_time - self.dispatch_time).total_seconds() / 60.0


# ------------------------------------------------------------------ #
# Backend interface (sim vs real)                                      #
# ------------------------------------------------------------------ #
class Backend:
    """
    Base class for execution backends.

    In simulation mode the default implementation is used (jobs are
    "executed" instantly after dispatch_latency).  Real backends
    (DIRAC, SLURM) override `submit()` and `poll()`.
    """

    def submit(self, job: Job, site_id: str,
               dispatch_time: datetime) -> str:
        """Submit a job; return backend job handle / ID."""
        return job.job_id   # sim: handle == job_id

    def poll(self, handle: str) -> Optional[str]:
        """Return 'done' | 'running' | 'failed' | None (unknown)."""
        return "done"       # sim: always instant completion

    def cancel(self, handle: str) -> None:
        pass


# ------------------------------------------------------------------ #
# Main simulator                                                       #
# ------------------------------------------------------------------ #
class WMSSimulator:
    """
    Discrete-event WMS simulator at 15-minute resolution.

    Usage
    -----
    >>> from dirac_sim.core.wms import WMSSimulator
    >>> from dirac_sim.baselines.greedy_carbon import GreedyCarbonScheduler
    >>> from dirac_sim.api.forecast_client import ForecastClient
    >>>
    >>> sim = WMSSimulator(
    ...     queue=JobQueue.from_csv("data/job_trace.csv"),
    ...     registry=SiteRegistry.from_json("data/site_config.json"),
    ...     scheduler=GreedyCarbonScheduler(),
    ...     forecast_client=ForecastClient(base_url="http://localhost:8000"),
    ...     start_time=datetime(2026, 2, 17, 14, 0, tzinfo=timezone.utc),
    ...     end_time=datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
    ... )
    >>> report = sim.run()
    >>> print(report.summary())
    """

    TICK_MINUTES = 15

    def __init__(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        scheduler: Scheduler,
        forecast_client=None,        # ForecastClient | None
        backend: Optional[Backend] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        offline_forecast_path: Optional[str] = None,
    ) -> None:
        self.queue = queue
        self.registry = registry
        self.scheduler = scheduler
        self.forecast_client = forecast_client
        self.backend = backend or Backend()
        self.offline_forecast_path = offline_forecast_path

        # Default window: full test period
        self.start_time = start_time or datetime(2026, 2, 17, 14, 0,
                                                 tzinfo=timezone.utc)
        self.end_time = end_time or datetime(2026, 2, 24, 0, 0,
                                             tzinfo=timezone.utc)
        self._records: List[ExecutionRecord] = []
        self._pending_dispatches: Dict[str, DispatchDecision] = {}
        self._backend_handles: Dict[str, str] = {}  # job_id -> handle
        self._running_jobs: Dict[str, ExecutionRecord] = {}

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    def run(self) -> "SimulationReport":
        """Run the simulation and return a report."""
        logger.info("WMSSimulator starting: %s -> %s",
                    self.start_time.isoformat(), self.end_time.isoformat())

        tick = self.start_time
        tick_delta = timedelta(minutes=self.TICK_MINUTES)

        while tick <= self.end_time:
            self._tick(tick)
            tick += tick_delta

        self._complete_jobs_due(self.end_time + tick_delta)

        # Mark any still-pending jobs as failed (deadline missed)
        for job in self.queue.all_jobs():
            if (job.arrival_time <= self.end_time
                    and job.status in {JobStatus.PENDING, JobStatus.HELD}):
                self.queue.update_status(job.job_id, JobStatus.FAILED)
                logger.warning("Job %s never dispatched (deadline missed)",
                               job.job_id)

        logger.info("WMSSimulator finished. %d records.", len(self._records))
        return SimulationReport(
            records=self._records,
            queue=self.queue,
            start_time=self.start_time,
            end_time=self.end_time,
        )

    # ------------------------------------------------------------------ #
    # Per-tick logic                                                       #
    # ------------------------------------------------------------------ #
    def _tick(self, now: datetime) -> None:
        logger.debug("Tick %s", now.isoformat())

        # 1. Release jobs that completed by this tick
        self._complete_jobs_due(now)

        # 2. Fetch forecasts (REST API or offline)
        bundle = self._fetch_forecast(now)

        # 3. Ingest forecast into site registry
        self._update_site_signals(bundle)

        # 4. Notify scheduler
        self.scheduler.on_tick_start(now)
        self.scheduler.on_forecast_received(bundle)

        # 5. Execute any previously scheduled future dispatches due now
        self._execute_future_dispatches(now)

        # 6. Call scheduler for ready jobs
        ready = self.queue.ready_jobs(now)
        if ready:
            plan = self.scheduler.schedule(
                queue=self.queue,
                registry=self.registry,
                forecast=bundle,
                now=now,
            )
            self._apply_plan(plan, now)

    def _fetch_forecast(self, now: datetime) -> ForecastBundle:
        """Fetch forecast from REST API or offline CSV."""
        if self.forecast_client is not None:
            try:
                return self.forecast_client.fetch_bundle(now)
            except Exception as exc:
                logger.warning("Forecast API error at %s: %s — using empty bundle",
                               now.isoformat(), exc)
        # Offline / fallback: empty bundle (sites use last known values)
        return ForecastBundle(tick_time=now)

    def _update_site_signals(self, bundle: ForecastBundle) -> None:
        for record in bundle.horizon_1h + bundle.horizon_24h:
            site = self.registry.get(record.get("series_id", ""))
            if site:
                site.bulk_update([record])

    def _apply_plan(self, plan: DispatchPlan, now: datetime) -> None:
        for decision in plan.decisions:
            job = self.queue.all_jobs()
            job_map = {j.job_id: j for j in job}
            j = job_map.get(decision.job_id)
            if j is None or j.status != JobStatus.PENDING:
                continue

            if decision.dispatch_at <= now:
                # Dispatch immediately
                self._dispatch(j, decision, now)
            else:
                # Hold for future dispatch (temporal shifting)
                self._pending_dispatches[decision.job_id] = decision
                self.queue.update_status(j.job_id, JobStatus.HELD)
                logger.debug("Job %s held until %s",
                             j.job_id, decision.dispatch_at.isoformat())

    def _execute_future_dispatches(self, now: datetime) -> None:
        due = [d for d in self._pending_dispatches.values()
               if d.dispatch_at <= now]
        for decision in due:
            job_map = {j.job_id: j for j in self.queue.all_jobs()}
            j = job_map.get(decision.job_id)
            if j and j.status == JobStatus.HELD:
                dispatched = self._dispatch(j, decision, now)
                if not dispatched:
                    self.queue.update_status(j.job_id, JobStatus.PENDING)
            self._pending_dispatches.pop(decision.job_id, None)

    def _dispatch(self, job: Job, decision: DispatchDecision,
                  now: datetime) -> bool:
        site = self.registry.get(decision.site_id)
        if site is None:
            logger.error("Unknown site %s for job %s",
                         decision.site_id, job.job_id)
            return False

        cores_required = self._job_cores(job)
        if (site.capacity.available_slots <= 0
                or site.capacity.available_cores < cores_required
                or site.capacity.available_memory_gb < job.memory_gb):
            logger.warning("Site %s full; job %s re-queued",
                           site.site_id, job.job_id)
            return False

        # Submit to backend (simulation or real)
        handle = self.backend.submit(job, decision.site_id, now)
        self._backend_handles[job.job_id] = handle
        site.capacity.current_jobs += 1
        site.capacity.current_cores += cores_required
        site.capacity.current_memory_gb += job.memory_gb

        self.queue.update_status(job.job_id, JobStatus.RUNNING,
                                 dispatch_time=now,
                                 site_id=decision.site_id)

        # Simulate execution (in real backends, poll() handles this)
        start_time = now + timedelta(minutes=site.dispatch_latency)
        end_time = start_time + timedelta(minutes=job.cpu_minutes)

        # Energy / carbon accounting
        energy = site.get_energy(now) * (job.cpu_minutes / 15.0)
        carbon = site.get_carbon(now) * (job.cpu_minutes / 15.0)
        deadline_met = now <= job.deadline

        record = ExecutionRecord(
            job_id=job.job_id,
            series_id=job.series_id,
            site_id=decision.site_id,
            dispatch_time=now,
            start_time=start_time,
            end_time=end_time,
            energy_wh=energy,
            cfp_g=carbon,
            deadline_met=deadline_met,
            rationale=decision.rationale,
        )
        self._records.append(record)
        self._running_jobs[job.job_id] = record

        logger.debug("Dispatched job %s -> site %s (energy=%.1f Wh, "
                     "carbon=%.1f g, deadline_met=%s)",
                     job.job_id, decision.site_id, energy, carbon, deadline_met)
        return True

    def _complete_jobs_due(self, now: datetime) -> None:
        completed = [
            record for record in self._running_jobs.values()
            if record.end_time <= now
        ]
        if not completed:
            return
        job_map = {j.job_id: j for j in self.queue.all_jobs()}
        for record in completed:
            job = job_map.get(record.job_id)
            site = self.registry.get(record.site_id)
            if job is not None:
                self.queue.update_status(job.job_id, JobStatus.DONE)
                self.scheduler.on_job_done(job, record.energy_wh, record.cfp_g)
            if site is not None:
                site.capacity.current_jobs = max(0, site.capacity.current_jobs - 1)
                site.capacity.current_cores = max(
                    0, site.capacity.current_cores - self._job_cores(job))
                if job is not None:
                    site.capacity.current_memory_gb = max(
                        0.0, site.capacity.current_memory_gb - job.memory_gb)
            record.status = "done"
            self._running_jobs.pop(record.job_id, None)

    @staticmethod
    def _job_cores(job: Optional[Job]) -> int:
        if job is None:
            return 1
        raw = job.metadata.get("ncores", job.metadata.get("cores", 1))
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return 1


# ------------------------------------------------------------------ #
# Simulation report                                                    #
# ------------------------------------------------------------------ #
@dataclass
class SimulationReport:
    records: List[ExecutionRecord]
    queue: JobQueue
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def summary(self) -> str:
        n = len(self.records)
        if n == 0:
            return "No jobs executed."
        total_energy = sum(r.energy_wh for r in self.records)
        total_carbon = sum(r.cfp_g for r in self.records)
        total_makespan = sum(r.makespan_minutes for r in self.records)
        deadlines_met = sum(1 for r in self.records if r.deadline_met)
        return (
            f"Jobs executed : {n}\n"
            f"Deadlines met : {deadlines_met}/{n} "
            f"({100*deadlines_met/n:.1f}%)\n"
            f"Total energy  : {total_energy:,.1f} Wh\n"
            f"Total carbon  : {total_carbon:,.1f} gCO2eq\n"
            f"Total makespan: {total_makespan:,.1f} min"
        )

    def to_csv(self, path: str) -> None:
        import csv
        if not self.records:
            return
        keys = list(self.records[0].__dataclass_fields__.keys())
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in self.records:
                row = {k: getattr(r, k) for k in keys}
                row["dispatch_time"] = row["dispatch_time"].isoformat()
                row["start_time"] = row["start_time"].isoformat()
                row["end_time"] = row["end_time"].isoformat()
                writer.writerow(row)

    def to_dispatch_csv(self, path: str, declared_objective: str = "carbon") -> None:
        import csv
        keys = [
            "job_id", "dispatch_timestamp_utc", "site_id",
            "declared_objective", "deadline_met", "status",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in self.records:
                writer.writerow({
                    "job_id": r.job_id,
                    "dispatch_timestamp_utc": r.dispatch_time.isoformat(),
                    "site_id": r.site_id,
                    "declared_objective": declared_objective,
                    "deadline_met": r.deadline_met,
                    "status": r.status,
                })
