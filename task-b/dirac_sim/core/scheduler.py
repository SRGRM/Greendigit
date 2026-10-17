"""
dirac_sim.core.scheduler
========================
Abstract base class every participant must implement.

The interface is intentionally thin: one `schedule()` call per simulation
tick.  Participants choose when to call the REST forecast API, how to cache
results, and how to combine multi-objective signals.

Integration paths
-----------------
* **Simulation-only**: implement `Scheduler` and run `WMSSimulator`.
* **Real DIRAC**: wrap `schedule()` output in a DIRAC `Optimizer` agent
  (see `dirac_sim.backends.dirac_backend`).
* **Real SLURM**: wrap output in `sbatch` hold/release calls
  (see `dirac_sim.backends.slurm_backend`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from dirac_sim.core.job_queue import Job, JobQueue
from dirac_sim.core.site_model import Site, SiteRegistry


class Objective(str, Enum):
    ENERGY = "energy"
    CARBON = "carbon"
    MAKESPAN = "makespan"


@dataclass
class ForecastBundle:
    """
    Wraps forecast data for the current simulation tick.

    Populated by the WMSSimulator either from the REST forecast API or
    from local CSV files (offline mode).

    Attributes
    ----------
    tick_time     : Current simulation clock.
    horizon_1h    : List of records for 1h-ahead horizon.
                    Each record: {series_id, forecast_timestamp_utc,
                                  energy_wh_pred, cfp_g_pred}
    horizon_24h   : Same structure for 24h-ahead horizon.
    raw_response  : Optional: raw API response for custom parsing.
    """
    tick_time: datetime
    horizon_1h: List[Dict] = field(default_factory=list)
    horizon_24h: List[Dict] = field(default_factory=list)
    raw_response: Optional[Dict] = None

    def by_series(self, horizon: str = "1h") -> Dict[str, List[Dict]]:
        """Group records by series_id for fast lookup."""
        records = self.horizon_1h if horizon == "1h" else self.horizon_24h
        out: Dict[str, List[Dict]] = {}
        for r in records:
            out.setdefault(r["series_id"], []).append(r)
        return out


@dataclass
class DispatchDecision:
    """
    A single scheduling decision for one job.

    Attributes
    ----------
    job_id          : Matches Job.job_id.
    site_id         : Target site.
    dispatch_at     : Absolute UTC timestamp at which to dispatch.
                      May be in the future (temporal shifting / holding).
    rationale       : Optional human-readable explanation for debugging.
    """
    job_id: str
    site_id: str
    dispatch_at: datetime
    rationale: str = ""


@dataclass
class DispatchPlan:
    """
    Collection of dispatch decisions produced by one schedule() call.
    """
    decisions: List[DispatchDecision] = field(default_factory=list)
    declared_objective: Objective = Objective.CARBON
    delta_energy_budget: float = 0.10   # allow up to 10% energy overhead
    delta_makespan_budget: float = 0.10

    def add(self, decision: DispatchDecision) -> None:
        self.decisions.append(decision)

    def by_job(self) -> Dict[str, DispatchDecision]:
        return {d.job_id: d for d in self.decisions}


class Scheduler(ABC):
    """
    Abstract base class for all participant schedulers.

    Participants must implement `schedule()`.  Optionally override
    `on_tick_start()` and `on_job_done()` for stateful schedulers.

    Parameters
    ----------
    declared_objective    : Primary optimisation objective.
    delta_energy_budget   : Fractional slack on energy vs FCFS baseline.
    delta_carbon_budget   : Fractional slack on carbon vs FCFS baseline.
    delta_makespan_budget : Fractional slack on makespan vs FCFS baseline.
    """

    def __init__(
        self,
        declared_objective: Objective = Objective.CARBON,
        delta_energy_budget: float = 0.10,
        delta_carbon_budget: float = 0.10,
        delta_makespan_budget: float = 0.10,
    ) -> None:
        self.declared_objective = declared_objective
        self.delta_energy_budget = delta_energy_budget
        self.delta_carbon_budget = delta_carbon_budget
        self.delta_makespan_budget = delta_makespan_budget

    # ------------------------------------------------------------------ #
    # Core interface — must implement                                      #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def schedule(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        forecast: ForecastBundle,
        now: datetime,
    ) -> DispatchPlan:
        """
        Produce a dispatch plan for the current simulation tick.

        Parameters
        ----------
        queue    : Current job queue (read-only recommended).
        registry : Site registry with current + forecast signals pre-loaded.
        forecast : Forecast bundle from the REST API (or offline CSV).
        now      : Current simulation clock (UTC).

        Returns
        -------
        DispatchPlan with zero or more DispatchDecision entries.
        """

    # ------------------------------------------------------------------ #
    # Optional hooks                                                       #
    # ------------------------------------------------------------------ #
    def on_tick_start(self, now: datetime) -> None:
        """Called once per simulation tick before schedule()."""

    def on_job_done(self, job: Job, energy_wh: float, cfp_g: float) -> None:
        """Called when a job completes; useful for online learning."""

    def on_forecast_received(self, bundle: ForecastBundle) -> None:
        """Called after each successful REST API forecast fetch."""
