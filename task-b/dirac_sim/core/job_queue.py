"""
dirac_sim.core.job_queue
========================
Job model and queue management, inspired by the DIRAC WMS TaskQueue design.

Each Job maps to a real DIRAC Job or a SLURM batch script — the fields are
chosen to be directly serialisable to both backends.
"""
from __future__ import annotations

import csv
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


class Priority(str, Enum):
    EXPRESS = "express"
    NORMAL = "normal"
    BEST_EFFORT = "best_effort"


class JobStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    HELD = "held"          # deferred by scheduler for green window


@dataclass
class Job:
    """
    Single compute job in the queue.

    Attributes
    ----------
    job_id        : Unique identifier (UUID string).
    series_id     : Challenge series_id this job is associated with
                    (used to look up forecast signals).
    arrival_time  : Earliest the job may be dispatched (UTC).
    deadline      : Latest dispatch start time; violation triggers penalty (UTC).
    cpu_minutes   : Expected wall-clock duration on a reference node.
    memory_gb     : Memory requirement.
    site_whitelist: Compatible site IDs (empty = any site).
    priority      : Express / normal / best-effort.
    status        : Current lifecycle status.
    dispatch_time : Actual dispatch timestamp (set by simulator).
    site_id       : Site where job was dispatched.
    metadata      : Arbitrary key-value bag for backend-specific info.
    """
    job_id: str
    series_id: str
    arrival_time: datetime
    deadline: datetime
    cpu_minutes: float
    memory_gb: float = 2.0
    site_whitelist: List[str] = field(default_factory=list)
    priority: Priority = Priority.NORMAL
    status: JobStatus = JobStatus.PENDING
    dispatch_time: Optional[datetime] = None
    site_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #
    def is_ready(self, now: datetime) -> bool:
        """True if the job has arrived and is still pending."""
        return self.status == JobStatus.PENDING and self.arrival_time <= now

    def deadline_slack_minutes(self, now: datetime) -> float:
        """Minutes remaining before deadline (negative = overdue)."""
        return (self.deadline - now).total_seconds() / 60.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "series_id": self.series_id,
            "arrival_time": self.arrival_time.isoformat(),
            "deadline": self.deadline.isoformat(),
            "cpu_minutes": self.cpu_minutes,
            "memory_gb": self.memory_gb,
            "site_whitelist": self.site_whitelist,
            "priority": self.priority.value,
            "status": self.status.value,
            "dispatch_time": self.dispatch_time.isoformat() if self.dispatch_time else None,
            "site_id": self.site_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        known = {
            "job_id", "series_id", "arrival_time", "deadline", "cpu_minutes",
            "memory_gb", "site_whitelist", "priority", "status",
            "dispatch_time", "site_id", "metadata",
        }
        metadata = dict(d.get("metadata", {}))
        for key, value in d.items():
            if key not in known and value not in (None, ""):
                metadata[key] = value
        return cls(
            job_id=d.get("job_id", str(uuid.uuid4())),
            series_id=d["series_id"],
            arrival_time=_parse_dt(d["arrival_time"]),
            deadline=_parse_dt(d["deadline"]),
            cpu_minutes=float(d["cpu_minutes"]),
            memory_gb=float(d.get("memory_gb", 2.0)),
            site_whitelist=d.get("site_whitelist", []),
            priority=Priority(d.get("priority", "normal")),
            metadata=metadata,
        )


class JobQueue:
    """
    In-memory job queue with FIFO + priority ordering.

    Mirrors DIRAC's TaskQueue concept: jobs with identical requirements are
    grouped and served in priority order.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}

    # ------------------------------------------------------------------ #
    # Mutators                                                             #
    # ------------------------------------------------------------------ #
    def add(self, job: Job) -> None:
        self._jobs[job.job_id] = job

    def update_status(self, job_id: str, status: JobStatus,
                      dispatch_time: Optional[datetime] = None,
                      site_id: Optional[str] = None) -> None:
        j = self._jobs[job_id]
        j.status = status
        if dispatch_time:
            j.dispatch_time = dispatch_time
        if site_id:
            j.site_id = site_id

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #
    def ready_jobs(self, now: datetime) -> List[Job]:
        """All jobs that have arrived and are still pending, priority-sorted."""
        _priority_order = {
            Priority.EXPRESS: 0,
            Priority.NORMAL: 1,
            Priority.BEST_EFFORT: 2,
        }
        jobs = [j for j in self._jobs.values() if j.is_ready(now)]
        return sorted(jobs, key=lambda j: (_priority_order[j.priority],
                                           j.arrival_time))

    def all_jobs(self) -> List[Job]:
        return list(self._jobs.values())

    def __len__(self) -> int:
        return len(self._jobs)

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_csv(cls, path: str | Path) -> "JobQueue":
        """Load job trace from CSV (see data/job_trace.csv for schema)."""
        q = cls()
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                wl = row.get("site_whitelist", "")
                row["site_whitelist"] = [s.strip() for s in wl.split("|") if s.strip()]
                q.add(Job.from_dict(row))
        return q

    def to_csv(self, path: str | Path) -> None:
        jobs = self.all_jobs()
        if not jobs:
            return
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(jobs[0].to_dict().keys()))
            writer.writeheader()
            for j in jobs:
                d = j.to_dict()
                d["site_whitelist"] = "|".join(d["site_whitelist"])
                writer.writerow(d)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #
def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
