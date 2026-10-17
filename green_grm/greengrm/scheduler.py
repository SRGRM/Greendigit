"""Task B: forecast-driven Pareto scheduler (Sect. 4.6).

For every ready job the scheduler compares the predicted energy and carbon
cost of dispatching the job immediately with the predicted cost of every
candidate bucket inside the forecast horizon (24 hours), restricted to a
maximum hold of 12 hours. For a candidate bucket b the relative savings are

    s_E(b) = (E_now - E_b) / E_now,      s_C(b) = (C_now - C_b) / C_now

and the trade-off score combines them, w_E * s_E(b) + w_C * s_C(b), minus
an optional delay penalty. Only candidates on the Pareto front of predicted
energy, predicted carbon and delay are kept, and the one with the best
trade-off is retained. The job is deferred to it only when the predicted
saving exceeds ``saving_threshold`` (0.01) and the deferral is compatible
with the deadline; otherwise it is dispatched at once at its whitelisted
site. The threshold keeps out deferrals whose expected benefit is smaller
than the forecast uncertainty, and the deadline check holds whatever the
forecast quality is.

Implementation details that the paper leaves open:

* Forecast timeline. The service issues a 1-hour and a 24-hour forecast per
  site at every tick. The scheduler keeps every forecast it has received and
  uses, for each bucket, the most recent one; at tick t it therefore knows
  24-hour forecasts for all buckets up to t + 24 h and 1-hour forecasts for
  the next hour.
* Capacity. Deferred jobs reserve job slots, cores and memory in their
  target bucket, and deferrals may fill at most ``capacity_share`` of the
  site's capacity in any bucket. Jobs deferred independently therefore
  cannot converge on, and exhaust, the same low-cost bucket, which is the
  weakness of ``greedy_carbon``; the remaining share is left for the jobs
  that arrive in that bucket.
* Missing telemetry (optional). The simulator charges a bucket without
  telemetry with the default signal of ``dirac_sim`` (1000 Wh and 200 g per
  15 minutes). With ``availability`` given, the predicted cost of a
  candidate bucket is the expectation p * forecast + (1 - p) * default,
  where p is the share of buckets with telemetry for that site and hour of
  day in the training data. The immediate option keeps the plain forecast
  when the job arrived in the current bucket, since its arrival shows that
  the site is active there; for a job that waited, the same expectation is
  applied.
* End of the evaluation window. Jobs are not deferred into the last
  ``end_margin_hours`` before ``horizon_end``. Deferred jobs are executed
  before the new arrivals of a bucket, so without this margin a large job
  arriving near the end could find every remaining bucket occupied and
  never be dispatched.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from dirac_sim.core.job_queue import Job, JobQueue
from dirac_sim.core.scheduler import (DispatchDecision, DispatchPlan, ForecastBundle, Objective,
                                      Scheduler)
from dirac_sim.core.site_model import Site, SiteRegistry

from .forecast_feed import ForecastFeed, bucket_key

STEP = timedelta(minutes=15)
EPS = 1e-9


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


class GreenGRMParetoScheduler(Scheduler):
    def __init__(self, declared_objective: str = "carbon", feed: Optional[ForecastFeed] = None,
                 max_hold_hours: float = 12.0, forecast_horizon_hours: float = 24.0,
                 saving_threshold: float = 0.01, energy_weight: float = 0.5,
                 carbon_weight: float = 0.5, delay_penalty_per_hour: float = 0.0,
                 deadline_rule: str = "completion",
                 availability: Optional[Dict[str, Dict[int, float]]] = None,
                 fallback_energy_wh: float = 1000.0, fallback_cfp_g: float = 200.0,
                 capacity_share: float = 0.5, horizon_end: Optional[datetime] = None,
                 end_margin_hours: float = 3.0, warm_start: bool = True,
                 record_decisions: bool = True):
        objective = Objective(declared_objective) if isinstance(declared_objective, str) else declared_objective
        super().__init__(declared_objective=objective)
        if deadline_rule not in ("completion", "start"):
            raise ValueError("deadline_rule must be 'completion' or 'start'")
        total = energy_weight + carbon_weight
        if total <= 0:
            raise ValueError("energy_weight + carbon_weight must be positive")
        self.feed = feed
        self.max_hold = timedelta(hours=max_hold_hours)
        self.forecast_horizon = timedelta(hours=forecast_horizon_hours)
        self.saving_threshold = saving_threshold
        self.w_energy = energy_weight / total
        self.w_carbon = carbon_weight / total
        self.delay_penalty = delay_penalty_per_hour
        self.deadline_rule = deadline_rule
        self.availability = availability
        self.fallback = (fallback_energy_wh, fallback_cfp_g)
        if not 0.0 < capacity_share <= 1.0:
            raise ValueError("capacity_share must be in (0, 1]")
        self.capacity_share = capacity_share
        self.horizon_end = horizon_end
        self.end_margin = timedelta(hours=end_margin_hours)
        self.warm_start = warm_start
        self.record_decisions = record_decisions

        self.timeline: Dict[str, Dict[str, Tuple[float, float]]] = {}
        # (site, bucket) -> [jobs, cores, memory_gb] reserved by deferrals
        self.reservations: Dict[Tuple[str, str], List[float]] = {}
        self.decisions: List[Dict] = []
        self.counters = {"immediate": 0, "deferred": 0, "no_forecast": 0}
        self._started = False

    # ------------------------------------------------------------------
    # Forecast intake
    # ------------------------------------------------------------------
    def _ingest(self, records) -> None:
        for rec in records:
            site = rec.get("series_id", "")
            ts = rec.get("forecast_timestamp_utc")
            if not site or ts is None:
                continue
            try:
                key = bucket_key(_parse(ts))
                values = (float(rec["energy_wh_pred"]), float(rec["cfp_g_pred"]))
            except (KeyError, TypeError, ValueError):
                continue
            # Records arrive in issue order, so the newest forecast of a
            # bucket replaces the older one (24 h forecast -> 1 h forecast).
            self.timeline.setdefault(site, {})[key] = values

    def on_forecast_received(self, bundle: ForecastBundle) -> None:
        now = bundle.tick_time
        if self.feed is None:
            # Use the forecasts delivered by the simulator (live API or CSV).
            self._ingest(bundle.horizon_1h + bundle.horizon_24h)
            return
        if not self._started and self.warm_start:
            # A running service would already have issued the forecasts of
            # the previous day.
            self._ingest(self.feed.issued_between(now - self.forecast_horizon, now))
        self._started = True
        self._ingest(self.feed.issued_at(now))
        self._prune(now - timedelta(days=1))

    def _prune(self, before: datetime) -> None:
        limit = bucket_key(before)
        for site, table in self.timeline.items():
            stale = [k for k in table if k < limit]
            for key in stale:
                del table[key]
        stale = [k for k in self.reservations if k[1] < limit]
        for key in stale:
            del self.reservations[key]

    def _signal(self, site: str, bucket: datetime) -> Optional[Tuple[float, float]]:
        return self.timeline.get(site, {}).get(bucket_key(bucket))

    def _current_signal(self, site: str, now: datetime) -> Optional[Tuple[float, float]]:
        """Predicted cost of dispatching now; if the bucket of ``now`` has no
        forecast yet, the next forecast bucket is used."""
        value = self._signal(site, now)
        if value is not None:
            return value
        table = self.timeline.get(site, {})
        later = sorted(k for k in table if k >= bucket_key(now))
        return table[later[0]] if later else None

    def _expected_cost(self, site: str, bucket: datetime, values: Tuple[float, float]) -> Tuple[float, float]:
        if self.availability is None:
            return values
        p = self.availability.get(site, {}).get(bucket.hour, 1.0)
        return (p * values[0] + (1.0 - p) * self.fallback[0],
                p * values[1] + (1.0 - p) * self.fallback[1])

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def schedule(self, queue: JobQueue, registry: SiteRegistry, forecast: ForecastBundle,
                 now: datetime) -> DispatchPlan:
        plan = DispatchPlan(declared_objective=self.declared_objective)
        ready = sorted(queue.ready_jobs(now), key=lambda j: j.deadline_slack_minutes(now))
        for job in ready:
            decision = self._decide(job, registry, now)
            if decision is not None:
                plan.add(decision)
        return plan

    @staticmethod
    def _job_cores(job: Job) -> int:
        """Cores of a job, read the same way as ``WMSSimulator._job_cores``."""
        raw = job.metadata.get("ncores", job.metadata.get("cores", 1))
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return 1

    def _fits(self, job: Job, site: Site, bucket: datetime) -> bool:
        """True when the deferral keeps the reserved load of the bucket within
        ``capacity_share`` of the site's slots, cores and memory."""
        jobs, cores, memory = self.reservations.get((site.site_id, bucket_key(bucket)), (0, 0, 0.0))
        cap = site.capacity
        share = self.capacity_share
        return (jobs + 1 <= share * cap.max_concurrent_jobs
                and cores + self._job_cores(job) <= share * cap.max_cores
                and memory + job.memory_gb <= share * cap.max_memory_gb)

    def _reserve(self, job: Job, site: Site, bucket: datetime) -> None:
        entry = self.reservations.setdefault((site.site_id, bucket_key(bucket)), [0, 0, 0.0])
        entry[0] += 1
        entry[1] += self._job_cores(job)
        entry[2] += job.memory_gb

    def _deadline_ok(self, job: Job, site: Site, dispatch_at: datetime) -> bool:
        if self.deadline_rule == "start":
            return dispatch_at <= job.deadline
        finish = dispatch_at + timedelta(minutes=site.dispatch_latency + job.cpu_minutes)
        return finish <= job.deadline

    def _decide(self, job: Job, registry: SiteRegistry, now: datetime) -> Optional[DispatchDecision]:
        sites = [s for s in registry.available_sites(job.site_whitelist) if s.capacity.available_slots > 0]
        if not sites:
            return None                                   # no capacity: wait for the next tick

        current = {s.site_id: self._current_signal(s.site_id, now) for s in sites}
        known = [s for s in sites if current[s.site_id] is not None]
        if not known:
            self.counters["no_forecast"] += 1
            site = max(sites, key=lambda s: s.capacity.available_slots)
            return self._immediate(job, site, now, None, "no forecast available")

        # Immediate option: the whitelisted site with the lowest weighted cost.
        mean_e = sum(current[s.site_id][0] for s in known) / len(known)
        mean_c = sum(current[s.site_id][1] for s in known) / len(known)
        now_site = min(known, key=lambda s: self.w_energy * current[s.site_id][0] / (mean_e + EPS)
                       + self.w_carbon * current[s.site_id][1] / (mean_c + EPS))
        e_now, c_now = current[now_site.site_id]
        if job.arrival_time < now:
            e_now, c_now = self._expected_cost(now_site.site_id, now, (e_now, c_now))

        # Candidate buckets inside the hold window.
        last = now + min(self.max_hold, self.forecast_horizon)
        if self.horizon_end is not None:
            last = min(last, self.horizon_end - self.end_margin)
        candidates = []
        for site in sites:
            bucket = now + STEP
            while bucket <= last:
                if self._deadline_ok(job, site, bucket):
                    values = self._signal(site.site_id, bucket)
                    if values is not None and self._fits(job, site, bucket):
                        e_b, c_b = self._expected_cost(site.site_id, bucket, values)
                        s_e = (e_now - e_b) / (abs(e_now) + EPS)
                        s_c = (c_now - c_b) / (abs(c_now) + EPS)
                        saving = self.w_energy * s_e + self.w_carbon * s_c
                        delay_h = (bucket - now).total_seconds() / 3600.0
                        candidates.append({"site": site, "bucket": bucket, "energy": e_b, "carbon": c_b,
                                           "delay": delay_h, "saving_energy": s_e,
                                           "saving_carbon": s_c, "saving": saving,
                                           "score": saving - self.delay_penalty * delay_h})
                bucket += STEP

        front = self._pareto_front(candidates)
        best = None
        for cand in front:
            if best is None or cand["score"] > best["score"]:
                best = cand
        if best is not None:
            best["front_size"] = len(front)

        if best is not None and best["saving"] > self.saving_threshold:
            self._reserve(job, best["site"], best["bucket"])
            self.counters["deferred"] += 1
            self._log(job, now, best["site"].site_id, best["bucket"], e_now, c_now, best)
            return DispatchDecision(
                job_id=job.job_id, site_id=best["site"].site_id, dispatch_at=best["bucket"],
                rationale=(f"green_grm_pareto: hold {int((best['bucket'] - now).total_seconds() // 60)} min, "
                           f"predicted saving {best['saving']:.3f} "
                           f"(energy {best['saving_energy']:+.3f}, carbon {best['saving_carbon']:+.3f})"))
        return self._immediate(job, now_site, now, (e_now, c_now),
                               "no candidate above the saving threshold")

    def _immediate(self, job: Job, site: Site, now: datetime,
                   costs: Optional[Tuple[float, float]], reason: str) -> DispatchDecision:
        self.counters["immediate"] += 1
        if costs is not None:
            self._log(job, now, site.site_id, now, costs[0], costs[1], None)
        return DispatchDecision(job_id=job.job_id, site_id=site.site_id, dispatch_at=now,
                                rationale=f"green_grm_pareto: immediate ({reason})")

    @staticmethod
    def _pareto_front(candidates: List[Dict]) -> List[Dict]:
        """Candidates that are not dominated in (predicted energy, predicted
        carbon, delay). With non-negative weights the trade-off maximiser
        always lies on this front; the front size is kept in the decision
        log to show how many efficient options a job had."""
        front: List[Dict] = []
        # Candidates are generated in order of increasing delay, so a later
        # candidate can only be dominated by an earlier (or equally late) one
        # with lower or equal energy and carbon.
        for cand in candidates:
            dominated = any(
                other["energy"] <= cand["energy"] and other["carbon"] <= cand["carbon"]
                and other["delay"] <= cand["delay"]
                and (other["energy"] < cand["energy"] or other["carbon"] < cand["carbon"]
                     or other["delay"] < cand["delay"])
                for other in front)
            if not dominated:
                front = [o for o in front if not (
                    cand["energy"] <= o["energy"] and cand["carbon"] <= o["carbon"]
                    and cand["delay"] <= o["delay"]
                    and (cand["energy"] < o["energy"] or cand["carbon"] < o["carbon"]
                         or cand["delay"] < o["delay"]))]
                front.append(cand)
        return front

    def _log(self, job: Job, now: datetime, site_id: str, dispatch_at: datetime,
             e_now: float, c_now: float, chosen: Optional[Dict]) -> None:
        if not self.record_decisions:
            return
        self.decisions.append({
            "job_id": job.job_id,
            "decided_at": now.isoformat(),
            "site_id": site_id,
            "dispatch_at": dispatch_at.isoformat(),
            "hold_minutes": (dispatch_at - now).total_seconds() / 60.0,
            "pred_energy_now": e_now,
            "pred_carbon_now": c_now,
            "pred_energy_dispatch": chosen["energy"] if chosen else e_now,
            "pred_carbon_dispatch": chosen["carbon"] if chosen else c_now,
            "pred_saving": chosen["saving"] if chosen else 0.0,
            "pareto_front_size": chosen.get("front_size") if chosen else None,
        })
