"""
tests/test_core.py
==================
Unit tests for core components.
Run with: pytest tests/ -v
"""
import pytest
from datetime import datetime, timedelta, timezone
from dirac_sim.core.job_queue import Job, JobQueue, Priority, JobStatus
from dirac_sim.core.site_model import Site, SiteCapacity, SiteRegistry
from dirac_sim.core.scheduler import (
    ForecastBundle, DispatchDecision, DispatchPlan, Objective,
)
from dirac_sim.core.evaluator import Evaluator
from dirac_sim.core.wms import WMSSimulator, SimulationReport, ExecutionRecord


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #
T0 = datetime(2026, 2, 17, 14, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(hours=24)


def make_job(job_id="j1", series_id="CE1", cpu_min=30.0,
             hours_to_deadline=12, priority="normal") -> Job:
    return Job(
        job_id=job_id,
        series_id=series_id,
        arrival_time=T0,
        deadline=T0 + timedelta(hours=hours_to_deadline),
        cpu_minutes=cpu_min,
        priority=Priority(priority),
    )


def make_site(site_id="CE1", carbon=100.0, energy=500.0,
              slots=100) -> Site:
    site = Site(
        site_id=site_id,
        name=site_id,
        capacity=SiteCapacity(max_concurrent_jobs=slots),
    )
    # Populate a few buckets
    for i in range(200):
        t = T0 + timedelta(minutes=15 * i)
        site.update_signals(t, energy, carbon)
    return site


def make_registry(*sites) -> SiteRegistry:
    reg = SiteRegistry()
    for s in sites:
        reg.register(s)
    return reg


# ------------------------------------------------------------------ #
# JobQueue tests                                                       #
# ------------------------------------------------------------------ #
class TestJobQueue:
    def test_add_and_ready(self):
        q = JobQueue()
        j = make_job()
        q.add(j)
        assert len(q.ready_jobs(T0)) == 1

    def test_not_ready_before_arrival(self):
        q = JobQueue()
        j = make_job()
        q.add(j)
        assert len(q.ready_jobs(T0 - timedelta(minutes=1))) == 0

    def test_priority_ordering(self):
        q = JobQueue()
        q.add(make_job("j_be", priority="best_effort"))
        q.add(make_job("j_ex", priority="express"))
        q.add(make_job("j_no", priority="normal"))
        ready = q.ready_jobs(T0)
        assert [j.job_id for j in ready] == ["j_ex", "j_no", "j_be"]

    def test_status_update(self):
        q = JobQueue()
        j = make_job()
        q.add(j)
        q.update_status(j.job_id, JobStatus.DISPATCHED,
                        dispatch_time=T0, site_id="CE1")
        updated = [x for x in q.all_jobs() if x.job_id == j.job_id][0]
        assert updated.status == JobStatus.DISPATCHED
        assert updated.site_id == "CE1"

    def test_csv_roundtrip(self, tmp_path):
        q = JobQueue()
        for i in range(5):
            q.add(make_job(f"j{i}", series_id=f"CE{i%4+1}"))
        path = tmp_path / "jobs.csv"
        q.to_csv(str(path))
        q2 = JobQueue.from_csv(str(path))
        assert len(q2.all_jobs()) == 5


# ------------------------------------------------------------------ #
# SiteModel tests                                                      #
# ------------------------------------------------------------------ #
class TestSiteModel:
    def test_get_energy_with_pue(self):
        site = make_site(energy=1000.0)
        site.pue = 1.5
        # expect 1000 * 1.5
        assert site.get_energy(T0) == pytest.approx(1500.0)

    def test_get_carbon(self):
        site = make_site(carbon=150.0)
        assert site.get_carbon(T0) == pytest.approx(150.0)

    def test_fallback_on_missing_bucket(self):
        site = Site(site_id="CE_EMPTY", name="empty")
        val = site.get_carbon(T0, fallback=999.0)
        assert val == 999.0

    def test_green_windows(self):
        site = make_site(carbon=300.0)
        # Overwrite some buckets to be green
        for i in range(4):  # 4 x 15min = 1h green window
            t = T0 + timedelta(minutes=15 * (i + 4))
            site.update_signals(t, 500.0, 50.0)  # low carbon

        windows = site.green_windows(T0, 180, carbon_threshold=100.0)
        assert len(windows) >= 1

    def test_cheapest_carbon_site(self):
        reg = make_registry(
            make_site("A", carbon=50.0),
            make_site("B", carbon=200.0),
            make_site("C", carbon=150.0),
        )
        best = reg.cheapest_carbon_site(T0, reg.all_sites())
        assert best.site_id == "A"

    def test_available_sites_whitelist(self):
        reg = make_registry(
            make_site("A"), make_site("B"), make_site("C"))
        available = reg.available_sites(["A", "C"])
        assert {s.site_id for s in available} == {"A", "C"}

    def test_site_registry_json_roundtrip(self, tmp_path):
        reg = make_registry(make_site("CE1"), make_site("CE2"))
        path = tmp_path / "sites.json"
        reg.to_json(str(path))
        reg2 = SiteRegistry.from_json(str(path))
        assert {s.site_id for s in reg2.all_sites()} == {"CE1", "CE2"}


# ------------------------------------------------------------------ #
# Scheduler / ForecastBundle tests                                     #
# ------------------------------------------------------------------ #
class TestForecastBundle:
    def test_by_series_grouping(self):
        bundle = ForecastBundle(
            tick_time=T0,
            horizon_1h=[
                {"series_id": "CE1", "forecast_timestamp_utc": T0.isoformat(),
                 "energy_wh_pred": 400, "cfp_g_pred": 100},
                {"series_id": "CE2", "forecast_timestamp_utc": T0.isoformat(),
                 "energy_wh_pred": 600, "cfp_g_pred": 200},
            ],
        )
        grouped = bundle.by_series("1h")
        assert set(grouped.keys()) == {"CE1", "CE2"}
        assert len(grouped["CE1"]) == 1

    def test_empty_bundle(self):
        bundle = ForecastBundle(tick_time=T0)
        assert bundle.by_series("1h") == {}
        assert bundle.by_series("24h") == {}


# ------------------------------------------------------------------ #
# WMS Simulator tests                                                  #
# ------------------------------------------------------------------ #
class TestWMSSimulator:
    def _run(self, scheduler, jobs=None, sites=None):
        from dirac_sim.baselines.fcfs import FCFSScheduler
        q = JobQueue()
        for j in (jobs or [make_job()]):
            q.add(j)
        reg = make_registry(*(sites or [make_site()]))
        sim = WMSSimulator(
            queue=q, registry=reg,
            scheduler=scheduler or FCFSScheduler(),
            start_time=T0,
            end_time=T0 + timedelta(hours=2),
        )
        return sim.run()

    def test_fcfs_dispatches_all(self):
        from dirac_sim.baselines.fcfs import FCFSScheduler
        jobs = [make_job(f"j{i}") for i in range(10)]
        report = self._run(FCFSScheduler(), jobs=jobs)
        assert len(report.records) == 10

    def test_greedy_carbon_runs(self):
        from dirac_sim.baselines.greedy_carbon import GreedyCarbonScheduler
        report = self._run(GreedyCarbonScheduler())
        assert len(report.records) >= 1

    def test_deadline_tracked(self):
        from dirac_sim.baselines.fcfs import FCFSScheduler
        # Job with very tight deadline (1 min)
        tight_job = make_job("tight", hours_to_deadline=0.01)
        # Job with normal deadline
        normal_job = make_job("normal", hours_to_deadline=12)
        report = self._run(FCFSScheduler(),
                           jobs=[tight_job, normal_job])
        recs = {r.job_id: r for r in report.records}
        # normal job should meet deadline
        if "normal" in recs:
            assert recs["normal"].deadline_met is True

    def test_site_capacity_respected(self):
        from dirac_sim.baselines.fcfs import FCFSScheduler
        # Site with only 1 slot
        small_site = make_site("SMALL", slots=1)
        jobs = [make_job(f"j{i}") for i in range(5)]
        # Should still process jobs (over multiple ticks)
        report = self._run(FCFSScheduler(), jobs=jobs,
                           sites=[small_site])
        assert len(report.records) >= 1

    def test_capacity_is_held_until_job_end(self):
        from dirac_sim.baselines.fcfs import FCFSScheduler
        small_site = make_site("CE1", slots=1)
        jobs = [make_job("j1", cpu_min=60), make_job("j2", cpu_min=60)]
        report = self._run(FCFSScheduler(), jobs=jobs, sites=[small_site])
        recs = sorted(report.records, key=lambda r: r.dispatch_time)
        assert len(recs) == 2
        assert recs[1].dispatch_time >= recs[0].end_time


# ------------------------------------------------------------------ #
# Evaluator tests                                                      #
# ------------------------------------------------------------------ #
class TestEvaluator:
    def _make_report(self, n=10, energy=1000.0, carbon=500.0,
                     makespan=60.0, deadline_met=True):
        records = [
            ExecutionRecord(
                job_id=f"j{i}",
                series_id="CE1",
                site_id="CE1",
                dispatch_time=T0,
                start_time=T0 + timedelta(minutes=2),
                end_time=T0 + timedelta(minutes=2 + makespan),
                energy_wh=energy / n,
                cfp_g=carbon / n,
                deadline_met=deadline_met,
            )
            for i in range(n)
        ]
        q = JobQueue()
        return SimulationReport(records=records, queue=q)

    def test_perfect_carbon_score(self):
        baseline = self._make_report(energy=1000, carbon=500, makespan=60)
        # Submission with 50% less carbon (energy/makespan unchanged)
        # Pareto: 1 - (1/3*(1-0) + 1/3*(1-0.5) + 1/3*(1-0)) = 0.167
        # + declaration bonus 0.15*0.5 = 0.075  → ~0.24
        submission = self._make_report(energy=1000, carbon=250, makespan=60)
        result = Evaluator.score(submission, baseline, "carbon")
        assert result.delta_carbon_norm == pytest.approx(0.5)
        assert result.final_score > 0.20   # correctly ~0.24

    def test_worse_than_baseline(self):
        baseline = self._make_report(energy=1000, carbon=500, makespan=60)
        # Submission using 2x energy (much worse)
        submission = self._make_report(energy=2000, carbon=500, makespan=60)
        result = Evaluator.score(submission, baseline, "energy")
        assert result.delta_energy_norm < 0   # negative = worse

    def test_deadline_penalty(self):
        baseline = self._make_report()
        submission = self._make_report(deadline_met=False)
        result = Evaluator.score(submission, baseline, "carbon")
        assert result.penalty_deadline > 0

    def test_never_dispatched_jobs_are_penalized(self):
        baseline = self._make_report(n=2)
        submission = self._make_report(n=1)
        submission.queue.add(make_job("j0"))
        submission.queue.add(make_job("j1"))
        result = Evaluator.score(submission, baseline, "carbon")
        assert result.submission.n_jobs == 2
        assert result.submission.jobs_dispatched == 1
        assert result.penalty_deadline > 0

    def test_oracle_gap(self):
        baseline = self._make_report(carbon=500)
        oracle   = self._make_report(carbon=100)    # oracle: 80% carbon reduction
        submission = self._make_report(carbon=300)  # submission: 40% carbon reduction
        gap = Evaluator.compare_with_oracle(submission, oracle, baseline,
                                            "carbon")
        assert 0 < gap["capture_ratio"] < 1.0
        assert gap["value_gap"] > 0

    def test_zero_baseline_safe(self):
        from dirac_sim.core.wms import SimulationReport
        empty = SimulationReport(records=[], queue=JobQueue())
        sub = self._make_report()
        result = Evaluator.score(sub, empty, "carbon")
        assert result.final_score == pytest.approx(0.0)
