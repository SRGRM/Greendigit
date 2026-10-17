# dirac_sim/__init__.py
"""
dirac_sim — Starter kit for ECML PKDD Challenge Task B
=======================================================
Forecast-driven sustainable job scheduling for distributed computing.

Quick start
-----------
    from dirac_sim.core.job_queue import JobQueue
    from dirac_sim.core.site_model import SiteRegistry
    from dirac_sim.core.wms import WMSSimulator
    from dirac_sim.api.forecast_client import ForecastClient
    from dirac_sim.baselines.greedy_carbon import GreedyCarbonScheduler

    queue    = JobQueue.from_csv("data/job_trace.csv")
    registry = SiteRegistry.from_json("data/site_config.json")
    client   = ForecastClient(base_url="http://localhost:8000",
                               offline_csv="data/forecast_baseline.csv")
    sim = WMSSimulator(queue=queue, registry=registry,
                       scheduler=GreedyCarbonScheduler(),
                       forecast_client=client)
    report = sim.run()
    print(report.summary())
"""
__version__ = "0.1.0"
