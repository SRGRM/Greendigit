"""
dirac_sim.backends.dirac_backend
=================================
Real-world DIRAC WMS integration.

Wraps `dirac_sim.core.wms.Backend` to submit, poll and cancel jobs via the
DIRAC Python API (diracgrid.org).  The REST forecast API is called to inform
the DIRAC JobOptimisationAgent before each pilot submission.

Requirements
------------
    pip install DIRAC          # or install via DIRACOS2 / conda-forge

Authentication
--------------
DIRAC uses X.509 proxy certificates.  The proxy must be initialised before
calling any DIRAC API:

    $ dirac-proxy-init --legacy

Alternatively, supply DIRAC_PROXY_PATH and DIRAC_SERVER_URL environment
variables for container-based deployments.

Usage
-----
>>> from dirac_sim.backends.dirac_backend import DiracBackend
>>> from dirac_sim.api.forecast_client import ForecastClient
>>> from dirac_sim.baselines.greedy_carbon import GreedyCarbonScheduler
>>> from dirac_sim.core.wms import WMSSimulator
>>>
>>> backend = DiracBackend(
...     vo="gridpp",
...     default_site="LCG.UKI-SOUTHGRID-OX-HEP.uk",
...     forecast_client=ForecastClient(base_url="http://my-model:8000"),
... )
>>> sim = WMSSimulator(
...     queue=queue, registry=registry,
...     scheduler=GreedyCarbonScheduler(),
...     backend=backend,
... )
>>> report = sim.run()
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Optional

from dirac_sim.core.job_queue import Job
from dirac_sim.core.wms import Backend

logger = logging.getLogger(__name__)


class DiracBackend(Backend):
    """
    Production DIRAC backend.

    Wraps DIRAC's Python API to:
      1. Convert dirac_sim Job objects into DIRAC Job JDL descriptions.
      2. Submit them to the DIRAC Workload Management System.
      3. Poll job status and propagate state back to the simulator.
      4. Call the REST forecast API to attach carbon-aware metadata to each
         pilot, which can then be read by a custom DIRAC JobOptimisationAgent.

    Parameters
    ----------
    vo              : Virtual Organisation name (e.g. "gridpp", "lhcb").
    default_site    : Fallback site if none specified by the scheduler.
    forecast_client : REST client for the forecast API (optional).
    executable      : Default payload executable inside the pilot.
    cpu_time        : CPU time limit in seconds (DIRAC JDL CPUTime).
    """

    def __init__(
        self,
        vo: str = "gridpp",
        default_site: str = "",
        forecast_client=None,
        executable: str = "/bin/echo",
        cpu_time: int = 3600,
    ) -> None:
        self.vo = vo
        self.default_site = default_site
        self.forecast_client = forecast_client
        self.executable = executable
        self.cpu_time = cpu_time
        self._dirac = None       # lazy init
        self._handles: Dict[str, str] = {}   # job_id -> DIRAC jobID

    # ------------------------------------------------------------------ #
    # Backend interface                                                    #
    # ------------------------------------------------------------------ #
    def submit(self, job: Job, site_id: str, dispatch_time: datetime) -> str:
        """
        Submit a job to DIRAC WMS and return the DIRAC job ID as handle.
        """
        api = self._get_dirac()

        # Attach forecast metadata as DIRAC job parameters
        carbon_tag = self._get_carbon_tag(job, dispatch_time)

        dirac_job = self._build_dirac_job(job, site_id, carbon_tag)

        result = api.submit(dirac_job)
        if not result.get("OK", False):
            raise RuntimeError(f"DIRAC submit failed: {result.get('Message')}")

        handle = str(result["Value"])
        self._handles[job.job_id] = handle
        logger.info("DIRAC submitted job_id=%s -> DIRAC_ID=%s at site=%s",
                    job.job_id, handle, site_id)
        return handle

    def poll(self, handle: str) -> Optional[str]:
        """
        Poll DIRAC job status.
        Returns: 'done' | 'running' | 'failed' | None
        """
        api = self._get_dirac()
        result = api.status(int(handle))
        if not result.get("OK", False):
            logger.warning("DIRAC status poll failed for handle %s", handle)
            return None

        dirac_status = result["Value"].get(int(handle), {}).get("Status", "")
        mapping = {
            "Done": "done",
            "Completed": "done",
            "Running": "running",
            "Failed": "failed",
            "Stalled": "failed",
            "Waiting": "running",
            "Received": "running",
            "Checking": "running",
            "Staging": "running",
        }
        return mapping.get(dirac_status)

    def cancel(self, handle: str) -> None:
        api = self._get_dirac()
        result = api.delete([int(handle)])
        if result.get("OK"):
            logger.info("Cancelled DIRAC job %s", handle)
        else:
            logger.warning("Failed to cancel DIRAC job %s: %s",
                           handle, result.get("Message"))

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _get_dirac(self):
        """Lazy initialise DIRAC API object."""
        if self._dirac is None:
            try:
                from DIRAC import initialize
                initialize()
                from DIRAC.Interfaces.API.Dirac import Dirac
                self._dirac = Dirac()
                logger.info("DIRAC API initialised (VO=%s)", self.vo)
            except ImportError:
                raise ImportError(
                    "DIRAC is not installed.  Install via DIRACOS2 or "
                    "conda-forge: mamba install -c conda-forge dirac-grid"
                )
        return self._dirac

    def _build_dirac_job(self, job: Job, site_id: str,
                         carbon_tag: str) -> "DIRAC.Interfaces.API.Job.Job":
        """Convert a dirac_sim Job to a DIRAC Job object."""
        from DIRAC.Interfaces.API.Job import Job as DiracJob

        j = DiracJob()
        j.setName(job.job_id)
        j.setExecutable(self.executable, arguments=job.job_id)
        j.setCPUTime(int(job.cpu_minutes * 60))

        if site_id:
            j.setDestination([site_id])
        elif self.default_site:
            j.setDestination([self.default_site])

        if job.memory_gb:
            j.setJobParameters([("Memory", str(int(job.memory_gb * 1024)))])

        # Carbon metadata as DIRAC job parameter (visible in monitoring portal)
        if carbon_tag:
            j.setJobParameters([("CarbonTag", carbon_tag)])

        # Priority mapping
        prio_map = {"express": 10, "normal": 5, "best_effort": 1}
        j.setPriority(prio_map.get(job.priority.value, 5))

        return j

    def _get_carbon_tag(self, job: Job, dispatch_time: datetime) -> str:
        """
        Fetch carbon forecast for this dispatch window and encode as a tag.
        The tag is attached to the DIRAC job for monitoring purposes.
        """
        if self.forecast_client is None:
            return ""
        try:
            bundle = self.forecast_client.fetch_bundle(dispatch_time)
            by_series = bundle.by_series("1h")
            records = by_series.get(job.series_id, [])
            if records:
                cfp = records[0].get("cfp_g_pred", "?")
                energy = records[0].get("energy_wh_pred", "?")
                return f"cfp={cfp:.1f}g,energy={energy:.1f}Wh"
        except Exception as exc:
            logger.debug("Carbon tag fetch failed: %s", exc)
        return ""


# ------------------------------------------------------------------ #
# DIRAC JobOptimisationAgent hook (advanced)                           #
# ------------------------------------------------------------------ #
DIRAC_AGENT_TEMPLATE = '''
# dirac_carbon_optimizer.py
# Drop this file into your DIRAC agent directory as a custom optimizer.
#
# It hooks into DIRAC's JobOptimisationAgent pipeline to:
#   1. Fetch carbon forecasts from the REST API before each match-making cycle.
#   2. Filter out high-carbon windows by delaying dispatch.
#
# Install: copy to $DIRAC/WorkloadManagementSystem/Agent/
# Configure in dirac.cfg:
#   [Systems/WorkloadManagement/Agents]
#     CarbonOptimizer
#     {
#       ForecastAPIURL = http://my-model-server:8000
#       CarbonThreshold = 200   # gCO2eq/15min
#       Module = CarbonOptimizer
#     }

import requests
from DIRAC.WorkloadManagementSystem.DB.JobDB import JobDB
from DIRAC import S_OK, S_ERROR, gConfig

FORECAST_URL = gConfig.getValue(
    "/Systems/WorkloadManagement/Agents/CarbonOptimizer/ForecastAPIURL",
    "http://localhost:8000"
)
CARBON_THRESHOLD = float(gConfig.getValue(
    "/Systems/WorkloadManagement/Agents/CarbonOptimizer/CarbonThreshold",
    "200"
))


def is_green_window(series_id, now):
    """Return True if forecast carbon is below threshold."""
    try:
        resp = requests.post(
            f"{FORECAST_URL}/forecast",
            json={
                "series_ids": [series_id],
                "reference_timestamp_utc": now.isoformat(),
                "horizons": ["1h"],
            },
            timeout=5,
        )
        resp.raise_for_status()
        forecasts = resp.json().get("forecasts", [])
        if forecasts:
            return float(forecasts[0]["cfp_g_pred"]) < CARBON_THRESHOLD
    except Exception:
        pass
    return True   # fail-open: dispatch if API unavailable
'''
