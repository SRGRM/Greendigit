"""
dirac_sim.backends.slurm_backend
=================================
Real-world SLURM HPC cluster integration.

Maps dirac_sim scheduling decisions to SLURM `sbatch` / `scontrol` calls.
The REST forecast API is queried before each submission to:
  1. Decide whether to submit immediately or hold the job.
  2. Annotate the SLURM job with carbon metadata via `--comment`.
  3. Optionally set `--begin` to shift the job to a greener time window.

Requirements
------------
    SLURM command-line tools: sbatch, scontrol, sacct, scancel
    (or optionally: pip install pyslurm)

Environment variables
---------------------
SLURM_PARTITION        : Default SLURM partition.
SLURM_ACCOUNT          : Default SLURM account.
FORECAST_API_BASE_URL  : Forecast REST API base URL.
SLURM_USE_PYSLURM      : Set to "1" to use pyslurm instead of subprocess.

Usage
-----
>>> from dirac_sim.backends.slurm_backend import SlurmBackend
>>> backend = SlurmBackend(
...     partition="hpc-green",
...     account="sustainability-proj",
...     forecast_client=ForecastClient(base_url="http://my-model:8000"),
... )
>>> sim = WMSSimulator(queue=queue, registry=registry,
...                   scheduler=scheduler, backend=backend)
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import textwrap
from datetime import datetime, timedelta
from typing import Dict, Optional

from dirac_sim.core.job_queue import Job
from dirac_sim.core.wms import Backend

logger = logging.getLogger(__name__)


class SlurmBackend(Backend):
    """
    SLURM backend.

    Submits jobs via `sbatch` with optional temporal shifting (`--begin`)
    derived from the forecast API's green-window recommendations.

    Parameters
    ----------
    partition       : SLURM partition name.
    account         : SLURM account / project.
    forecast_client : REST client for forecast API (optional).
    use_pyslurm     : Use pyslurm library instead of subprocess (if available).
    green_threshold : Carbon intensity threshold (gCO2eq) to trigger holding.
    max_hold_hours  : Maximum hours to defer a job looking for a green window.
    """

    def __init__(
        self,
        partition: str = "",
        account: str = "",
        forecast_client=None,
        use_pyslurm: bool = False,
        green_threshold: float = 200.0,
        max_hold_hours: float = 6.0,
    ) -> None:
        self.partition = partition or os.environ.get("SLURM_PARTITION", "batch")
        self.account = account or os.environ.get("SLURM_ACCOUNT", "")
        self.forecast_client = forecast_client
        self.use_pyslurm = use_pyslurm and self._pyslurm_available()
        self.green_threshold = green_threshold
        self.max_hold_hours = max_hold_hours
        self._handles: Dict[str, str] = {}   # job_id -> SLURM job ID

    # ------------------------------------------------------------------ #
    # Backend interface                                                    #
    # ------------------------------------------------------------------ #
    def submit(self, job: Job, site_id: str, dispatch_time: datetime) -> str:
        """Submit a SLURM batch job; return SLURM job ID as handle."""
        begin_time = self._find_green_begin(job, dispatch_time)
        script = self._render_script(job, site_id, begin_time)

        if self.use_pyslurm:
            handle = self._submit_pyslurm(script, job, begin_time)
        else:
            handle = self._submit_subprocess(script, job, begin_time)

        self._handles[job.job_id] = handle
        logger.info("SLURM submitted job_id=%s -> slurm_id=%s "
                    "(begin=%s, site=%s)",
                    job.job_id, handle,
                    begin_time.isoformat() if begin_time else "now",
                    site_id)
        return handle

    def poll(self, handle: str) -> Optional[str]:
        """Poll job state via sacct."""
        try:
            result = subprocess.run(
                ["sacct", "-j", handle, "--format=State", "--noheader",
                 "--parsable2"],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l.strip() for l in result.stdout.strip().splitlines()
                     if l.strip()]
            if not lines:
                return None
            state = lines[0].split("|")[0].upper()
            mapping = {
                "COMPLETED": "done",
                "RUNNING": "running",
                "PENDING": "running",
                "FAILED": "failed",
                "CANCELLED": "failed",
                "TIMEOUT": "failed",
                "NODE_FAIL": "failed",
            }
            return mapping.get(state)
        except Exception as exc:
            logger.warning("sacct poll failed for %s: %s", handle, exc)
            return None

    def cancel(self, handle: str) -> None:
        """Cancel a SLURM job."""
        try:
            subprocess.run(["scancel", handle], check=True, timeout=10)
            logger.info("Cancelled SLURM job %s", handle)
        except Exception as exc:
            logger.warning("scancel failed for %s: %s", handle, exc)

    # ------------------------------------------------------------------ #
    # Green window selection via forecast API                             #
    # ------------------------------------------------------------------ #
    def _find_green_begin(self, job: Job,
                          now: datetime) -> Optional[datetime]:
        """
        Query the 24h forecast to find the earliest green window for this job.

        Returns a `--begin` datetime if a greener future window is found
        within max_hold_hours, otherwise None (dispatch immediately).
        """
        if self.forecast_client is None:
            return None

        try:
            bundle = self.forecast_client.fetch_bundle(now)
            by_series = bundle.by_series("24h")
            records = by_series.get(job.series_id, [])

            # Find first future bucket below threshold within hold horizon
            max_hold = timedelta(hours=self.max_hold_hours)
            for rec in sorted(records,
                               key=lambda r: r["forecast_timestamp_utc"]):
                ts = datetime.fromisoformat(
                    rec["forecast_timestamp_utc"].replace("Z", "+00:00"))
                if ts > now + max_hold:
                    break
                cfp = float(rec.get("cfp_g_pred", 9999))
                if cfp < self.green_threshold and ts > now:
                    logger.debug("Green window for job %s at %s (cfp=%.1f)",
                                 job.job_id, ts.isoformat(), cfp)
                    return ts
        except Exception as exc:
            logger.debug("Green window lookup failed: %s", exc)
        return None

    # ------------------------------------------------------------------ #
    # Script generation                                                    #
    # ------------------------------------------------------------------ #
    def _render_script(self, job: Job, site_id: str,
                       begin_time: Optional[datetime]) -> str:
        """
        Generate a minimal SLURM batch script for the job.

        Participants should override this to generate application-specific
        scripts; the default produces a no-op placeholder.
        """
        lines = ["#!/bin/bash"]
        lines.append(f"#SBATCH --job-name={job.job_id}")
        lines.append(f"#SBATCH --partition={self.partition}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        lines.append(f"#SBATCH --time={int(job.cpu_minutes)}:00")
        lines.append(f"#SBATCH --mem={int(job.memory_gb * 1024)}M")
        lines.append(f"#SBATCH --comment=series_id={job.series_id}")
        if begin_time:
            lines.append(
                f"#SBATCH --begin={begin_time.strftime('%Y-%m-%dT%H:%M:%S')}")
        lines.append("")
        lines.append("# --- Participant payload below ---")
        lines.append(f"echo 'Running job {job.job_id} on site {site_id}'")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Submission helpers                                                   #
    # ------------------------------------------------------------------ #
    def _submit_subprocess(self, script: str, job: Job,
                           begin_time: Optional[datetime]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                        delete=False) as fh:
            fh.write(script)
            script_path = fh.name

        cmd = ["sbatch", "--parsable", script_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=30, check=True)
            slurm_id = result.stdout.strip().split(";")[0]
            return slurm_id
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"sbatch failed for job {job.job_id}: {exc.stderr}") from exc
        finally:
            os.unlink(script_path)

    def _submit_pyslurm(self, script: str, job: Job,
                        begin_time: Optional[datetime]) -> str:
        """Alternative submission via pyslurm (requires pip install pyslurm)."""
        try:
            import pyslurm
        except ImportError:
            raise ImportError("pyslurm not installed; "
                              "unset SLURM_USE_PYSLURM or pip install pyslurm")
        desc = pyslurm.job_submit_description()
        desc.name = job.job_id
        desc.partition = self.partition
        desc.time_limit = int(job.cpu_minutes)
        desc.memory_per_node = int(job.memory_gb * 1024)
        if begin_time:
            desc.begin_time = int(begin_time.timestamp())
        job_id = pyslurm.job().submit_batch_job(desc)
        return str(job_id)

    @staticmethod
    def _pyslurm_available() -> bool:
        try:
            import pyslurm  # noqa: F401
            return True
        except ImportError:
            return False


# ------------------------------------------------------------------ #
# SLURM Prolog / Epilog hooks for carbon metadata                     #
# ------------------------------------------------------------------ #
SLURM_PROLOG_TEMPLATE = textwrap.dedent("""
    #!/bin/bash
    # slurm_carbon_prolog.sh
    # Install as SLURM PrologSlurmctld in slurm.conf:
    #   PrologSlurmctld=/path/to/slurm_carbon_prolog.sh
    #
    # Fetches current carbon intensity from the forecast API and writes it
    # to the job's environment, making it visible to the payload.

    FORECAST_URL="${FORECAST_API_BASE_URL:-http://localhost:8000}"
    SERIES_ID=$(scontrol show job $SLURM_JOB_ID | grep -oP 'series_id=\\K[^,]+')

    if [ -n "$SERIES_ID" ]; then
        CARBON=$(curl -s -X POST "$FORECAST_URL/forecast" \\
            -H "Content-Type: application/json" \\
            -d '{"series_ids":["'$SERIES_ID'"],"reference_timestamp_utc":"'$(date -u +%Y-%m-%dT%H:%M:%S)'","horizons":["1h"]}' \\
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['forecasts'][0]['cfp_g_pred'])" 2>/dev/null)
        echo "SLURM_CARBON_INTENSITY=$CARBON" >> /etc/environment
        logger -t slurm_prolog "job $SLURM_JOB_ID carbon_intensity=$CARBON"
    fi
""")
