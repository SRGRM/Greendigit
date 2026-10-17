# Task B Scheduling Framework

Task B asks participants to build a scheduler that uses Task A forecasts to
decide when and where grid or HPC jobs should run. The goal is to reduce
energy, carbon, and makespan relative to the FCFS baseline while respecting job
deadlines and site constraints.

This folder contains the simulator, scheduler interface, baselines, local
evaluator, forecast API client, and participant scheduler template.

## Install

Use an isolated virtual environment inside `task-b/`.

```bash
cd task-b
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,server]"
```

## Inputs

Simulator inputs live in the repository root `data/` directory:

- `data/job_trace.csv`: aggregate job trace, one job per site, VO, and 15-minute bucket.
- `data/site_config.json`: anonymized site registry with exact site IDs and capacity limits.
- `data/forecast_baseline.csv`: offline forecast source using summarized site buckets.

Regenerate these files from the repository root after replacing
`data/raw_metrics/`:

```bash
python3 data/prepare_data.py
```

The site IDs in `job_trace.csv`, `site_config.json`, and forecast responses must
match exactly. The generated trace pins each aggregate job to its source site
through `site_whitelist`.

## Baseline E2E Run

From `task-b/`, run the offline simulator without a live model server:

```bash
./.venv/bin/python -m dirac_sim simulate --offline
```

By default this runs a 24-hour raw-metrics smoke window:
`2025-11-19T23:00:00` through `2025-11-20T23:00:00`.

Run a longer local window with `--start` and `--end`:

```bash
./.venv/bin/python -m dirac_sim simulate --offline \
  --start 2025-11-19T23:00:00 \
  --end 2026-03-12T17:00:00
```

The default `greedy_carbon` scheduler can tie FCFS on the smoke window. To
confirm that scoring responds to changed dispatches, run:

```bash
./.venv/bin/python -m dirac_sim simulate --offline \
  --scheduler test_policy \
  --output-dir /tmp/taskb_test_policy
```

`test_policy` intentionally ignores job site whitelists and is not a valid
submission policy. It is only a local scoring sanity check.

## Outputs

The runner writes to `results/` unless `--output-dir` is provided:

- `baseline_execution_report.csv`: FCFS baseline execution records.
- `execution_report.csv`: selected scheduler execution records.
- `dispatch_log.csv`: participant-facing dispatch decisions.
- `score_summary.txt`: human-readable evaluator output.
- `score_metrics.json`: machine-readable run metadata, objective totals,
  deltas, penalties, and final score.

## Implement Your Scheduler

Start from [examples/custom_scheduler_template.py](examples/custom_scheduler_template.py).
Implement a class that subclasses `dirac_sim.core.scheduler.Scheduler` and
returns a `DispatchPlan` from `schedule()`.

Run a scheduler by dotted class path:

```bash
./.venv/bin/python -m dirac_sim simulate --offline \
  --scheduler my_package.my_scheduler.MyScheduler \
  --objective carbon
```

The scheduler receives:

- `queue`: pending jobs, priorities, deadlines, and site whitelists.
- `registry`: available sites, capacities, and current energy/carbon signals.
- `forecast`: 1-hour and 24-hour Task A forecast bundles.
- `now`: current simulation timestamp.

Jobs with no decision remain pending for the next tick. To defer a job, return
a `DispatchDecision` with `dispatch_at` set to a future timestamp.

## Task A Forecast Integration

Offline mode reads `data/forecast_baseline.csv`. Live mode calls a Task
A-compatible forecast service:

```bash
./.venv/bin/python -m dirac_sim simulate \
  --api-url http://localhost:8000 \
  --scheduler my_package.my_scheduler.MyScheduler
```

The server must implement:

```text
POST /forecast
```

Request:

```json
{
  "series_ids": ["site_a", "site_b"],
  "reference_timestamp_utc": "2025-11-20T00:00:00+00:00",
  "horizons": ["1h", "24h"]
}
```

Response:

```json
{
  "reference_timestamp_utc": "2025-11-20T00:00:00+00:00",
  "forecasts": [
    {
      "series_id": "site_a",
      "forecast_timestamp_utc": "2025-11-20T01:00:00+00:00",
      "horizon_steps_15m": 4,
      "energy_wh_pred": 412.5,
      "cfp_g_pred": 98.3
    }
  ]
}
```

The Task A baseline server in `../task-a` already implements this contract.

## What Participants May Change

Participants should add or modify their own scheduler code and supporting
policy/model code.

Do not modify these files for a submitted result:

- `dirac_sim/core/evaluator.py`
- `dirac_sim/core/wms.py`
- `dirac_sim/core/job_queue.py`
- `dirac_sim/core/site_model.py`
- `dirac_sim/core/scheduler.py`
- `tests/`
- bundled input data under `../data/`

The local evaluator is a convenience check. Official scoring should be run by
organizers from a clean checkout with the official evaluator and hidden data.

## Scoring

The local evaluator compares your scheduler report to FCFS:

```text
Score = 1 - (1/3)((1 - delta_energy) + (1 - delta_carbon) + (1 - delta_makespan))
      + 0.15 * max(0, delta_declared)
      - deadline_penalty
```

Where each normalized delta is `(baseline - submission) / baseline`; positive
values mean improvement over FCFS. The declaration bonus rewards improvement in
the objective selected by `--objective`.

The evaluator writes both text and JSON summaries. The important machine fields
are in `score_metrics.json` under `evaluation.scores` and
`evaluation.deltas`.

## Submission Artifacts

For local review, provide:

- the scheduler implementation,
- the command used to run it,
- `dispatch_log.csv`,
- `score_metrics.json`,
- any documented dependencies needed by the scheduler.

The simulator creates `dispatch_log.csv` automatically. Its columns are:

```text
job_id,dispatch_timestamp_utc,site_id,declared_objective,deadline_met,status
```

## Tests

```bash
cd task-b
./.venv/bin/python -m pytest
```

## Optional Deployment

The repository includes a Docker Compose setup for API and monitoring demos:

```bash
docker-compose up --build
```

Grafana is exposed at `http://localhost:3000` when the monitoring services are
enabled.
