# Task A Participant Guidelines

## Goal

Build a model that produces three CSV files:

- `forecast_submission.csv` for 1-hour and 24-hour `energy_wh` and `cfp_g` forecasts.
- `detection_submission.csv` for A.1 valid or invalid signal detection.
- `peak_submission.csv` for A.2 peak event detection.

Use the local evaluator to check format, coverage, and scoring behavior before
submitting.

## Required Commands

From `task-a/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,server]"
```

Run the reference baseline:

```bash
./.venv/bin/task-a train-baseline
./.venv/bin/task-a predict
./.venv/bin/task-a predict-detection
./.venv/bin/task-a predict-peaks
./.venv/bin/task-a evaluate \
  --forecasts outputs/forecast_submission.csv \
  --detections outputs/detection_submission.csv \
  --peaks outputs/peak_submission.csv
```

## Submission Rules

Do not change the evaluator, validators, schemas, tests, or official data to
improve your score. Organizer evaluation will use a clean copy of those files.

Allowed work:

- Add your own model code.
- Add scripts that train your model and write the three required CSVs.
- Add dependencies if they are documented and installable.
- Add tests for your own code.

Not allowed for submitted results:

- Editing `src/task_a/evaluation.py`.
- Editing `src/task_a/submission.py`.
- Editing `src/task_a/schemas.py`.
- Editing `config/eval.yaml`.
- Editing files under `../data/raw_metrics/`.
- Hard-coding hidden labels or future ground truth.

## Expected File Formats

Forecast:

```text
series_id,forecast_timestamp_utc,horizon_steps_15m,energy_wh_pred,cfp_g_pred
```

A.1 detection:

```text
series_id,bucket_15m,valid_signal_score,valid_signal_pred
```

A.2 peaks:

```text
series_id,bucket_15m,peak_score,peak_pred
```

Run the validators before evaluating:

```bash
./.venv/bin/task-a validate-submission --forecasts outputs/forecast_submission.csv
./.venv/bin/task-a validate-detection --detections outputs/detection_submission.csv
./.venv/bin/task-a validate-peaks --peaks outputs/peak_submission.csv
```

## Notes

The bundled split is for local development only:

- Train: timestamps at or before `2026-02-18T14:00:00+00:00`.
- Local test: timestamps after `2026-02-18T14:00:00+00:00`.

Official scoring may use hidden data with the same schemas. Treat the local
score as a smoke test and debugging signal, not as the final leaderboard score.
