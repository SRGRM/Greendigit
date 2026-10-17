from pathlib import Path

import pytest

from task_a.submission import validate_detection_csv, validate_forecast_csv, validate_peak_csv


def test_forecast_validation_rejects_empty_submission(tmp_path: Path):
    path = tmp_path / "forecasts.csv"
    path.write_text("series_id,forecast_timestamp_utc,horizon_steps_15m,energy_wh_pred,cfp_g_pred\n")

    with pytest.raises(ValueError, match="no forecast rows"):
        validate_forecast_csv(path)


def test_forecast_validation_rejects_unsupported_horizon(tmp_path: Path):
    path = tmp_path / "forecasts.csv"
    path.write_text(
        "series_id,forecast_timestamp_utc,horizon_steps_15m,energy_wh_pred,cfp_g_pred\n"
        "s1,2026-02-18T15:00:00+00:00,8,1.0,1.0\n"
    )

    with pytest.raises(ValueError, match="unsupported horizon"):
        validate_forecast_csv(path)


def test_detection_and_peak_validation(tmp_path: Path):
    detections = tmp_path / "detections.csv"
    detections.write_text(
        "series_id,bucket_15m,valid_signal_score,valid_signal_pred\n"
        "s1,2026-02-18T14:15:00+00:00,0.8,1\n"
    )
    peaks = tmp_path / "peaks.csv"
    peaks.write_text(
        "series_id,bucket_15m,peak_score,peak_pred\n"
        "s1,2026-02-18T14:15:00+00:00,0.2,0\n"
    )

    assert validate_detection_csv(detections) == 1
    assert validate_peak_csv(peaks) == 1
