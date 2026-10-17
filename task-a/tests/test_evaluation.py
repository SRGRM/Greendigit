from datetime import datetime, timedelta, timezone

import pytest

from task_a.evaluation import auroc, compose_task_a_score, f1_score, score_forecasts, smape
from task_a.schemas import ForecastRow, SeriesRow


def test_smape_zero_for_exact_predictions():
    assert smape([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_binary_metrics():
    assert f1_score([1, 0, 1], [1, 0, 0]) == 2 / 3
    assert auroc([0, 1], [0.1, 0.9]) == 1.0


def test_composed_score_uses_forecast_and_detection_parts():
    score = compose_task_a_score({"S_A_4": 0.2, "S_A_96": 0.4, "S_A1": 0.1, "S_A2": 0.3})
    assert round(score, 6) == 0.27


def test_complete_forecast_scoring_rejects_missing_required_keys():
    start = datetime(2026, 2, 18, 14, 15, tzinfo=timezone.utc)
    truth = [
        SeriesRow("s1", start + timedelta(minutes=15 * step), 1.0, 10.0, 2.0)
        for step in range(0, 100)
    ]
    incomplete = [ForecastRow("s1", start + timedelta(hours=1), 4, 10.0, 2.0)]

    with pytest.raises(ValueError, match="missing .* required rows"):
        score_forecasts(truth, incomplete, require_complete=True)
