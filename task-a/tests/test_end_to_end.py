from datetime import datetime, timedelta, timezone

from task_a.dataio import split_temporal
from task_a.evaluation import compose_task_a_score, score_detection, score_forecasts, score_peaks
from task_a.models.baseline import BaselineForecaster
from task_a.schemas import SeriesRow
from task_a.submission import validate_forecast_csv, write_forecasts


def test_end_to_end_baseline_evaluation(tmp_path):
    start = datetime(2026, 2, 17, 14, 0, tzinfo=timezone.utc)
    rows = [
        SeriesRow("s1", start + timedelta(minutes=15 * step), 1.0, 10.0 + step, 2.0 + step)
        for step in range(0, 108)
    ]
    train, test = split_temporal(rows, "2026-02-18T14:00:00+00:00")
    model = BaselineForecaster.fit(train)

    origin = datetime(2026, 2, 18, 14, 15, tzinfo=timezone.utc)
    predictions = model.predict(["s1"], origin, [4, 96])
    submission = tmp_path / "forecasts.csv"
    write_forecasts(predictions, submission)

    assert validate_forecast_csv(submission) == 2
    parts = {}
    parts.update(score_forecasts(test, predictions))
    parts.update(score_detection(test))
    parts.update(score_peaks(train, test))
    assert "ScoreTA" not in parts
    assert compose_task_a_score(parts) >= 0.0
