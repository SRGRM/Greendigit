from datetime import datetime, timezone

from task_a.dataio import split_temporal
from task_a.models.baseline import BaselineForecaster
from task_a.schemas import SeriesRow


def _row(series_id, ts, energy, cfp):
    return SeriesRow(series_id, datetime.fromisoformat(ts).replace(tzinfo=timezone.utc), 1.0, energy, cfp)


def test_split_temporal_cutoff_is_in_train():
    rows = [
        _row("s1", "2026-02-18T14:00:00", 1.0, 1.0),
        _row("s1", "2026-02-18T14:15:00", 2.0, 2.0),
    ]
    train, test = split_temporal(rows, "2026-02-18T14:00:00+00:00")
    assert len(train) == 1
    assert len(test) == 1


def test_baseline_predicts_previous_day_when_available():
    rows = [
        _row("s1", "2026-02-17T15:00:00", 10.0, 2.0),
        _row("s1", "2026-02-18T13:00:00", 20.0, 4.0),
    ]
    model = BaselineForecaster.fit(rows)
    pred = model.predict(["s1"], datetime(2026, 2, 18, 14, 0, tzinfo=timezone.utc), [4])[0]
    assert pred.forecast_timestamp_utc.hour == 15
    assert pred.energy_wh_pred == 10.0
