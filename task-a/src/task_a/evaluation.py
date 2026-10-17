from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

from .labels import peak_label, peak_threshold, valid_signal_label
from .schemas import DetectionRow, ForecastRow, PeakRow, SeriesRow, parse_timestamp, require_columns


def smape(y_true: list[float], y_pred: list[float], eps: float = 1e-9) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0
    total = 0.0
    for actual, pred in zip(y_true, y_pred):
        total += abs(pred - actual) / max(eps, (abs(actual) + abs(pred)) / 2.0)
    return total / len(y_true)


def f1_score(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    denom = (2 * tp) + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def auroc(y_true: list[int], scores: list[float]) -> float:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    pairs = sorted(zip(scores, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def load_forecasts(path: str | Path) -> list[ForecastRow]:
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        require_columns(
            reader.fieldnames,
            ["series_id", "forecast_timestamp_utc", "horizon_steps_15m", "energy_wh_pred", "cfp_g_pred"],
            str(path),
        )
        return [
            ForecastRow(
                row["series_id"],
                parse_timestamp(row["forecast_timestamp_utc"]),
                int(row["horizon_steps_15m"]),
                float(row["energy_wh_pred"]),
                float(row["cfp_g_pred"]),
            )
            for row in reader
        ]


def load_detections(path: str | Path) -> list[DetectionRow]:
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        require_columns(
            reader.fieldnames,
            ["series_id", "bucket_15m", "valid_signal_score", "valid_signal_pred"],
            str(path),
        )
        return [
            DetectionRow(
                row["series_id"],
                parse_timestamp(row["bucket_15m"]),
                float(row["valid_signal_score"]),
                int(row["valid_signal_pred"]),
            )
            for row in reader
        ]


def load_peaks(path: str | Path) -> list[PeakRow]:
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        require_columns(
            reader.fieldnames,
            ["series_id", "bucket_15m", "peak_score", "peak_pred"],
            str(path),
        )
        return [
            PeakRow(
                row["series_id"],
                parse_timestamp(row["bucket_15m"]),
                float(row["peak_score"]),
                int(row["peak_pred"]),
            )
            for row in reader
        ]


def expected_forecast_keys(
    truth: list[SeriesRow],
    horizons: tuple[int, ...] = (4, 96),
) -> set[tuple[str, object, int]]:
    truth_index = {(row.series_id, row.bucket_15m) for row in truth}
    origins = sorted({row.bucket_15m for row in truth})
    series_ids = sorted({row.series_id for row in truth})
    keys = set()
    for series_id in series_ids:
        for origin in origins:
            for horizon in horizons:
                forecast_ts = origin + timedelta(minutes=15 * horizon)
                if (series_id, forecast_ts) in truth_index:
                    keys.add((series_id, forecast_ts, horizon))
    return keys


def score_forecasts(
    truth: list[SeriesRow],
    predictions: list[ForecastRow],
    eps: float = 1e-9,
    require_complete: bool = False,
) -> dict:
    truth_index = {(row.series_id, row.bucket_15m): row for row in truth}
    prediction_keys = {
        (pred.series_id, pred.forecast_timestamp_utc, pred.horizon_steps_15m)
        for pred in predictions
    }
    if require_complete:
        expected = expected_forecast_keys(truth)
        missing = sorted(expected - prediction_keys)
        if missing:
            preview = ", ".join(f"{sid}@{ts.isoformat()}/h{h}" for sid, ts, h in missing[:5])
            raise ValueError(f"forecast submission is missing {len(missing)} required rows: {preview}")
    by_horizon: dict[int, dict[str, list[float]]] = {}
    for pred in predictions:
        actual = truth_index.get((pred.series_id, pred.forecast_timestamp_utc))
        if actual is None:
            continue
        bucket = by_horizon.setdefault(
            pred.horizon_steps_15m,
            {"energy_true": [], "energy_pred": [], "cfp_true": [], "cfp_pred": []},
        )
        bucket["energy_true"].append(actual.energy_wh)
        bucket["energy_pred"].append(pred.energy_wh_pred)
        bucket["cfp_true"].append(actual.cfp_g)
        bucket["cfp_pred"].append(pred.cfp_g_pred)

    scores = {}
    for horizon, values in sorted(by_horizon.items()):
        energy = smape(values["energy_true"], values["energy_pred"], eps)
        cfp = smape(values["cfp_true"], values["cfp_pred"], eps)
        scores[f"S_A_{horizon}"] = 0.5 * energy + 0.5 * cfp
        scores[f"smape_energy_{horizon}"] = energy
        scores[f"smape_cfp_{horizon}"] = cfp
        scores[f"n_{horizon}"] = len(values["energy_true"])
    return scores


def score_detection(rows: list[dict | SeriesRow], predictions: list[DetectionRow] | None = None) -> dict:
    labels_by_key = {
        (row.series_id, row.bucket_15m): valid_signal_label(row)
        for row in rows
        if isinstance(row, SeriesRow)
    }
    if predictions is None:
        labels = [valid_signal_label(row) for row in rows]
        scores = [float(label) for label in labels]
        preds = labels
    else:
        pred_by_key = {(row.series_id, row.bucket_15m): row for row in predictions}
        missing = sorted(set(labels_by_key) - set(pred_by_key))
        if missing:
            preview = ", ".join(f"{sid}@{ts.isoformat()}" for sid, ts in missing[:5])
            raise ValueError(f"detection submission is missing {len(missing)} required rows: {preview}")
        labels = [labels_by_key[key] for key in sorted(labels_by_key)]
        scores = [pred_by_key[key].valid_signal_score for key in sorted(labels_by_key)]
        preds = [pred_by_key[key].valid_signal_pred for key in sorted(labels_by_key)]
    roc = auroc(labels, scores)
    f1 = f1_score(labels, preds)
    return {"S_A1": 1.0 - 0.5 * (roc + f1), "auroc_a1": roc, "f1_a1": f1}


def score_peaks(
    train: list[SeriesRow],
    test: list[SeriesRow],
    quantile: float = 0.95,
    predictions: list[PeakRow] | None = None,
) -> dict:
    thresholds = peak_threshold(train, quantile)
    labels_by_key = {(row.series_id, row.bucket_15m): peak_label(row, thresholds) for row in test}
    if predictions is None:
        labels = [peak_label(row, thresholds) for row in test]
        scores = [max(row.energy_wh, row.cfp_g) / max(1e-9, thresholds.get(row.series_id, 1e-9)) for row in test]
        preds = [int(score >= 1.0) for score in scores]
    else:
        pred_by_key = {(row.series_id, row.bucket_15m): row for row in predictions}
        missing = sorted(set(labels_by_key) - set(pred_by_key))
        if missing:
            preview = ", ".join(f"{sid}@{ts.isoformat()}" for sid, ts in missing[:5])
            raise ValueError(f"peak submission is missing {len(missing)} required rows: {preview}")
        labels = [labels_by_key[key] for key in sorted(labels_by_key)]
        scores = [pred_by_key[key].peak_score for key in sorted(labels_by_key)]
        preds = [pred_by_key[key].peak_pred for key in sorted(labels_by_key)]
    roc = auroc(labels, scores)
    f1 = f1_score(labels, preds)
    return {"S_A2": 1.0 - 0.5 * (roc + f1), "auroc_a2": roc, "f1_a2": f1}


def compose_task_a_score(parts: dict) -> float:
    horizon_scores = [
        value for key, value in parts.items()
        if key.startswith("S_A_") and key not in {"S_A1", "S_A2"}
    ]
    avg_forecast = sum(horizon_scores) / len(horizon_scores) if horizon_scores else 0.0
    return 0.7 * avg_forecast + 0.15 * parts.get("S_A1", 0.0) + 0.15 * parts.get("S_A2", 0.0)
