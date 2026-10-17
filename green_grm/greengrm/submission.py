"""Task A submission files and official local scoring.

The three files follow the schemas of the challenge
(``task-a/src/task_a/schemas.py``):

    forecast_submission.csv   series_id, forecast_timestamp_utc,
                              horizon_steps_15m, energy_wh_pred, cfp_g_pred
    detection_submission.csv  series_id, bucket_15m, valid_signal_score,
                              valid_signal_pred
    peak_submission.csv       series_id, bucket_15m, peak_score, peak_pred

They are checked with the official validators and scored with the official
evaluator of the starter kit (Eq. 2 of the paper; lower is better). The
ground truth handed to the official scorer is the canonical series of the
modelled sites (``write_scoring_series``), so the score is also defined when
the input was the event-level file or a subset of the sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .utils import ensure_challenge_packages, get_logger

FORECAST_FILE = "forecast_submission.csv"
DETECTION_FILE = "detection_submission.csv"
PEAK_FILE = "peak_submission.csv"


def _iso(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def write_task_a_submission(forecasts: pd.DataFrame, detections: pd.DataFrame,
                            peaks: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fc = forecasts.sort_values(["forecast_timestamp_utc", "series_id", "horizon_steps_15m"])
    fc = pd.DataFrame({
        "series_id": fc["series_id"].to_numpy(),
        "forecast_timestamp_utc": _iso(fc["forecast_timestamp_utc"]).to_numpy(),
        "horizon_steps_15m": fc["horizon_steps_15m"].astype(int).to_numpy(),
        "energy_wh_pred": fc["energy_wh_pred"].to_numpy(),
        "cfp_g_pred": fc["cfp_g_pred"].to_numpy(),
    })
    det = detections.sort_values(["bucket_15m", "series_id"])
    det = pd.DataFrame({
        "series_id": det["series_id"].to_numpy(),
        "bucket_15m": _iso(det["bucket_15m"]).to_numpy(),
        "valid_signal_score": det["valid_signal_score"].to_numpy(),
        "valid_signal_pred": det["valid_signal_pred"].astype(int).to_numpy(),
    })
    pk = peaks.sort_values(["bucket_15m", "series_id"])
    pk = pd.DataFrame({
        "series_id": pk["series_id"].to_numpy(),
        "bucket_15m": _iso(pk["bucket_15m"]).to_numpy(),
        "peak_score": pk["peak_score"].to_numpy(),
        "peak_pred": pk["peak_pred"].astype(int).to_numpy(),
    })

    paths = {"forecasts": out_dir / FORECAST_FILE, "detections": out_dir / DETECTION_FILE,
             "peaks": out_dir / PEAK_FILE}
    fc.to_csv(paths["forecasts"], index=False)
    det.to_csv(paths["detections"], index=False)
    pk.to_csv(paths["peaks"], index=False)
    get_logger().info("[submission] wrote %s forecast, %s detection and %s peak rows to %s",
                      f"{len(fc):,}", f"{len(det):,}", f"{len(pk):,}", out_dir)
    return paths


def write_scoring_series(buckets: pd.DataFrame, path: Path) -> Path:
    """Canonical Task A series (series_id, bucket_15m, records, energy_wh,
    cfp_g) of the loaded sites, in the format of ``task_a_series.csv``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = buckets.sort_values(["series_id", "bucket_15m"])
    pd.DataFrame({
        "series_id": frame["series_id"].to_numpy(),
        "bucket_15m": _iso(frame["bucket_15m"]).to_numpy(),
        "records": frame["records"].fillna(0.0).to_numpy(),
        "energy_wh": frame["energy_wh"].to_numpy(),
        "cfp_g": frame["cfp_g"].to_numpy(),
    }).to_csv(path, index=False)
    return path


def validate_with_official_kit(paths: Dict[str, Path]) -> Dict[str, int]:
    """Run the starter-kit validators; they raise ValueError on any problem."""
    ensure_challenge_packages()
    from task_a.submission import validate_detection_csv, validate_forecast_csv, validate_peak_csv

    counts = {
        "forecast_rows": validate_forecast_csv(paths["forecasts"]),
        "detection_rows": validate_detection_csv(paths["detections"]),
        "peak_rows": validate_peak_csv(paths["peaks"]),
    }
    get_logger().info("[submission] official validators passed: %s", counts)
    return counts


def score_with_official_kit(data_path: Path, cutoff: str, paths: Dict[str, Path],
                            eps: float = 1e-9, peak_quantile: float = 0.95) -> Dict[str, float]:
    """Local ScoreTA exactly as ``task-a evaluate`` computes it (keep eps and
    peak_quantile at the official values to obtain the official score)."""
    ensure_challenge_packages()
    from task_a.dataio import load_series_csv, split_temporal
    from task_a.evaluation import (compose_task_a_score, load_detections, load_forecasts,
                                   load_peaks, score_detection, score_forecasts, score_peaks)

    rows = load_series_csv(data_path)
    train, test = split_temporal(rows, cutoff)
    parts: Dict[str, float] = {}
    parts.update(score_forecasts(test, load_forecasts(paths["forecasts"]), eps, require_complete=True))
    parts.update(score_detection(test, load_detections(paths["detections"])))
    parts.update(score_peaks(train, test, peak_quantile, load_peaks(paths["peaks"])))
    parts["ScoreTA"] = compose_task_a_score(parts)
    get_logger().info("[submission] official local ScoreTA = %.5f (S_A_4=%.4f, S_A_96=%.4f, "
                      "S_A1=%.4f, S_A2=%.4f)", parts["ScoreTA"], parts.get("S_A_4", float("nan")),
                      parts.get("S_A_96", float("nan")), parts["S_A1"], parts["S_A2"])
    return parts
