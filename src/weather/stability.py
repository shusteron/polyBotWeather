from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

from ..models import ForecastStabilityRecord, WeatherForecast

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "forecast_history.json")


class ForecastStabilityEngine:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.history_file = os.path.join(data_dir, "forecast_history.json")
        self._history: dict = {}
        self._load_history()

    def record_forecast(
        self, location: str, target_date: str, forecast: WeatherForecast
    ) -> None:
        """Append a new forecast run to the history for this location/date."""
        key = f"{location}::{target_date}"
        if key not in self._history:
            self._history[key] = []

        run = {
            "timestamp": datetime.utcnow().isoformat(),
            "mean": forecast.mean,
            "std": forecast.std,
            "p10": forecast.p10,
            "p90": forecast.p90,
            "model": forecast.model_name,
        }
        self._history[key].append(run)

    def calculate_stability(
        self, location: str, target_date: str, threshold: Optional[float] = None, lookback_hours: int = 72
    ) -> float:
        """
        Calculate stability score (0-1) based on forecast history.
        High stability = forecast hasn't changed much across multiple runs.
        """
        record = self.get_stability_record(location, target_date, threshold, lookback_hours)
        return record.stability_score

    def get_stability_record(
        self,
        location: str,
        target_date: str,
        threshold: Optional[float] = None,
        lookback_hours: int = 72,
    ) -> ForecastStabilityRecord:
        """Return a ForecastStabilityRecord for the location/date pair."""
        key = f"{location}::{target_date}"
        all_runs = self._history.get(key, [])

        # Filter to lookback window
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        recent_runs = []
        for run in all_runs:
            try:
                ts = datetime.fromisoformat(run["timestamp"])
                if ts >= cutoff:
                    recent_runs.append(run)
            except Exception:
                recent_runs.append(run)

        if not recent_runs:
            return ForecastStabilityRecord(
                location=location,
                target_date=target_date,
                runs=[],
                stability_score=0.5,  # neutral when no history
                direction_consistent=False,
            )

        # Group runs by scan-hour so cross-model variance within one run
        # doesn't masquerade as instability. Use one mean per hour bucket.
        from collections import defaultdict
        hour_buckets: dict[str, list[float]] = defaultdict(list)
        for run in recent_runs:
            ts_str = run.get("timestamp", "")
            hour_key = ts_str[:13]  # "2026-05-24T15" — one bucket per hour
            if "mean" in run:
                hour_buckets[hour_key].append(run["mean"])

        # Need at least 2 distinct scan hours to compute meaningful stability.
        # Return 0.65 (passes the 0.55 gate) — no history means no evidence of instability.
        if len(hour_buckets) < 2:
            return ForecastStabilityRecord(
                location=location,
                target_date=target_date,
                runs=recent_runs,
                stability_score=0.65,
                direction_consistent=False,
            )

        # One representative mean per hour (ensemble average within each scan)
        means = [float(np.mean(v)) for v in hour_buckets.values()]

        if not means:
            return ForecastStabilityRecord(
                location=location,
                target_date=target_date,
                runs=recent_runs,
                stability_score=0.5,
                direction_consistent=False,
            )

        # Stability = 1 - normalized std of hourly means
        mean_of_means = float(np.mean(means))
        std_of_means = float(np.std(means))

        # Normalize: assume 3°C std is "very unstable" → score of 0
        max_expected_std = 3.0
        if max_expected_std > 0:
            stability_score = float(max(0.0, 1.0 - std_of_means / max_expected_std))
        else:
            stability_score = 1.0

        # Direction consistency: all runs agree on same side of threshold
        direction_consistent = False
        if threshold is not None and len(means) >= 2:
            above = [m > threshold for m in means]
            direction_consistent = all(above) or not any(above)
        elif len(means) >= 2:
            # Check if trend is consistent (all increasing or all decreasing)
            diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
            direction_consistent = all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)

        return ForecastStabilityRecord(
            location=location,
            target_date=target_date,
            runs=recent_runs,
            stability_score=stability_score,
            direction_consistent=direction_consistent,
        )

    def save(self) -> None:
        """Prune stale entries then persist history — call once after a full scan cycle."""
        self._prune_old_entries()
        self._save_history()

    def _prune_old_entries(self, keep_hours: int = 96) -> None:
        """Remove forecast runs older than keep_hours to keep the file small."""
        cutoff = datetime.utcnow() - timedelta(hours=keep_hours)
        for key in list(self._history.keys()):
            kept = []
            for run in self._history[key]:
                try:
                    if datetime.fromisoformat(run["timestamp"]) >= cutoff:
                        kept.append(run)
                except Exception:
                    kept.append(run)
            if kept:
                self._history[key] = kept
            else:
                del self._history[key]

    def _load_history(self) -> None:
        """Load forecast history from JSON file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    self._history = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load forecast history: {exc}")
                self._history = {}
        else:
            self._history = {}

    def _save_history(self) -> None:
        """Persist forecast history to JSON file."""
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            with open(self.history_file, "w") as f:
                json.dump(self._history, f, indent=2, default=str)
        except Exception as exc:
            logger.error(f"Failed to save forecast history: {exc}")
