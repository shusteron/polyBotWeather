from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from loguru import logger

from ..models import CalibrationRecord

CALIBRATION_FILE = os.path.join("data", "calibration.json")

# Minimum resolved records needed before we trust a reliability score.
# Below this we return a neutral 50.0 — "we don't know yet."
MIN_SAMPLES_FOR_RELIABILITY = 20


class CalibrationEngine:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.calibration_file = os.path.join(data_dir, "calibration.json")
        self._records: list[dict] = []
        self._load_records()

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def record_trade(self, record: CalibrationRecord) -> None:
        self._records.append(record.model_dump())
        self._save_records()

    def update_outcome(self, market_id: str, outcome: bool, pnl: float) -> None:
        for r in self._records:
            if r.get("market_id") == market_id and r.get("outcome") is None:
                r["outcome"] = outcome
                r["pnl"] = pnl
                break
        self._save_records()

    # ------------------------------------------------------------------ #
    #  Overall accuracy                                                    #
    # ------------------------------------------------------------------ #

    def get_calibration_score(self) -> float:
        """Overall Brier-based score 0-100.  50 = neutral / no data."""
        records = self._get_resolved_records()
        if not records:
            return 50.0
        brier = self._brier(records)
        return float(max(0.0, min(100.0, 100.0 * (1.0 - brier / 0.25))))

    def get_brier_score(self, records: list[dict]) -> float:
        resolved = [r for r in records if r.get("outcome") is not None]
        if not resolved:
            return 0.25
        errors = [(float(r.get("predicted_probability", 0.5)) - float(r["outcome"])) ** 2
                  for r in resolved]
        return float(np.mean(errors))

    # ------------------------------------------------------------------ #
    #  Zone reliability — how accurate is the model in a probability band #
    # ------------------------------------------------------------------ #

    def get_zone_reliability(self, model_prob: float) -> float:
        """
        Returns 0-100 score for how well the model is calibrated in the
        probability band that contains model_prob.

        Bands (width 0.10):  0-10%, 10-20%, …, 90-100%

        If fewer than MIN_SAMPLES_FOR_RELIABILITY resolved records exist in
        the band we return 50.0 (neutral — don't penalise or reward yet).
        """
        resolved = self._get_resolved_records()
        band_low  = (int(model_prob * 10)) / 10.0        # e.g. 0.86 → 0.80
        band_high = band_low + 0.10

        in_band = [
            r for r in resolved
            if band_low <= float(r.get("predicted_probability", 0)) < band_high
        ]

        if len(in_band) < MIN_SAMPLES_FOR_RELIABILITY:
            return 50.0  # not enough data — stay neutral

        brier = self._brier(in_band)
        return float(max(0.0, min(100.0, 100.0 * (1.0 - brier / 0.25))))

    def get_zone_sample_count(self, model_prob: float) -> int:
        """How many resolved trades exist in this model_prob zone."""
        resolved = self._get_resolved_records()
        band_low  = (int(model_prob * 10)) / 10.0
        band_high = band_low + 0.10
        return sum(
            1 for r in resolved
            if band_low <= float(r.get("predicted_probability", 0)) < band_high
        )

    # ------------------------------------------------------------------ #
    #  City reliability — how accurate is the model for a specific city   #
    # ------------------------------------------------------------------ #

    def get_city_reliability(self, city: Optional[str]) -> float:
        """
        Returns 0-100 score for how accurate the model has been for this
        specific city.  Neutral (50.0) until MIN_SAMPLES_FOR_RELIABILITY
        resolved records exist for the city.
        """
        if not city:
            return 50.0

        resolved = self._get_resolved_records()
        city_records = [
            r for r in resolved
            if str(r.get("location", "")).lower() == city.lower()
        ]

        if len(city_records) < MIN_SAMPLES_FOR_RELIABILITY:
            return 50.0

        brier = self._brier(city_records)
        return float(max(0.0, min(100.0, 100.0 * (1.0 - brier / 0.25))))

    def get_city_sample_count(self, city: Optional[str]) -> int:
        """How many resolved trades exist for this city."""
        if not city:
            return 0
        resolved = self._get_resolved_records()
        return sum(
            1 for r in resolved
            if str(r.get("location", "")).lower() == city.lower()
        )

    # ------------------------------------------------------------------ #
    #  Calibration curve (for reporting)                                  #
    # ------------------------------------------------------------------ #

    def get_calibration_curve(self) -> list[dict]:
        """
        Bin predictions into deciles and compute actual hit rates per bin.
        Returns [{bin_low, bin_high, predicted_mid, actual_rate, count}]
        """
        resolved = self._get_resolved_records()
        bins = []
        for low_pct in range(0, 100, 10):
            low  = low_pct / 100.0
            high = (low_pct + 10) / 100.0
            bucket = [
                r for r in resolved
                if low <= float(r.get("predicted_probability", 0)) < high
            ]
            actual_rate = float(np.mean([r["outcome"] for r in bucket])) if bucket else 0.0
            bins.append({
                "bin_low":       low,
                "bin_high":      high,
                "predicted_mid": (low + high) / 2.0,
                "actual_rate":   actual_rate,
                "count":         len(bucket),
            })
        return bins

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _get_resolved_records(self) -> list[dict]:
        return [r for r in self._records if r.get("outcome") is not None]

    def _brier(self, records: list[dict]) -> float:
        errors = [
            (float(r.get("predicted_probability", 0.5)) - float(r["outcome"])) ** 2
            for r in records if r.get("outcome") is not None
        ]
        return float(np.mean(errors)) if errors else 0.25

    def _load_records(self) -> None:
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, "r") as f:
                    self._records = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load calibration records: {exc}")
                self._records = []
        else:
            self._records = []

    def _save_records(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        tmp = self.calibration_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self._records, f, indent=2, default=str)
            os.replace(tmp, self.calibration_file)
        except Exception as exc:
            logger.error(f"Failed to save calibration records: {exc}")
            try:
                os.unlink(tmp)
            except Exception:
                pass
