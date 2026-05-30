from __future__ import annotations

from math import erf, sqrt
from typing import Optional

import numpy as np
import requests
from loguru import logger

from ..models import WeatherForecast

NOAA_API_URL = "https://api.weather.gov"
AVIATION_WEATHER_URL = "https://aviationweather.gov/api/data"


class NOAAClient:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WeatherPredictionBot/1.0 (contact@example.com)",
            "Accept": "application/geo+json",
        })

    def get_metar(self, station_id: str, hours: int = 26) -> Optional[dict]:
        """
        Fetch recent METAR observations for a station.
        hours=26 covers the full previous calendar day for resolution validation.
        Returns the most recent observation or None.
        """
        try:
            url = f"{AVIATION_WEATHER_URL}/metar"
            params = {"ids": station_id.upper(), "format": "json", "hours": hours}
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
            return None
        except Exception as exc:
            logger.warning(f"METAR fetch failed for {station_id}: {exc}")
            return None

    def get_metar_history(self, station_id: str, hours: int = 26) -> list[dict]:
        """Return all METAR observations for a station over the last N hours."""
        try:
            url = f"{AVIATION_WEATHER_URL}/metar"
            params = {"ids": station_id.upper(), "format": "json", "hours": hours}
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning(f"METAR history failed for {station_id}: {exc}")
            return []

    def extract_daily_max_min(
        self, station_id: str, target_date: str
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Pull METAR history and compute daily max/min temperature in Celsius
        for the target_date (YYYY-MM-DD).
        Returns (max_c, min_c) or (None, None) if no data.
        """
        observations = self.get_metar_history(station_id, hours=30)
        temps = []
        for obs in observations:
            obs_time = obs.get("obsTime") or obs.get("reportTime") or ""
            if not obs_time.startswith(target_date):
                continue
            temp = obs.get("temp")
            if temp is not None:
                try:
                    temps.append(float(temp))
                except (TypeError, ValueError):
                    pass

        if not temps:
            return None, None
        return max(temps), min(temps)

    def validate_forecast_vs_station(
        self,
        station_id: str,
        target_date: str,
        model_max: Optional[float],
        model_min: Optional[float],
    ) -> dict:
        """
        Compare model forecast to METAR actual for the target date.
        Returns a dict with bias information for calibration.
        Only works for past dates (METAR is historical).
        """
        actual_max, actual_min = self.extract_daily_max_min(station_id, target_date)
        result = {
            "station": station_id,
            "date": target_date,
            "actual_max": actual_max,
            "actual_min": actual_min,
            "model_max": model_max,
            "model_min": model_min,
            "max_bias": None,
            "min_bias": None,
        }
        if actual_max is not None and model_max is not None:
            result["max_bias"] = round(model_max - actual_max, 2)
        if actual_min is not None and model_min is not None:
            result["min_bias"] = round(model_min - actual_min, 2)
        return result

    def get_nws_daily_forecast(
        self,
        lat: float,
        lon: float,
        target_date: str,
        location: str = "unknown",
        measurement: str = "max",
    ) -> Optional[WeatherForecast]:
        """
        Fetch NWS MOS-corrected hourly point forecast and return a WeatherForecast
        for the target_date (YYYY-MM-DD, in LOCAL city time).

        NWS startTime is already in local timezone (e.g. 2026-05-31T14:00:00-07:00)
        so filtering by date prefix is safe without UTC conversion.

        US cities only — returns None gracefully for non-US locations (Canada etc).
        Typical NWS 1-3 day forecast error: ~1.5°C (1-sigma), used as ensemble spread.
        """
        try:
            # Step 1: resolve NWS gridpoint for this lat/lon
            points_url = f"{NOAA_API_URL}/points/{lat:.4f},{lon:.4f}"
            resp = self.session.get(points_url, timeout=self.timeout)
            if resp.status_code == 404:
                logger.debug(f"NWS: no gridpoint for ({lat},{lon}) — non-US location")
                return None
            resp.raise_for_status()
            forecast_hourly_url = resp.json().get("properties", {}).get("forecastHourly")
            if not forecast_hourly_url:
                return None

            # Step 2: fetch hourly forecast
            resp2 = self.session.get(forecast_hourly_url, timeout=self.timeout)
            resp2.raise_for_status()
            periods = resp2.json().get("properties", {}).get("periods", [])

            # Step 3: collect hourly °F values for the target LOCAL date
            temps_f = []
            for p in periods:
                if p.get("startTime", "")[:10] != target_date:
                    continue
                val = p.get("temperature")
                unit = p.get("temperatureUnit", "F")
                if val is None:
                    continue
                try:
                    f = float(val) if unit == "F" else float(val) * 9/5 + 32
                    temps_f.append(f)
                except (TypeError, ValueError):
                    pass

            if not temps_f:
                logger.debug(f"NWS: no hourly data for {location} on {target_date}")
                return None

            # Step 4: compute daily stat in Celsius
            point_f = max(temps_f) if measurement == "max" else min(temps_f)
            point_c = (point_f - 32) * 5 / 9

            # Step 5: generate synthetic ensemble around NWS point (sigma = 1.5°C)
            # This represents typical NWS 1-3 day forecast error so the ensemble
            # probability calculation properly reflects station-level uncertainty.
            sigma = 1.5
            rng = np.random.default_rng(seed=int(abs(lat * lon * 1000)) % (2**31))
            members = rng.normal(point_c, sigma, 100).tolist()

            logger.info(
                f"NWS point forecast: {location} {target_date} "
                f"{measurement}={point_f:.1f}°F ({point_c:.2f}°C)"
            )
            return WeatherForecast(
                location=location,
                lat=lat,
                lon=lon,
                target_date=target_date,
                model_name="nws_point",
                ensemble_members=members,
                mean=float(point_c),
                std=float(sigma),
                spread=float(sigma * 2.56),
                p10=float(point_c - 1.28 * sigma),
                p25=float(point_c - 0.67 * sigma),
                p50=float(point_c),
                p75=float(point_c + 0.67 * sigma),
                p90=float(point_c + 1.28 * sigma),
            )

        except Exception as exc:
            logger.warning(f"NWS daily forecast failed for ({lat},{lon}): {exc}")
            return None
