from __future__ import annotations

from typing import Optional

import requests
from loguru import logger

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

    def get_point_forecast(self, lat: float, lon: float) -> Optional[dict]:
        """NWS point forecast — US cities only."""
        try:
            points_url = f"{NOAA_API_URL}/points/{lat:.4f},{lon:.4f}"
            resp = self.session.get(points_url, timeout=self.timeout)
            resp.raise_for_status()
            points_data = resp.json()
            forecast_hourly_url = points_data.get("properties", {}).get("forecastHourly")
            if not forecast_hourly_url:
                return None
            hourly_resp = self.session.get(forecast_hourly_url, timeout=self.timeout)
            hourly_resp.raise_for_status()
            return hourly_resp.json()
        except Exception as exc:
            logger.warning(f"NWS point forecast failed for ({lat},{lon}): {exc}")
            return None

    def parse_hourly_temperature(self, forecast_data: dict, target_date: str) -> list[float]:
        """Extract hourly °F values from NWS forecast for target_date."""
        periods = forecast_data.get("properties", {}).get("periods", [])
        temps = []
        for period in periods:
            if period.get("startTime", "").startswith(target_date):
                val = period.get("temperature")
                if val is not None:
                    try:
                        temps.append(float(val))
                    except (TypeError, ValueError):
                        pass
        return temps
