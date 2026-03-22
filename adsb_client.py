"""
Fetches live aircraft positions from the ADS-B receiver (readsb/tar1090 JSON API).
Returns the nearest aircraft and its distance from the fixed listener point.
"""

import asyncio
import logging
import math
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Fixed listener coordinates (Izhevsk receiver)
LISTENER_LAT = 56.83686
LISTENER_LON = 53.24976

# ADS-B receiver JSON endpoint (tar1090 default)
ADSB_URL = "http://192.168.1.134/tar1090/data/aircraft.json"

# Consider only aircraft within this radius
MAX_RADIUS_KM = 15.0


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in km between two WGS-84 points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _nearest(aircraft_list: list) -> Optional[dict]:
    best = None
    best_dist = MAX_RADIUS_KM

    for ac in aircraft_list:
        lat = ac.get("lat")
        lon = ac.get("lon")
        if lat is None or lon is None:
            continue

        dist = _haversine(LISTENER_LAT, LISTENER_LON, lat, lon)
        if dist < best_dist:
            best_dist = dist
            best = {
                "icao":        ac.get("hex", "").upper(),
                "callsign":    (ac.get("flight") or "").strip() or None,
                "altitude_ft": ac.get("alt_baro") or ac.get("alt_geom"),
                "distance_km": round(dist, 2),
                "lat":         lat,
                "lon":         lon,
            }

    return best


class AdsbClient:
    def __init__(self, url: str = ADSB_URL, timeout: float = 3.0):
        self._url = url
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Optional[dict] = None

    async def start(self):
        self._session = aiohttp.ClientSession(timeout=self._timeout)

    async def stop(self):
        if self._session:
            await self._session.close()

    async def get_nearest(self) -> Optional[dict]:
        try:
            async with self._session.get(self._url) as resp:
                data = await resp.json(content_type=None)
            aircraft = data.get("aircraft", [])
            result = _nearest(aircraft)
            self._cache = result
            return result
        except Exception as e:
            logger.debug("ADS-B fetch failed: %s", e)
            return self._cache  # return last known on error
