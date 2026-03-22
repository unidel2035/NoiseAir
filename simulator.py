"""
Simulator: generates synthetic audio frames based on real ADS-B data.
Replaces the microphone during development (no real mic connected).

Frame format (same as detector.py output):
    {
      "ts":         ISO timestamp string,
      "db_level":   float,
      "confidence": float 0..1,
      "adsb":       dict or None
    }
"""

import asyncio
import logging
import math
import random
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Background noise level (dB)
NOISE_FLOOR_DB = 42.0
NOISE_FLOOR_JITTER = 3.0

# Source level at 1 m (typical turboprop/jet)
SOURCE_LEVEL_DB = 140.0

# Frame interval in seconds
FRAME_SEC = 2.0


def _distance_to_db(distance_km: float, altitude_ft: float) -> float:
    """Estimate noise level from aircraft distance."""
    alt_m = (altitude_ft or 300) * 0.3048
    horiz_m = distance_km * 1000.0
    dist_3d = math.sqrt(horiz_m ** 2 + alt_m ** 2)
    dist_3d = max(dist_3d, 50.0)  # avoid log(0)

    db = SOURCE_LEVEL_DB - 20 * math.log10(dist_3d)
    # Add random jitter ±2 dB
    db += random.uniform(-2.0, 2.0)
    return round(db, 1)


def _db_to_confidence(db: float) -> float:
    """Map dB level above noise floor to aircraft confidence."""
    delta = db - NOISE_FLOOR_DB
    if delta < 5:
        return max(0.0, random.uniform(-0.1, 0.15))
    # Sigmoid-like mapping: 5 dB above floor → ~0.5 confidence
    conf = 1 / (1 + math.exp(-0.3 * (delta - 10)))
    conf += random.uniform(-0.05, 0.05)
    return round(max(0.0, min(1.0, conf)), 3)


def _make_demo_flyover(t: float) -> dict:
    """
    Synthetic flyover: aircraft passes overhead every ~90 seconds.
    Returns fake adsb dict with distance that follows a bell curve.
    """
    period = 90.0           # seconds between flyovers
    flyover_dur = 30.0      # seconds the aircraft is close
    phase = (t % period) / period

    if phase < (flyover_dur / period):
        # Bell curve: 0 → closest at midpoint → 0
        x = phase / (flyover_dur / period)  # 0..1
        envelope = math.sin(x * math.pi)   # 0→1→0
        dist_km = 0.2 + (1 - envelope) * 3.0   # 0.2..3.2 km
        alt_ft  = int(500 + (1 - envelope) * 2000)
        return {
            "icao":        "TEST01",
            "callsign":    "DEMO01",
            "altitude_ft": alt_ft,
            "distance_km": round(dist_km, 2),
            "lat": 56.84,
            "lon": 53.25,
        }
    return None


async def run_simulator(frame_callback, adsb_client=None, demo: bool = False):
    """
    Continuously produces frames and calls frame_callback(frame).
    If adsb_client is provided, uses real ADS-B data.
    If demo=True (or ADS-B unavailable), generates synthetic flyovers for testing.
    """
    logger.info("Simulator started (frame interval=%.1fs, demo=%s)", FRAME_SEC, demo)
    t0 = asyncio.get_event_loop().time()

    while True:
        ts = datetime.now(timezone.utc).isoformat()
        adsb = None

        if adsb_client:
            adsb = await adsb_client.get_nearest()

        # If no real ADS-B, fall back to demo flyovers
        if adsb is None:
            elapsed = asyncio.get_event_loop().time() - t0
            adsb = _make_demo_flyover(elapsed)

        if adsb and adsb.get("distance_km", 999) <= 15.0:
            db_level = _distance_to_db(adsb["distance_km"], adsb.get("altitude_ft", 1000))
        else:
            db_level = NOISE_FLOOR_DB + random.gauss(0, NOISE_FLOOR_JITTER)
            db_level = round(db_level, 1)

        confidence = _db_to_confidence(db_level)

        frame = {
            "ts":         ts,
            "db_level":   db_level,
            "confidence": confidence,
            "adsb":       adsb,
        }

        logger.debug("Frame: db=%.1f conf=%.3f adsb=%s",
                     db_level, confidence,
                     adsb.get("icao") if adsb else None)

        await frame_callback(frame)
        await asyncio.sleep(FRAME_SEC)
