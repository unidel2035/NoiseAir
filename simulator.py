"""
Simulator: generates synthetic audio frames for testing without a real microphone.
Produces synthetic flyovers on a bell-curve envelope every ~90 seconds.

Frame format (same as detector.py output):
    {
      "ts":         ISO timestamp string,
      "db_level":   float,
      "confidence": float 0..1
    }
"""

import asyncio
import logging
import math
import random
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

NOISE_FLOOR_DB    = 42.0
NOISE_FLOOR_JITTER = 3.0
SOURCE_LEVEL_DB   = 140.0
FRAME_SEC         = 2.0


def _distance_to_db(distance_km: float, altitude_ft: float) -> float:
    alt_m   = (altitude_ft or 300) * 0.3048
    horiz_m = distance_km * 1000.0
    dist_3d = max(math.sqrt(horiz_m ** 2 + alt_m ** 2), 50.0)
    return round(SOURCE_LEVEL_DB - 20 * math.log10(dist_3d) + random.uniform(-2.0, 2.0), 1)


def _db_to_confidence(db: float) -> float:
    delta = db - NOISE_FLOOR_DB
    if delta < 5:
        return max(0.0, random.uniform(-0.1, 0.15))
    conf = 1 / (1 + math.exp(-0.3 * (delta - 10))) + random.uniform(-0.05, 0.05)
    return round(max(0.0, min(1.0, conf)), 3)


def _flyover_distance(t: float):
    """Bell-curve flyover every 90s. Returns (dist_km, alt_ft) or None."""
    period      = 90.0
    flyover_dur = 30.0
    phase = (t % period) / period
    if phase < (flyover_dur / period):
        x        = phase / (flyover_dur / period)
        envelope = math.sin(x * math.pi)
        return (round(0.2 + (1 - envelope) * 3.0, 2),
                int(500 + (1 - envelope) * 2000))
    return None


async def run_simulator(frame_callback):
    logger.info("Simulator started (frame=%.1fs, flyover every 90s)", FRAME_SEC)
    t0 = asyncio.get_event_loop().time()

    while True:
        ts      = datetime.now(timezone.utc).isoformat()
        elapsed = asyncio.get_event_loop().time() - t0
        flyover = _flyover_distance(elapsed)

        if flyover:
            dist_km, alt_ft = flyover
            db_level = _distance_to_db(dist_km, alt_ft)
        else:
            db_level = round(NOISE_FLOOR_DB + random.gauss(0, NOISE_FLOOR_JITTER), 1)

        confidence = _db_to_confidence(db_level)

        frame = {"ts": ts, "db_level": db_level, "confidence": confidence}
        logger.debug("Frame: db=%.1f conf=%.3f", db_level, confidence)

        await frame_callback(frame)
        await asyncio.sleep(FRAME_SEC)
