"""
Simulator: generates synthetic audio frames for testing without a real microphone.
Produces synthetic flyovers on a bell-curve envelope every ~90 seconds.
Optionally plays the generated sound through speakers (audio=True).

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

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE       = 16000
NOISE_FLOOR_DB    = 42.0
NOISE_FLOOR_JITTER = 3.0
SOURCE_LEVEL_DB   = 140.0
FRAME_SEC         = 2.0
FRAME_SAMPLES     = int(SAMPLE_RATE * FRAME_SEC)


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
    """Bell-curve flyover every 90s. Returns (dist_km, alt_ft, envelope 0..1) or None."""
    period      = 90.0
    flyover_dur = 30.0
    phase = (t % period) / period
    if phase < (flyover_dur / period):
        x        = phase / (flyover_dur / period)
        envelope = math.sin(x * math.pi)
        return (round(0.2 + (1 - envelope) * 3.0, 2),
                int(500 + (1 - envelope) * 2000),
                envelope)
    return None


def _generate_audio(db_level: float, flyover_envelope: float = 0.0) -> np.ndarray:
    """
    Generate 2 seconds of synthetic audio matching the given dB level.
    During flyover: adds jet engine harmonics (low rumble + turbine whine).
    Returns float32 array normalised to -1..1.
    """
    t = np.linspace(0, FRAME_SEC, FRAME_SAMPLES, endpoint=False)

    # Background: white noise
    rms_target = 10 ** ((db_level - 94.0) / 20.0)  # dBFS from dB SPL
    noise = np.random.normal(0, 1, FRAME_SAMPLES)
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-9) * rms_target

    if flyover_envelope > 0.01:
        # Jet engine: fundamental ~130 Hz + harmonics + turbine ~2200 Hz
        engine = (
            0.6 * np.sin(2 * np.pi * 130  * t) +
            0.3 * np.sin(2 * np.pi * 260  * t) +
            0.2 * np.sin(2 * np.pi * 390  * t) +
            0.15 * np.sin(2 * np.pi * 520 * t) +
            0.1 * np.sin(2 * np.pi * 2200 * t) +
            0.05 * np.sin(2 * np.pi * 4400 * t)
        )
        # Doppler: slight frequency modulation as aircraft passes
        engine *= flyover_envelope * rms_target * 2.0
        noise = noise * (1 - flyover_envelope * 0.4) + engine

    # Normalize to -1..1, leave headroom
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak * min(0.9, rms_target * 80)

    return noise.astype(np.float32)


def _play_audio(audio: np.ndarray):
    """Play audio chunk through default output device (non-blocking)."""
    try:
        import sounddevice as sd
        sd.play(audio, samplerate=SAMPLE_RATE, blocking=False)
    except Exception as e:
        logger.debug("Audio playback error: %s", e)


async def run_simulator(frame_callback, audio: bool = False):
    logger.info("Simulator started (frame=%.1fs, flyover every 90s, audio=%s)",
                FRAME_SEC, audio)
    t0 = asyncio.get_event_loop().time()
    loop = asyncio.get_event_loop()

    while True:
        ts      = datetime.now(timezone.utc).isoformat()
        elapsed = asyncio.get_event_loop().time() - t0
        flyover = _flyover_distance(elapsed)

        if flyover:
            dist_km, alt_ft, envelope = flyover
            db_level = _distance_to_db(dist_km, alt_ft)
        else:
            envelope = 0.0
            db_level = round(NOISE_FLOOR_DB + random.gauss(0, NOISE_FLOOR_JITTER), 1)

        confidence = _db_to_confidence(db_level)

        if audio:
            audio_chunk = _generate_audio(db_level, envelope)
            await loop.run_in_executor(None, _play_audio, audio_chunk)

        frame = {"ts": ts, "db_level": db_level, "confidence": confidence}
        logger.debug("Frame: db=%.1f conf=%.3f envelope=%.2f", db_level, confidence, envelope)

        await frame_callback(frame)
        await asyncio.sleep(FRAME_SEC)
