"""
File player: replaces simulator.py for testing with real audio files.
Playlist of aircraft sounds at different noise levels.
Uses spectral analysis (not just dB) to compute confidence.
"""

import asyncio
import logging
import os
import wave
from datetime import datetime, timezone

import numpy as np

from spectral import compute_spectral_features, AircraftScorer

logger = logging.getLogger(__name__)

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")

# Playlist: (filename, label, volume_scale)
# volume_scale: 1.0 = original, 0.3 = quiet distant aircraft
PLAYLIST = [
    ("background.wav",  "background",    1.0),   # фон
    ("jet_civil.wav",   "jet_civil",     1.0),   # гражданский реактивный
    ("background.wav",  "background",    1.0),   # фон
    ("aircraft.wav",    "propeller",     1.0),   # винтовой
    ("background.wav",  "background",    1.0),   # фон
    ("jet_military.wav","jet_military",  1.2),   # военный (громче)
    ("background.wav",  "background",    1.0),   # фон
    ("jet_sonicboom.wav","sonic_boom",   1.5),   # сверхзвуковой (очень громко)
    ("background.wav",  "background",    1.0),   # фон
    ("jet_civil.wav",   "jet_civil_far", 0.25),  # гражданский далеко (тихо)
]

FRAME_SEC = 2.0


def _read_wav_chunks(path: str, duration_sec: float, volume: float = 1.0):
    """
    Read WAV file (any bit depth) and return list of (mono int16 list, framerate).
    Handles 8, 16, 24, 32-bit PCM.
    """
    chunks = []
    with wave.open(path, 'rb') as w:
        n_ch      = w.getnchannels()
        framerate = w.getframerate()
        sampwidth = w.getsampwidth()
        chunk_n   = int(framerate * duration_sec)
        total_frames = w.getnframes()

        while w.tell() + chunk_n <= total_frames:
            raw = w.readframes(chunk_n)
            arr = _raw_to_int16(raw, sampwidth, n_ch)
            if arr is None:
                break

            # Apply volume
            if volume != 1.0:
                arr = np.clip(arr * volume, -32768, 32767).astype(np.int16)

            # Mix to mono
            if n_ch == 2:
                mono = ((arr[0::2].astype(np.int32) + arr[1::2].astype(np.int32)) // 2).astype(np.int16)
            else:
                mono = arr

            chunks.append((mono.tolist(), framerate))
    return chunks


def _raw_to_int16(raw: bytes, sampwidth: int, n_ch: int) -> np.ndarray:
    """Convert raw PCM bytes to int16 numpy array (all channels interleaved)."""
    if sampwidth == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        return (arr * 256).astype(np.int16)
    elif sampwidth == 2:
        return np.frombuffer(raw, dtype=np.int16).copy()
    elif sampwidth == 3:
        # 24-bit: 3 bytes per sample, convert to int32 then scale to int16
        n = len(raw) // 3
        arr32 = np.zeros(n, dtype=np.int32)
        raw_np = np.frombuffer(raw, dtype=np.uint8)
        arr32 = (raw_np[0::3].astype(np.int32) |
                 (raw_np[1::3].astype(np.int32) << 8) |
                 (raw_np[2::3].astype(np.int32) << 16))
        # Sign extend from 24 to 32 bits
        arr32 = np.where(arr32 >= 0x800000, arr32 - 0x1000000, arr32)
        return (arr32 >> 8).astype(np.int16)
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).copy()
        return (arr >> 16).astype(np.int16)
    return None


async def run_file_player(frame_callback, broadcast_audio_state=None):
    logger.info("File player started, %d tracks in playlist", len(PLAYLIST))
    scorer = AircraftScorer()

    while True:
        for filename, label, volume in PLAYLIST:
            path = os.path.join(AUDIO_DIR, filename)
            if not os.path.exists(path):
                logger.warning("Audio file not found: %s", path)
                continue

            logger.info("Playing: %s (label=%s volume=%.2f)", filename, label, volume)

            if broadcast_audio_state:
                await broadcast_audio_state(label, volume)

            chunks = await asyncio.get_event_loop().run_in_executor(
                None, _read_wav_chunks, path, FRAME_SEC, volume
            )

            for mono, framerate in chunks:
                ts       = datetime.now(timezone.utc).isoformat()
                features = compute_spectral_features(mono, framerate)
                scorer.push(features)
                result   = scorer.score()

                frame = {
                    "ts":               ts,
                    "db_level":         features["db_level"],
                    "confidence":       result["confidence"],
                    "label":            label,
                    "lf_ratio":         features["lf_ratio"],
                    "spectral_centroid": features["spectral_centroid"],
                    "scores":           result["scores"],
                }
                logger.debug(
                    "db=%.1f lf=%.2f ct=%.0f conf=%.3f label=%s",
                    features["db_level"], features["lf_ratio"],
                    features["spectral_centroid"], result["confidence"], label
                )

                await frame_callback(frame)
                await asyncio.sleep(FRAME_SEC)
