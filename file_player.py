"""
File player: replaces simulator.py for testing with real audio files.
Playlist of aircraft sounds at different noise levels.
Uses spectral analysis (not just dB) to compute confidence.
"""

import asyncio
import logging
import os
import struct
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
    """Read WAV and return list of (mono_samples, framerate)."""
    chunks = []
    with wave.open(path, 'rb') as w:
        n_ch      = w.getnchannels()
        framerate = w.getframerate()
        sampwidth = w.getsampwidth()
        chunk_n   = int(framerate * duration_sec)

        while True:
            raw = w.readframes(chunk_n)
            if len(raw) < chunk_n * n_ch * sampwidth:
                break
            n_samples = len(raw) // sampwidth
            fmt = f"<{n_samples}h"
            samples = list(struct.unpack(fmt, raw))

            # Mix stereo → mono
            if n_ch == 2:
                mono = [(samples[i] + samples[i+1]) // 2
                        for i in range(0, len(samples), 2)]
            else:
                mono = samples

            # Apply volume scale (clip to int16)
            if volume != 1.0:
                mono = [max(-32768, min(32767, int(s * volume))) for s in mono]

            chunks.append((mono, framerate))
    return chunks


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
