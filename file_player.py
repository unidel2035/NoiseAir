"""
File player: replaces simulator.py for testing with real audio files.
Loops: background.wav → aircraft.wav → background.wav → ...
Computes real dB from audio samples, sends frames to event_tracker.
Browser plays the same files via Web Audio API.
"""

import asyncio
import logging
import math
import struct
import wave
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

AUDIO_DIR   = os.path.join(os.path.dirname(__file__), "audio")
AIRCRAFT_WAV = os.path.join(AUDIO_DIR, "aircraft.wav")
BACKGROUND_WAV = os.path.join(AUDIO_DIR, "background.wav")

FRAME_SEC    = 2.0
NOISE_FLOOR_DB = 42.0

# Playlist: (file, label, repeat_times)
PLAYLIST = [
    (BACKGROUND_WAV, "background", 1),
    (AIRCRAFT_WAV,   "aircraft",   1),
    (BACKGROUND_WAV, "background", 1),
    (AIRCRAFT_WAV,   "aircraft",   1),
]


def _read_wav_frames(path: str, duration_sec: float) -> list:
    """Read WAV file and return list of (samples_array, n_channels, framerate) per chunk."""
    chunks = []
    with wave.open(path, 'rb') as w:
        n_channels  = w.getnchannels()
        framerate   = w.getframerate()
        sampwidth   = w.getsampwidth()
        chunk_frames = int(framerate * duration_sec)

        while True:
            raw = w.readframes(chunk_frames)
            if len(raw) < chunk_frames * n_channels * sampwidth // 2:
                break
            # Unpack to int16
            n_samples = len(raw) // sampwidth
            fmt = f"<{n_samples}{'h' if sampwidth == 2 else 'b'}"
            samples = list(struct.unpack(fmt, raw))
            # Mix to mono
            if n_channels == 2:
                mono = [(samples[i] + samples[i+1]) // 2 for i in range(0, len(samples), 2)]
            else:
                mono = samples
            chunks.append(mono)
    return chunks


def _rms_db(samples: list) -> float:
    if not samples:
        return NOISE_FLOOR_DB
    mean_sq = sum(s * s for s in samples) / len(samples)
    rms = math.sqrt(max(mean_sq, 1.0))
    db = 20 * math.log10(rms / 32768.0) + 94.0
    return round(db, 1)


def _db_to_confidence(db: float) -> float:
    delta = db - NOISE_FLOOR_DB
    if delta < 5:
        return 0.05
    conf = 1 / (1 + math.exp(-0.3 * (delta - 10)))
    return round(min(1.0, conf), 3)


async def run_file_player(frame_callback, broadcast_audio_state=None):
    """
    Loops through PLAYLIST, reads audio in 2-sec chunks,
    computes real dB, sends frames.
    broadcast_audio_state(label) notifies browser which file is playing.
    """
    logger.info("File player started")

    while True:
        for wav_path, label, repeats in PLAYLIST:
            for _ in range(repeats):
                logger.info("Playing: %s (%s)", label, os.path.basename(wav_path))

                if broadcast_audio_state:
                    await broadcast_audio_state(label)

                chunks = await asyncio.get_event_loop().run_in_executor(
                    None, _read_wav_frames, wav_path, FRAME_SEC
                )

                for chunk in chunks:
                    ts       = datetime.now(timezone.utc).isoformat()
                    db_level = _rms_db(chunk)
                    confidence = _db_to_confidence(db_level)

                    frame = {
                        "ts":         ts,
                        "db_level":   db_level,
                        "confidence": confidence,
                        "label":      label,
                    }
                    logger.debug("Frame: db=%.1f conf=%.3f label=%s", db_level, confidence, label)

                    await frame_callback(frame)
                    await asyncio.sleep(FRAME_SEC)
