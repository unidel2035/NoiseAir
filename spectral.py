"""
Spectral analysis for aircraft sound classification.
Combines multiple criteria beyond simple dB level.

Criteria:
  1. yamnet_score  - AI class probability (when real mic available)
  2. lf_ratio      - low-frequency energy ratio (80-500 Hz)
  3. envelope_score - bell-curve shape over time (rise → peak → fall)
  4. duration_score - sustained signal > 15 sec
  5. continuity    - no sharp interruptions (unlike speech/horn)
"""

import math
import struct
import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100  # original WAV rate (resampled later if needed)

# Frequency bands (Hz)
LF_BAND   = (80,  500)   # jet engine fundamentals
MID_BAND  = (500, 2000)  # turbine harmonics
HF_BAND   = (2000, 8000) # high-frequency content (wind, birds)


def compute_spectral_features(samples: list, framerate: int) -> dict:
    """
    Compute spectral features from a mono int16 sample chunk.
    Returns dict with: lf_ratio, mid_ratio, spectral_centroid, db_level
    """
    if len(samples) < 512:
        return {"lf_ratio": 0.0, "mid_ratio": 0.0, "spectral_centroid": 1000.0, "db_level": 42.0}

    arr = np.array(samples, dtype=np.float32)

    # dB level
    rms = math.sqrt(max(float(np.mean(arr ** 2)), 1.0))
    db_level = round(20 * math.log10(rms / 32768.0) + 94.0, 1)

    # FFT
    N = min(len(arr), 4096)
    window = np.hanning(N)
    arr_w = arr[:N] * window
    fft_mag = np.abs(np.fft.rfft(arr_w))
    freqs = np.fft.rfftfreq(N, d=1.0 / framerate)

    total_energy = float(np.sum(fft_mag ** 2)) + 1e-9

    def band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(fft_mag[mask] ** 2))

    lf_energy  = band_energy(*LF_BAND)
    mid_energy = band_energy(*MID_BAND)
    hf_energy  = band_energy(*HF_BAND)

    lf_ratio  = round(lf_energy / total_energy, 3)
    mid_ratio = round(mid_energy / total_energy, 3)

    # Spectral centroid (weighted mean frequency)
    centroid = float(np.sum(freqs * fft_mag)) / (float(np.sum(fft_mag)) + 1e-9)

    return {
        "lf_ratio":         lf_ratio,
        "mid_ratio":        mid_ratio,
        "spectral_centroid": round(centroid, 1),
        "db_level":         db_level,
    }


class AircraftScorer:
    """
    Maintains a rolling window of frames and computes a combined
    aircraft confidence score using multiple criteria.
    """

    WINDOW = 15  # number of 2-sec frames to analyse (~30 sec)

    def __init__(self):
        self._db_history:      deque = deque(maxlen=self.WINDOW)
        self._lf_history:      deque = deque(maxlen=self.WINDOW)
        self._centroid_history: deque = deque(maxlen=self.WINDOW)

    def push(self, features: dict):
        self._db_history.append(features["db_level"])
        self._lf_history.append(features["lf_ratio"])
        self._centroid_history.append(features["spectral_centroid"])

    def score(self, yamnet_score: float = 0.0) -> dict:
        """
        Returns dict:
          confidence  - combined 0..1
          scores      - breakdown of each component
        """
        n = len(self._db_history)
        if n < 2:
            return {"confidence": 0.0, "scores": {}}

        db_list  = list(self._db_history)
        lf_list  = list(self._lf_history)
        ct_list  = list(self._centroid_history)

        # ── 1. LF ratio score ─────────────────────────────────────────────────
        # Aircraft: LF band dominates (ratio > 0.3)
        avg_lf = sum(lf_list) / len(lf_list)
        lf_score = min(1.0, avg_lf / 0.4)  # 0.4 LF ratio → full score

        # ── 2. Spectral centroid score ────────────────────────────────────────
        # Aircraft centroid typically 200–800 Hz
        # Birds/wind: > 2000 Hz, traffic: 100–300 Hz
        avg_ct = sum(ct_list) / len(ct_list)
        if 150 <= avg_ct <= 900:
            ct_score = 1.0
        elif avg_ct < 150:
            ct_score = max(0.0, avg_ct / 150)
        else:
            ct_score = max(0.0, 1.0 - (avg_ct - 900) / 2000)

        # ── 3. Envelope score (bell curve) ────────────────────────────────────
        # Rise then fall pattern: first half avg < second half avg → not a bell
        # Bell: peak in the middle
        if n >= 6:
            third = n // 3
            start_avg = sum(db_list[:third]) / third
            mid_avg   = sum(db_list[third:2*third]) / third
            end_avg   = sum(db_list[2*third:]) / (n - 2*third)
            # Good bell: mid > start AND mid > end
            bell = (mid_avg - start_avg) + (mid_avg - end_avg)
            envelope_score = min(1.0, max(0.0, bell / 10.0))
        else:
            envelope_score = 0.3  # not enough data yet

        # ── 4. Duration score ─────────────────────────────────────────────────
        duration_sec = n * 2.0
        duration_score = min(1.0, duration_sec / 30.0)  # 30 sec → full score

        # ── 5. Continuity score ───────────────────────────────────────────────
        # No sudden drops (aircraft is continuous, horn is not)
        diffs = [abs(db_list[i] - db_list[i-1]) for i in range(1, len(db_list))]
        avg_diff = sum(diffs) / len(diffs)
        continuity_score = max(0.0, 1.0 - avg_diff / 15.0)

        # ── Combined score ────────────────────────────────────────────────────
        # Weights
        W_yamnet      = 0.35
        W_lf          = 0.25
        W_centroid    = 0.15
        W_envelope    = 0.10
        W_duration    = 0.10
        W_continuity  = 0.05

        if yamnet_score > 0:
            confidence = (
                W_yamnet     * yamnet_score +
                W_lf         * lf_score +
                W_centroid   * ct_score +
                W_envelope   * envelope_score +
                W_duration   * duration_score +
                W_continuity * continuity_score
            )
        else:
            # Without YAMNet — redistribute its weight
            total_w = W_lf + W_centroid + W_envelope + W_duration + W_continuity
            confidence = (
                W_lf         * lf_score +
                W_centroid   * ct_score +
                W_envelope   * envelope_score +
                W_duration   * duration_score +
                W_continuity * continuity_score
            ) / total_w

        return {
            "confidence": round(min(1.0, confidence), 3),
            "scores": {
                "yamnet":      round(yamnet_score, 3),
                "lf_ratio":    round(lf_score, 3),
                "centroid":    round(ct_score, 3),
                "envelope":    round(envelope_score, 3),
                "duration":    round(duration_score, 3),
                "continuity":  round(continuity_score, 3),
            }
        }
