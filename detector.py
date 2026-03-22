"""
Real microphone detector using YAMNet TFLite.
Produces frames identical to simulator.py output.

Requirements (on Raspberry Pi):
    pip install pyaudio numpy tflite-runtime

YAMNet model file:
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/lite-model_yamnet_classification_tflite_1.tflite
    mv lite-model_yamnet_classification_tflite_1.tflite yamnet.tflite
"""

import asyncio
import logging
import os
import struct
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE   = 16000
WINDOW_SEC    = 2.0
WINDOW_FRAMES = int(SAMPLE_RATE * WINDOW_SEC)  # 32000 samples
MODEL_PATH    = os.environ.get("YAMNET_MODEL", "yamnet.tflite")

# YAMNet AudioSet class indices for aircraft sounds
# https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
AIRCRAFT_CLASSES = {
    0:   "Aircraft",
    1:   "Airplane",
    2:   "Jet aircraft",
    3:   "Light aircraft, 'small aircraft'",
    4:   "Propeller, airscrew",
    311: "Helicopter",
}

REFERENCE_AMPLITUDE = 32768.0  # int16 max → 0 dBFS


def _load_model():
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite

    interp = tflite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp


def _calc_db(samples: np.ndarray) -> float:
    """RMS dB SPL (relative, 0 dBFS = 94 dB SPL)."""
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    rms = max(rms, 1.0)  # avoid log(0)
    db = 20 * np.log10(rms / REFERENCE_AMPLITUDE) + 94.0
    return round(float(db), 1)


def _classify(interp, samples: np.ndarray) -> float:
    """Run YAMNet, return max probability across aircraft classes."""
    audio = samples.astype(np.float32) / REFERENCE_AMPLITUDE  # normalize -1..1

    # YAMNet TFLite expects shape (1, num_samples)
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()

    # Resize input if needed (some builds require fixed size)
    interp.resize_tensor_input(input_details[0]['index'], [1, len(audio)])
    interp.allocate_tensors()

    interp.set_tensor(input_details[0]['index'], audio[np.newaxis, :])
    interp.invoke()

    scores = interp.get_tensor(output_details[0]['index'])[0]  # shape (521,)
    aircraft_conf = max(scores[i] for i in AIRCRAFT_CLASSES if i < len(scores))
    return round(float(aircraft_conf), 3)


async def run_detector(frame_callback):
    """
    Captures audio from the default input device, runs YAMNet classification,
    and calls frame_callback(frame) every WINDOW_SEC seconds.
    """
    import pyaudio

    logger.info("Loading YAMNet model from %s", MODEL_PATH)
    interp = _load_model()
    logger.info("YAMNet loaded")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=WINDOW_FRAMES,
    )
    logger.info("Microphone opened (%d Hz, %.1fs window)", SAMPLE_RATE, WINDOW_SEC)

    loop = asyncio.get_event_loop()

    try:
        while True:
            # Read one window (blocking, run in executor to not block event loop)
            raw = await loop.run_in_executor(
                None, stream.read, WINDOW_FRAMES, False
            )

            samples = np.frombuffer(raw, dtype=np.int16)

            db_level   = _calc_db(samples)
            confidence = _classify(interp, samples)
            ts         = datetime.now(timezone.utc).isoformat()

            frame = {
                "ts":         ts,
                "db_level":   db_level,
                "confidence": confidence,
                "adsb":       None,
            }

            logger.debug("Frame: db=%.1f conf=%.3f", db_level, confidence)
            await frame_callback(frame)

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
