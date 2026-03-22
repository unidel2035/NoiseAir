"""
Event tracker: accumulates audio frames with aircraft classification,
detects start/end of flyover events, saves to DB.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime, timezone

import db

logger = logging.getLogger(__name__)

# Minimum confidence from classifier to count as aircraft signal
CONFIDENCE_THRESHOLD = 0.40
# Minimum consecutive seconds above threshold to open an event
MIN_OPEN_SEC = 3.0
# Seconds of silence (below threshold) before closing an event
CLOSE_GAP_SEC = 8.0
# Minimum total event duration to save (ignore short bursts)
MIN_EVENT_SEC = 6.0


@dataclass
class _ActiveEvent:
    ts_start: str
    db_samples: list = field(default_factory=list)
    confidence_samples: list = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)


class EventTracker:
    """
    Feed frames into push_frame().
    Each frame: { "ts": str, "db_level": float, "confidence": float }
    on_event(record): async callback called when an event is saved.
    """

    def __init__(self, on_event: Optional[Callable] = None):
        self._event: Optional[_ActiveEvent] = None
        self._pending_frames: list = []
        self._pending_start: float = 0.0
        self._on_event = on_event

    def push_frame(self, frame: dict):
        confidence = frame["confidence"]
        ts = frame["ts"]
        db_level = frame["db_level"]
        now = time.monotonic()

        if confidence >= CONFIDENCE_THRESHOLD:
            if self._event is None:
                if not self._pending_frames:
                    self._pending_start = now
                self._pending_frames.append(frame)

                if (now - self._pending_start) >= MIN_OPEN_SEC:
                    first = self._pending_frames[0]
                    self._event = _ActiveEvent(ts_start=first["ts"])
                    for pf in self._pending_frames:
                        self._event.db_samples.append(pf["db_level"])
                        self._event.confidence_samples.append(pf["confidence"])
                    self._pending_frames = []
                    logger.info("Event opened at %s", self._event.ts_start)
            else:
                self._event.db_samples.append(db_level)
                self._event.confidence_samples.append(confidence)
                self._event.last_active = now
        else:
            self._pending_frames = []
            if self._event is not None:
                if (now - self._event.last_active) >= CLOSE_GAP_SEC:
                    self._close_event(ts)

    def _close_event(self, ts_end: str):
        ev = self._event
        self._event = None

        duration = len(ev.db_samples) * 2.0  # assuming 2-sec frames
        if duration < MIN_EVENT_SEC:
            logger.debug("Event too short (%.1fs), discarded", duration)
            return

        peak_db = max(ev.db_samples)
        avg_db = sum(ev.db_samples) / len(ev.db_samples)
        avg_conf = sum(ev.confidence_samples) / len(ev.confidence_samples)

        record = {
            "ts_start":       ev.ts_start,
            "ts_end":         ts_end,
            "duration_sec":   round(duration, 1),
            "peak_db":        round(peak_db, 1),
            "avg_db":         round(avg_db, 1),
            "avg_confidence": round(avg_conf, 3),
            "icao":           None,
            "callsign":       None,
            "altitude_ft":    None,
            "distance_km":    None,
        }

        event_id = db.insert_event(record)
        record["id"] = event_id
        logger.info("Event saved id=%d  %s → %s  %.1fs  %.1fdB",
                    event_id, record["ts_start"], record["ts_end"],
                    duration, peak_db)

        if self._on_event:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._on_event(record))
            except RuntimeError:
                pass

    def flush(self, ts_end: str = None):
        """Force-close any open event (call on shutdown)."""
        if self._event:
            ts_end = ts_end or datetime.now(timezone.utc).isoformat()
            self._close_event(ts_end)
