"""Background resource sampler: raw process RSS + CPU% samples.

The server collects raw samples; the benchmark client reduces them to avg/peak.
The server never aggregates (separation of concerns).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    t_unix: float
    rss_mb: float
    cpu_percent: float


class ResourceSampler:
    """Samples this process's RSS and CPU% on a background thread."""

    def __init__(self, interval_sec: float = 0.05) -> None:
        import psutil

        self._proc = psutil.Process()
        self._interval = interval_sec
        self._samples: list[Sample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc.cpu_percent(None)  # prime the cpu_percent baseline

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            rss_mb = self._proc.memory_info().rss / 1e6
            cpu = self._proc.cpu_percent(None)
            with self._lock:
                self._samples.append(Sample(time.time(), rss_mb, cpu))

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
        self._proc.cpu_percent(None)

    def samples(self) -> list[Sample]:
        with self._lock:
            return list(self._samples)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
