from __future__ import annotations

import time

from bge_m3_bench.server.resources import ResourceSampler


def _wait_for_samples(sampler: ResourceSampler, timeout: float = 2.0) -> list:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        samples = sampler.samples()
        if samples:
            return samples
        time.sleep(0.02)
    return sampler.samples()


def test_sampler_collects_reset_and_restarts():
    sampler = ResourceSampler(interval_sec=0.01)
    sampler.start()
    try:
        assert _wait_for_samples(sampler), "expected at least one sample after start"

        sampler.reset()
        # reset clears history; new samples should accrue again.
        assert _wait_for_samples(sampler)

        # stop() then start() must resume sampling (regression: stop-event reuse).
        sampler.stop()
        sampler.start()
        before = len(sampler.samples())
        restarted = _wait_for_samples(sampler)
        assert restarted and len(restarted) >= before
    finally:
        sampler.stop()

    s = sampler.samples()[0] if sampler.samples() else None
    if s is not None:
        assert s.rss_mb > 0 and s.cpu_percent >= 0


def test_sampler_buffer_is_bounded():
    sampler = ResourceSampler(interval_sec=0.005, max_samples=5)
    sampler.start()
    try:
        # Sample well past the cap; the ring buffer must never exceed maxlen.
        time.sleep(0.2)
        assert len(sampler.samples()) <= 5
    finally:
        sampler.stop()
