"""Minimal ADC sampling stage for the LVS simulation."""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from config import AdcConfig
from lvs_tx import TxWaveformChunk


class LvsAdc:
    """Approximates the MCU ADC sampler."""

    def __init__(self, config: AdcConfig) -> None:
        self.config = config
        self._fs_full = float(getattr(config, "fs_full", 1e6))
        self._fs_subsample = float(getattr(config, "fs_subsample", 100e3))
        self._next_time: dict[float, float] = {}

    def process_chunk(
        self,
        chunk: TxWaveformChunk,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return time-aligned buffers at either the full or subsampled rate."""
        if not isinstance(chunk, TxWaveformChunk):
            raise TypeError("ADC expects a TxWaveformChunk input")

        if chunk.plot_time.size == 0:
            return chunk.plot_time, chunk.plot_samples

        full_time, full_chunk = self._sample_rate(chunk, self._fs_full)

        mode = self.config.mode.lower()
        if mode == "full":
            return full_time, full_chunk
        if mode != "subsample":
            raise ValueError(f"Unsupported ADC mode '{self.config.mode}'")

        subsampled_time, subsampled = self._sample_rate(chunk, self._fs_subsample)
        return subsampled_time, subsampled

    def _sample_rate(
        self,
        chunk: TxWaveformChunk,
        target_fs: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample the original TxWaveformChunk at the specified target sample rate."""
        if target_fs <= 0:
            raise ValueError("Target sample rate must be positive")

        times = self._build_time_grid(chunk, target_fs)
        if times.size == 0:
            return times, times

        samples = chunk.sample(times)
        return times, samples

    def _build_time_grid(self, chunk: TxWaveformChunk, target_fs: float) -> NDArray[np.float64]:
        """Construct a time grid for sampling within the chunk at the target sample rate."""
        dt = 1.0 / target_fs
        start = chunk.start_time
        end = chunk.end_time

        next_time = self._next_time.get(target_fs)
        if next_time is None:
            next_time = start

        """Align the next_time to be within [start, end] if chunk is ahead."""
        if next_time < start:
            steps = math.ceil((start - next_time) / dt)
            next_time += steps * dt

        times: list[float] = []
        tol = 1e-12 # Numerical tolerance for floating-point comparisons
        while next_time <= end + tol:
            times.append(next_time)
            next_time += dt

        self._next_time[target_fs] = next_time
        if not times:
            return np.empty(0, dtype=np.float64)
        return np.asarray(times, dtype=np.float64)
