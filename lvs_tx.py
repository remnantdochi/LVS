"""Transmitter block for the LVS simulation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from config import TxConfig


@dataclass
class TxWaveformChunk:
    """ Metadata describing a continuous waveform segment produced by the TX."""

    start_time: float
    end_time: float
    plot_time: NDArray[np.float64]
    plot_samples: NDArray[np.float64]
    transmitter: "Transmitter"

    def sample(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample the underlying continuous waveform at arbitrary instants."""
        time_arr = np.asarray(time, dtype=np.float64)
        if time_arr.size == 0:
            return np.empty_like(time_arr)
        return self.transmitter.build_waveform(time_arr)


class Transmitter:
    """Generates beacon bursts based on the provided configuration."""

    def __init__(self, config: TxConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._sample_cursor = 0
        self._carrier_freq = config.center_freq
        self._pulse_length = config.pulse_length
        self._pulse_period = config.pulse_period
        self._extra_carrier_freqs: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._noise_std = 0.0
        self._noise_resolution = 1e-12
        self._noise_seed_base = 0.0
        if self.config.randomize:
            self._init_random_variation()
        self._init_extra_carriers()
        if self.config.use_awgn:
            self._init_noise_model()

    def reset(self) -> None:
        """Reset the internal state to the start of the signal."""
        self._sample_cursor = 0
        self._noise_seed_base = self._draw_noise_seed()

        if self.config.randomize:
            self._init_random_variation()
        self._init_extra_carriers()
        if self.config.use_awgn:
            self._init_noise_model()

    def generate_chunk(
        self,
        num_samples: Optional[int] = None,
    ) -> TxWaveformChunk:
        """Produce the next chunk of the transmit waveform for plotting."""
        n = self.config.chunk_size if num_samples is None else num_samples
        if n <= 0:
            raise ValueError("chunk_size must be positive")

        fs = self.config.fs
        start_idx = self._sample_cursor
        indices = start_idx + np.arange(n)
        time = indices / fs
        samples = self.build_waveform(time)
        self._sample_cursor += n

        return TxWaveformChunk(
            start_time=float(time[0]),
            end_time=float(time[-1]),
            plot_time=time,
            plot_samples=samples,
            transmitter=self,
        )

    def build_waveform(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compose the deterministic carrier, envelope, and AWGN components."""
        time_arr = np.asarray(time, dtype=np.float64)
        carrier = np.cos(2.0 * np.pi * self._carrier_freq * time_arr)
        if self._extra_carrier_freqs.size:
            extra = np.cos(2.0 * np.pi * self._extra_carrier_freqs[:, None] * time_arr[None, :])
            carrier = carrier + np.sum(extra, axis=0)

        if self.config.use_pulse_envelope:
            envelope = ((time_arr % self._pulse_period) < self._pulse_length).astype(np.float64)
        else:
            envelope = np.ones_like(time_arr, dtype=np.float64)

        waveform = envelope * carrier
        if self.config.use_awgn and self._noise_std > 0.0:
            waveform = waveform + self._awgn(time_arr)
        return waveform

    def _init_random_variation(self) -> None:
        """Initialize random offsets within the configured tolerances."""
        freq = self._rng.uniform(
            self.config.center_freq - self.config.freq_tolerance,
            self.config.center_freq + self.config.freq_tolerance,
        )
        self._carrier_freq = float(freq)

        period = self._rng.uniform(
            self.config.pulse_period - self.config.pulse_period_tolerance,
            self.config.pulse_period + self.config.pulse_period_tolerance,
        )
        self._pulse_period = float(max(period, self.config.pulse_length + 1e-6))

    def _init_extra_carriers(self) -> None:
        count = int(getattr(self.config, "extra_carriers", 0))
        if count <= 0:
            self._extra_carrier_freqs = np.empty(0, dtype=np.float64)
            return
        center = float(self.config.center_freq)
        if not getattr(self.config, "randomize", False):
            offsets = tuple(getattr(self.config, "extra_carrier_offsets", ()))
            if offsets:
                selected = np.asarray(offsets[:count], dtype=np.float64)
            else:
                half_span = float(getattr(self.config, "freq_tolerance", 0.0))
                grid = np.linspace(-half_span, half_span, count + 1, dtype=np.float64)
                selected = np.delete(grid, count // 2)
            self._extra_carrier_freqs = center + selected
            return

        half_span = float(getattr(self.config, "freq_tolerance", 0.0))
        self._extra_carrier_freqs = self._rng.uniform(
            center - half_span,
            center + half_span,
            size=count,
        ).astype(np.float64)

    def _init_noise_model(self) -> None:
        """Initialize the AWGN variance"""
        base_power = 0.5  # mean power of cos wave with amplitude = 1
        if self.config.use_pulse_envelope:
            duty_cycle = min(1.0, self._pulse_length / self._pulse_period)
            base_power *= duty_cycle

        snr_linear = 10.0 ** (self.config.awgn_snr / 10.0)
        noise_power = base_power / snr_linear
        self._noise_std = math.sqrt(noise_power)
        self._noise_seed_base = self._draw_noise_seed()

    def _awgn(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return deterministic AWGN samples keyed by quantized time."""
        if self._noise_std == 0.0:
            return np.zeros_like(time, dtype=np.float64)

        time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
        if time_arr.size == 0:
            return np.empty_like(time_arr)

        scale = 1.0 / self._noise_resolution
        keys = np.round(time_arr * scale).astype(np.int64)
        seeds = self._mix_keys(keys)

        unique_seeds, inverse = np.unique(seeds, return_inverse=True)
        noise_unique = np.empty_like(unique_seeds, dtype=np.float64)
        for idx, seed in enumerate(unique_seeds):
            rng = np.random.default_rng(int(seed))
            noise_unique[idx] = rng.normal(0.0, self._noise_std)

        noise = noise_unique[inverse]
        return noise.reshape(time.shape)

    def _draw_noise_seed(self) -> int:
        """Return a fresh random 64-bit seed derived from the main RNG."""
        return int(self._rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))

    def _mix_keys(self, keys: NDArray[np.int64]) -> NDArray[np.uint64]:
        """Hash integer time keys with the base seed to get deterministic RNG seeds."""
        mask = np.uint64(0xFFFFFFFFFFFFFFFF)
        mixed = (keys.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) & mask
        mixed ^= mixed >> np.uint64(33)
        mixed = (mixed * np.uint64(0xC2B2AE3D27D4EB4F)) & mask
        mixed ^= mixed >> np.uint64(29)
        mixed = (mixed * np.uint64(0x165667B19E3779F9)) & mask
        mixed ^= mixed >> np.uint64(32)
        return np.bitwise_xor(mixed, np.uint64(self._noise_seed_base))
