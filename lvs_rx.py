"""Receiver DSP block for the LVS simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from config import RxConfig


@dataclass
class ReceiverOutputs:
    """Container for the receiver stages."""

    time: NDArray[np.float64]
    processed: NDArray[np.float64]


class Receiver:
    """DSP pipeline for the MCU-side processing (NCO → mixer → CIC → FIR)."""

    def __init__(self, config: RxConfig, mode: str) -> None:
        self.config = config
        self._mode = mode

        self._nco_phase: float = 0.0

        self._cic_stages = config.cic_stages
        self._cic_decimation = config.cic_decimation_full if mode == "full" else config.cic_decimation_subsample

        self._fir_taps = self._init_fir_taps(config)
        self._fir_decimation = config.fir_decimation_full if mode == "full" else 1

        self._reset_cic_state()
        self._reset_fir_state()

    def reset(self) -> None:
        """Reset any accumulated receiver state."""
        self._nco_phase = 0.0
        self._reset_cic_state()
        self._reset_fir_state()

    def process_chunk(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.float64],
    ) -> ReceiverOutputs:
        """Run the DSP chain on a chunk of samples."""
        if time.size == 0 or samples.size == 0:
            return ReceiverOutputs(
                time=np.empty(0, dtype=np.float64),
                processed=np.empty(0, dtype=np.float64),
            )

        baseband = self._mix_stage(time, samples)
        if self.config.pipeline_idx <= 0:
            processed = baseband.real.astype(np.float64, copy=False)
            return ReceiverOutputs(time=time, processed=processed)

        cic_complex, cic_time = self._cic_stage(time, baseband)
        if self.config.pipeline_idx == 1:
            processed = cic_complex.real.astype(np.float64, copy=False)
            return ReceiverOutputs(time=cic_time, processed=processed)

        fir_complex, fir_time = self._fir_stage(cic_time, cic_complex)
        processed = fir_complex.real.astype(np.float64, copy=False)
        return ReceiverOutputs(time=fir_time, processed=processed)

    def _mix_stage(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        """Generate the NCO, mix the input to baseband, and return complex baseband."""
        sample_rate = self._estimate_sample_rate(time)
        if sample_rate is None or sample_rate <= 0.0:
            # Without a valid sample rate, fall back to pass-through real-only.
            return samples.astype(np.complex128, copy=True)

        nco = self._generate_nco(len(samples), sample_rate)
        return samples.astype(np.complex128, copy=False) * nco

    def _estimate_sample_rate(self, time: NDArray[np.float64]) -> Optional[float]:
        """Estimate the sampling rate from incoming time stamps."""
        if time.size < 2:
            return None

        deltas = np.diff(time)
        positive = deltas[deltas > 0.0]
        if positive.size == 0:
            return None

        mean_dt = float(np.mean(positive))
        if mean_dt <= 0.0:
            return None

        return 1.0 / mean_dt

    def _cic_stage(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.complex128],
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
        """Apply the CIC integrator/comb decimator chain."""
        decimation = max(1, int(self._cic_decimation))
        data = samples.astype(np.complex128, copy=False)

        # Integrator stages
        for stage in range(self._cic_stages):
            data = np.cumsum(data, dtype=np.complex128)
            data += self._cic_integrator_state[stage]
            self._cic_integrator_state[stage] = data[-1]
            # add last chunk sample to integrator state

        # Decimation with phase tracking
        decimated, mask = self._decimate_with_phase(data, decimation, phase_attr="_cic_phase")
        if decimated.size == 0:
            return decimated, np.empty(0, dtype=np.float64)

        if mask.size == time.size:
            time_decimated = time[mask]
        else:
            raise RuntimeError("CIC decimation mask size does not match time array size")

        # Comb stages
        comb_data = decimated
        for stage in range(self._cic_stages):
            prev = self._cic_comb_state[stage]
            diff = np.empty_like(comb_data)
            diff[0] = comb_data[0] - prev
            if comb_data.size > 1:
                diff[1:] = comb_data[1:] - comb_data[:-1]
            self._cic_comb_state[stage] = comb_data[-1]
            comb_data = diff

        return comb_data, time_decimated

    def _fir_stage(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.complex128],
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
        """Apply an FIR filter (optional decimation) with stateful overlap."""

        taps = self._fir_taps
        if taps.size == 0:
            return samples, time

        combined = np.concatenate((self._fir_state, samples))
        filtered = np.convolve(combined, taps, mode="valid")

        if taps.size > 1:
            self._fir_state = combined[-(taps.size - 1) :]
            # save overlap for next chunk
        else:
            self._fir_state = np.empty(0, dtype=np.complex128)

        decim = max(1, int(self._fir_decimation))
        if decim > 1:
            filtered, mask = self._decimate_with_phase(
                filtered, decim, phase_attr="_fir_phase"
            )
        else:
            mask = np.ones(filtered.size, dtype=bool)

        if filtered.size == 0:
            return filtered, np.empty(0, dtype=np.float64)

        if mask.size != time.size:
            raise RuntimeError("FIR decimation mask size does not match time array size")

        # Group delay compensation
        delay_samples = (taps.size - 1) / 2.0
        dt = self._estimate_time_step(time)
        delay_offset = delay_samples * dt if dt is not None else 0.0
        time_out = time[mask] + delay_offset

        return filtered, time_out

    def _generate_nco(self, num_samples: int, sample_rate: float) -> NDArray[np.complex128]:
        """Produce a complex exponential for the configured carrier frequency."""
        if num_samples <= 0:
            return np.empty(0, dtype=np.complex128)

        freq = float(getattr(self.config, "carrier_freq", 0.0))
        if sample_rate <= 0.0:
            return np.ones(num_samples, dtype=np.complex128)

        phase_inc = 2.0 * np.pi * freq / sample_rate
        sample_indices = np.arange(num_samples, dtype=np.float64)
        phases = self._nco_phase + phase_inc * sample_indices
        self._nco_phase = float((phases[-1] + phase_inc) % (2.0 * np.pi))
        return np.exp(-1j * phases)

    def _decimate_with_phase(
        self,
        data: NDArray[np.complex128],
        decimation: int,
        *,
        phase_attr: str = "_cic_phase",
    ) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
        """Down-sample while retaining phase across calls using the specified phase attribute."""
        if data.size == 0:
            return data, np.zeros(0, dtype=bool)

        if decimation <= 1:
            mask = np.ones(data.size, dtype=bool)
            return data, mask

        phase_val = int(getattr(self, phase_attr, 0))
        indices = (phase_val + np.arange(data.size)) % decimation
        mask = indices == 0  # keep the earliest sample in each decimation window
        setattr(self, phase_attr, int((phase_val + data.size) % decimation))
        return data[mask], mask

    def _reset_cic_state(self) -> None:
        """Initialize CIC integrator and comb memories."""
        self._cic_integrator_state = np.zeros(self._cic_stages, dtype=np.complex128)
        self._cic_comb_state = np.zeros(self._cic_stages, dtype=np.complex128)
        self._cic_phase = 0

    def _reset_fir_state(self) -> None:
        """Initialize FIR overlap buffer."""
        taps_len = int(self._fir_taps.size)
        if taps_len > 1:
            self._fir_state = np.zeros(taps_len - 1, dtype=np.complex128)
        else:
            self._fir_state = np.empty(0, dtype=np.complex128)
        self._fir_phase = 0

    def _init_fir_taps(self, config: RxConfig) -> NDArray[np.float64]:
        """Return FIR taps from config or a simple placeholder."""
        taps = np.asarray(getattr(config, "fir_taps", (1.0,)), dtype=np.float64)
        if taps.size == 0:
            return np.asarray([1.0], dtype=np.float64)
        return taps

    def _estimate_time_step(self, time: NDArray[np.float64]) -> Optional[float]:
        """Return mean positive time step if available, else None."""
        if time.size < 2:
            return None

        deltas = np.diff(time)
        positive = deltas[deltas > 0.0]
        if positive.size == 0:
            return None

        dt = float(np.mean(positive))
        return dt if dt > 0.0 else None
