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
    fft_freqs: Optional[NDArray[np.float64]] = None
    fft_magnitude: Optional[NDArray[np.float64]] = None
    detection_hz: Optional[float] = None
    detection_mag: Optional[float] = None


class Receiver:
    """DSP pipeline for the MCU-side processing (NCO → mixer → CIC → FIR)."""

    def __init__(self, config: RxConfig, mode: str) -> None:
        self.config = config
        self._mode = mode

        self._filter_type = getattr(config, "filter_type", "fir").lower()
        if self._filter_type not in {"fir", "iir"}:
            raise ValueError(f"Unsupported filter_type '{self._filter_type}'")

        self._nco_phase: float = 0.0

        self._cic_stages = config.cic_stages
        self._cic_decimation = config.cic_decimation_full if mode == "full" else config.cic_decimation_subsample

        self._fir_coefficients = self._init_fir_coefficients(config)
        self._fir_decimation = config.fir_decimation_full if mode == "full" else 1
        (
            self._iir_b_coefficients,
            self._iir_a_coefficients,
        ) = self._init_iir_coefficients(config)
        self._iir_decimation = (
            config.iir_decimation_full if mode == "full" else config.iir_decimation_subsample
        )
        self._init_iir_buffers()
        self._fft_size = config.fft_size_full if mode == "full" else config.fft_size_subsample

        self._reset_cic_state()
        self._reset_fir_state()
        self._reset_iir_state()

    def reset(self) -> None:
        """Reset any accumulated receiver state."""
        self._nco_phase = 0.0
        self._reset_cic_state()
        self._reset_fir_state()
        self._reset_iir_state()

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

        filter_stage = self._iir_stage if self._filter_type == "iir" else self._fir_stage
        filt_complex, filt_time = filter_stage(cic_time, cic_complex)
        
        if self.config.pipeline_idx == 2:
            processed = filt_complex.real.astype(np.float64, copy=False)
            return ReceiverOutputs(time=filt_time, processed=processed)

        fft_freqs, fft_mag, detection_hz, detection_mag = self._fft_stage(filt_time, filt_complex)
        processed = filt_complex.real.astype(np.float64, copy=False)
        return ReceiverOutputs(
            time=filt_time,
            processed=processed,
            fft_freqs=fft_freqs,
            fft_magnitude=fft_mag,
            detection_hz=detection_hz,
            detection_mag=detection_mag,
        )

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
        """Apply an FIR filter (+ decimation) with stateful overlap."""

        taps = self._fir_coefficients
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
    
    def _iir_stage(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.complex128],
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
        """Apply an IIR (biquad-style) filter with optional decimation."""
        if samples.size == 0:
            return samples, np.empty(0, dtype=np.float64)

        order = self._iir_order
        b = self._iir_b_padded
        a = self._iir_a_padded
        data = samples.astype(np.complex128, copy=False)

        if order == 0:
            filtered = data * b[0]
        else:
            filtered = np.empty_like(data, dtype=np.complex128)
            z = self._iir_state.copy()
            for idx, sample in enumerate(data):
                acc = b[0] * sample + z[0]
                filtered[idx] = acc
                for state_idx in range(order - 1):
                    z[state_idx] = z[state_idx + 1] + b[state_idx + 1] * sample - a[state_idx + 1] * acc
                z[order - 1] = b[order] * sample - a[order] * acc
            self._iir_state = z

        decim = max(1, int(self._iir_decimation))
        if decim > 1:
            filtered, mask = self._decimate_with_phase(filtered, decim, phase_attr="_iir_phase")
        else:
            mask = np.ones(filtered.size, dtype=bool)

        if filtered.size == 0:
            return filtered, np.empty(0, dtype=np.float64)

        if mask.size != time.size:
            raise RuntimeError("IIR decimation mask size does not match time array size")

        return filtered, time[mask]

    def _fft_stage(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.complex128],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[float], Optional[float]]:
        """
        Compute a windowed FFT over the chunk, detect peaks above a dynamic threshold,
        and return the strongest detection.
        """
        if samples.size == 0 or time.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), None, None

        sample_rate = self._estimate_sample_rate(time)
        if sample_rate is None or sample_rate <= 0.0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), None, None
        
        nfft = self._fft_size
        threshold_db = float(getattr(self.config, "fft_threshold_db", 8.0))
        min_mag = float(getattr(self.config, "fft_min_magnitude", 1e-9))

        data = samples.astype(np.complex128, copy=False)
        if data.size < nfft:
            pad = np.zeros(nfft - data.size, dtype=np.complex128)
            data = np.concatenate((data, pad))
        else:
            data = data[:nfft]

        window = np.hanning(nfft)
        spectrum = np.fft.fft(data * window, n=nfft)
        # Keep non-negative frequencies for magnitude view
        half_n = nfft // 2 + 1
        spectrum = spectrum[:half_n]
        magnitude = np.abs(spectrum)
        freqs = np.fft.fftfreq(nfft, d=1.0 / sample_rate)[:half_n]

        noise_floor = float(np.median(magnitude) + min_mag)
        threshold = noise_floor * (10.0 ** (threshold_db / 20.0))

        peak_idx = int(np.argmax(magnitude)) if magnitude.size else -1
        if peak_idx < 0 or magnitude[peak_idx] < threshold:
            return freqs, magnitude, None, None

        return freqs, magnitude, float(freqs[peak_idx]), float(magnitude[peak_idx])

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
        taps_len = int(self._fir_coefficients.size)
        if taps_len > 1:
            self._fir_state = np.zeros(taps_len - 1, dtype=np.complex128)
        else:
            self._fir_state = np.empty(0, dtype=np.complex128)
        self._fir_phase = 0
    
    def _reset_iir_state(self) -> None:
        """Initialize IIR filter state memory."""
        if getattr(self, "_iir_order", 0) > 0:
            self._iir_state = np.zeros(self._iir_order, dtype=np.complex128)
        else:
            self._iir_state = np.empty(0, dtype=np.complex128)
        self._iir_phase = 0

    def _init_fir_coefficients(self, config: RxConfig) -> NDArray[np.float64]:
        """Return FIR taps from config or a simple placeholder."""
        if self._mode == "full":
            taps = np.asarray(getattr(config, "fir_coefficients_full", (1.0,)), dtype=np.float64)
        else:
            taps = np.asarray(getattr(config, "fir_coefficients_subsample", (1.0,)), dtype=np.float64)
        if taps.size == 0:
            return np.asarray([1.0], dtype=np.float64)
        return taps
    
    def _init_iir_coefficients(self, config: RxConfig) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return normalized IIR numerator/denominator coefficients."""
        b = np.asarray(getattr(config, "iir_b_coefficients", (1.0,)), dtype=np.float64)
        a = np.asarray(getattr(config, "iir_a_coefficients", (1.0,)), dtype=np.float64)
        if b.size == 0:
            b = np.asarray([1.0], dtype=np.float64)
        if a.size == 0:
            a = np.asarray([1.0], dtype=np.float64)
        if a[0] == 0.0:
            raise ValueError("IIR denominator a[0] may not be zero")
        b_norm = b / a[0]
        a_norm = a / a[0]
        return b_norm, a_norm

    def _init_iir_buffers(self) -> None:
        """Prepare padded coefficient arrays for the IIR stage."""
        order = max(int(self._iir_a_coefficients.size), int(self._iir_b_coefficients.size)) - 1
        self._iir_order = max(order, 0)
        padded_len = self._iir_order + 1
        self._iir_b_padded = np.zeros(padded_len, dtype=np.float64)
        self._iir_b_padded[: self._iir_b_coefficients.size] = self._iir_b_coefficients
        self._iir_a_padded = np.zeros(padded_len, dtype=np.float64)
        self._iir_a_padded[: self._iir_a_coefficients.size] = self._iir_a_coefficients

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
