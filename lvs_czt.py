"""CZT-based receiver stage for the LVS simulation."""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from lvs_rx import Receiver


class CztReceiver(Receiver):
    """Receiver pipeline that replaces the FFT stage with a CZT stage."""

    def __init__(self, config, mode: str) -> None:
        try:
            from scipy.signal import czt as scipy_czt  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires scipy
            raise RuntimeError("CztReceiver requires scipy to be installed.") from exc
        super().__init__(config, mode)
        self._scipy_czt = scipy_czt
        self._czt_bins = int(getattr(config, "czt_bins", self._fft_size))
        self._czt_span_hz = float(getattr(config, "czt_span_hz", 200.0))
        self._czt_center_hz = float(getattr(config, "czt_center_hz", 457000.0))

    def process_chunk(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.float64],
    ):
        """Run the RX pipeline with CZT detection, or CZT-only when pipeline_idx == 4."""
        if int(getattr(self.config, "pipeline_idx", 0)) == 4:
            return self._process_czt_only(time, samples)
        return super().process_chunk(time, samples)

    def _process_czt_only(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.float64],
    ):
        """Compute CZT directly on the incoming samples (no RX pipeline)."""
        if time.size == 0 or samples.size == 0:
            return self._empty_outputs()

        sample_rate = float(getattr(self.config, "czt_sample_rate", 0.0))
        if sample_rate <= 0.0:
            sample_rate = self._estimate_sample_rate(time) or 0.0
        if sample_rate <= 0.0:
            return self._empty_outputs()

        nfft = self._fft_size
        m = max(1, int(self._czt_bins))
        threshold_db = float(getattr(self.config, "fft_threshold_db", 8.0))
        min_mag = float(getattr(self.config, "fft_min_magnitude", 1e-9))

        data = samples.astype(np.complex128, copy=False)
        data = data - np.mean(data)
        if data.size < nfft:
            pad = np.zeros(nfft - data.size, dtype=np.complex128)
            data = np.concatenate((data, pad))
        else:
            data = data[:nfft]

        span = max(0.0, self._czt_span_hz)
        center = self._czt_center_hz
        f_start = center - span / 2.0
        f_end = center + span / 2.0

        window_type = str(getattr(self.config, "czt_window", "none")).lower()
        if window_type == "hann":
            window = np.hanning(nfft)
        else:
            window = np.ones(nfft, dtype=np.float64)
        W = np.exp(-1j * 2.0 * np.pi * (f_end - f_start) / (m * sample_rate))
        A = np.exp(1j * 2.0 * np.pi * f_start / sample_rate)
        spectrum = self._scipy_czt(data * window, m=m, w=W, a=A)
        magnitude = np.abs(spectrum)
        freqs = np.linspace(f_start, f_end, m, endpoint=False)

        noise_floor = float(np.median(magnitude) + min_mag)
        threshold = noise_floor * (10.0 ** (threshold_db / 20.0))
        peak_idx = int(np.argmax(magnitude)) if magnitude.size else -1

        detection_hz = None
        detection_mag = None
        snr_db = None
        if peak_idx >= 0:
            signal_mag = float(magnitude[peak_idx])
            snr_linear = signal_mag / max(noise_floor, min_mag)
            snr_db = float(20.0 * np.log10(snr_linear)) if snr_linear > 0.0 else None
            if signal_mag >= threshold:
                detection_mag = signal_mag
                detection_hz = float(freqs[peak_idx])

        return self._build_outputs(
            time=time,
            processed=samples.astype(np.float64, copy=False),
            fft_freqs=freqs,
            fft_magnitude=magnitude,
            detection_hz=detection_hz,
            detection_mag=detection_mag,
            snr_db=snr_db,
        )

    def _build_outputs(
        self,
        *,
        time: NDArray[np.float64],
        processed: NDArray[np.float64],
        fft_freqs: NDArray[np.float64],
        fft_magnitude: NDArray[np.float64],
        detection_hz: Optional[float],
        detection_mag: Optional[float],
        snr_db: Optional[float],
    ):
        return self._outputs_type()(
            time=time,
            processed=processed,
            fft_freqs=fft_freqs,
            fft_magnitude=fft_magnitude,
            detection_hz=detection_hz,
            detection_mag=detection_mag,
            snr_db=snr_db,
        )

    def _empty_outputs(self):
        return self._outputs_type()(
            time=np.empty(0, dtype=np.float64),
            processed=np.empty(0, dtype=np.float64),
        )

    @staticmethod
    def _outputs_type():
        from lvs_rx import ReceiverOutputs

        return ReceiverOutputs

    def _fft_stage(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.complex128],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """
        Compute a windowed CZT over a narrow frequency span and return
        the strongest detection.
        """
        if samples.size == 0 or time.size == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                None,
                None,
                None,
            )

        sample_rate = self._estimate_sample_rate(time)
        if sample_rate is None or sample_rate <= 0.0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                None,
                None,
                None,
            )

        nfft = self._fft_size
        m = max(1, int(self._czt_bins))
        threshold_db = float(getattr(self.config, "fft_threshold_db", 8.0))
        min_mag = float(getattr(self.config, "fft_min_magnitude", 1e-9))

        data = samples.astype(np.complex128, copy=False)
        if data.size < nfft:
            pad = np.zeros(nfft - data.size, dtype=np.complex128)
            data = np.concatenate((data, pad))
        else:
            data = data[:nfft]

        span = max(0.0, self._czt_span_hz)
        center = self._czt_center_hz
        nyquist = sample_rate / 2.0
        span = min(span, sample_rate)
        f_start = max(-nyquist, center - span / 2.0)
        f_end = min(nyquist, center + span / 2.0)
        if f_end <= f_start:
            f_start = -min(nyquist, span / 2.0)
            f_end = min(nyquist, span / 2.0)

        window = np.hanning(nfft)
        W = np.exp(-1j * 2.0 * np.pi * (f_end - f_start) / (m * sample_rate))
        A = np.exp(1j * 2.0 * np.pi * f_start / sample_rate)
        spectrum = self._scipy_czt(data * window, m=m, w=W, a=A)
        magnitude = np.abs(spectrum)
        freqs = np.linspace(f_start, f_end, m, endpoint=False)

        noise_floor = float(np.median(magnitude) + min_mag)
        threshold = noise_floor * (10.0 ** (threshold_db / 20.0))

        peak_idx = int(np.argmax(magnitude)) if magnitude.size else -1
        if peak_idx < 0:
            return freqs, magnitude, None, None, None

        signal_mag = float(magnitude[peak_idx])
        snr_linear = signal_mag / max(noise_floor, min_mag)
        snr_db = float(20.0 * np.log10(snr_linear)) if snr_linear > 0.0 else None
        if signal_mag < threshold:
            return freqs, magnitude, None, None, snr_db

        return freqs, magnitude, float(freqs[peak_idx]), signal_mag, snr_db
