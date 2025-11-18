"""Quick demo for visualizing the mix stage output."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from config import RxConfig
from lvs_rx import Receiver

def plot_mix_example() -> None:
    """Quick demo for visualizing the mix stage output."""
    cfg = RxConfig()
    cfg.pipeline_idx = 0
    rx = Receiver(cfg, mode="full")

    fs = 1e6
    duration = 20e-3  # longer to see multiple chunks
    time = np.arange(0.0, duration, 1.0 / fs, dtype=np.float64)

    rf_offset = 2e3
    rf_freq = cfg.carrier_freq + rf_offset
    rf_samples = np.cos(2.0 * np.pi * rf_freq * time)
    noise_std = 0.1
    rf_samples += np.random.normal(0, noise_std, size=rf_samples.shape)

    chunk_size = 5000
    i_parts, q_parts, t_parts = [], [], []
    for start in range(0, rf_samples.size, chunk_size):
        stop = min(start + chunk_size, rf_samples.size)
        chunk = rf_samples[start:stop]
        t_chunk = time[start:stop]
        mixed = rx._mix_stage(t_chunk, chunk)
        i_parts.append(np.real(mixed))
        q_parts.append(np.imag(mixed))
        t_parts.append(t_chunk)

    i_all = np.concatenate(i_parts)
    q_all = np.concatenate(q_parts)
    t_all = np.concatenate(t_parts)

    _plot_time_domain(i_all + 1j * q_all, t_all, title="Mix Time Domain", input_waveform=rf_samples)
    _plot_real_spectrum(i_all, t_all, title="Mix Spectrum", input_waveform=rf_samples)
    plt.show()


def _plot_time_domain(data: np.ndarray, time: np.ndarray, title: str, input_waveform: np.ndarray) -> None:
    """Plot time-domain waveform of a complex baseband signal."""
    if data.size == 0:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(time * 1e3, data.real, label="I")
    plt.plot(time * 1e3, data.imag, label="Q")
    plt.plot(time * 1e3, input_waveform, label="Input RF", alpha=0.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 0.1)
    plt.show(block=False)

def _plot_real_spectrum(data: np.ndarray, time: np.ndarray, title: str, input_waveform: np.ndarray) -> None:
    """Plot magnitude spectrum (frequency domain) of a complex baseband signal."""
    if data.size == 0:
        return
    spec = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(data.size, d=np.mean(np.diff(time)))
    mag_dB = 20 * np.log10(np.abs(spec) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs / 1e3, mag_dB, label="Mix")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.tight_layout()

    noise_floor_dB = np.median(mag_dB)
    peak_idx = np.where(mag_dB > noise_floor_dB + 50)[0]
    for idx in peak_idx:
        peak_freq = freqs[idx]
        peak_mag = mag_dB[idx]
        plt.annotate(
            f"Peak: {peak_freq/1e3:.1f} kHz, {peak_mag:.1f} dB",
            xy=(peak_freq / 1e3, peak_mag),
            xytext=(peak_freq / 1e3, peak_mag - 20),
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            horizontalalignment='left'
        )

    input_spec = np.fft.rfft(input_waveform)
    input_freqs = np.fft.rfftfreq(input_waveform.size, d=np.mean(np.diff(time)))
    input_mag_dB = 20 * np.log10(np.abs(input_spec) + 1e-12)
    plt.plot(input_freqs / 1e3, input_mag_dB, label="Input", alpha=0.5)
    plt.legend()

    plt.show(block=False)

def _plot_complex_spectrum(data: np.ndarray, time: np.ndarray, title: str, input_waveform: np.ndarray) -> None:
    """Plot magnitude spectrum (frequency domain) of a complex baseband signal."""
    if data.size == 0:
        return
    spec = np.fft.fft(data)
    freqs = np.fft.fftfreq(data.size, d=np.mean(np.diff(time)))
    mag_dB = 20 * np.log10(np.abs(spec) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs / 1e3, mag_dB)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.tight_layout()

    input_spec = np.fft.fft(input_waveform)
    input_freqs = np.fft.fftfreq(input_waveform.size, d=np.mean(np.diff(time)))
    input_mag_dB = 20 * np.log10(np.abs(input_spec) + 1e-12)
    plt.plot(input_freqs / 1e3, input_mag_dB, label="Input", alpha=0.5)
    plt.legend()

    plt.show(block=False)

if __name__ == "__main__":
    plot_mix_example()
    #plot_cic_example()
    
