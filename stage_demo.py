"""Quick demo for visualizing the mix stage output."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import CZT as ScipyCZT

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

    _plot_time_domain(i_all + 1j * q_all, t_all, title="Mix Time Domain", input_data=rf_samples, input_time=time)
    _plot_real_spectrum(i_all, t_all, title="Mix Spectrum", input_data=rf_samples, input_time=time)
    plt.show()

def plot_cic_example() -> None:
    """Quick demo for visualizing the CIC stage output."""
    cfg = RxConfig()
    cfg.pipeline_idx = 1
    rx = Receiver(cfg, mode="full")

    fs = 1e6
    duration = 20e-3
    time = np.arange(0.0, duration, 1.0 / fs, dtype=np.float64)

    rf_offset = 2e3
    rf_freq = cfg.carrier_freq + rf_offset
    rf_samples = np.cos(2.0 * np.pi * rf_freq * time)
    noise_std = 0.1
    rf_samples += np.random.normal(0, noise_std, size=rf_samples.shape)

    chunk_size = 5000
    mixed_parts, mixed_time, cic_parts, cic_time = [], [], [], []
    for start in range(0, rf_samples.size, chunk_size):
        stop = min(start + chunk_size, rf_samples.size)
        chunk = rf_samples[start:stop]
        t_chunk = time[start:stop]
        mixed = rx._mix_stage(t_chunk, chunk)
        mixed_parts.append(mixed)
        mixed_time.append(t_chunk)
        cic_output, cic_time_chunk = rx._cic_stage(t_chunk, mixed)
        cic_parts.append(cic_output)
        cic_time.append(cic_time_chunk)

    mixed_all = np.concatenate(mixed_parts)
    mixed_time_all = np.concatenate(mixed_time)
    cic_all = np.concatenate(cic_parts)
    cic_time_all = np.concatenate(cic_time)


    #_plot_time_domain(cic_all, cic_time_all, title="CIC Time Domain", input_data=mixed_all, input_time=mixed_time_all)
    _plot_complex_spectrum(cic_all, cic_time_all, title="CIC Spectrum", input_data=mixed_all, input_time=mixed_time_all)
    plt.show()

# --- Interactive CIC demo ---
def plot_cic_interactive() -> None:
    """Interactive CIC demo: adjust N and R via sliders and see spectra update."""
    cfg = RxConfig()
    cfg.pipeline_idx = 1
    rx = Receiver(cfg, mode="full")

    fs = 1e6

    # Use current CIC parameters as initial values
    R0 = int(rx._cic_decimation)
    N0 = int(rx._cic_stages)
    M = 1  # CIC delay

    # Normalized frequency axis for theoretical response
    freqs = np.linspace(-0.5, 0.5, 4000)

    # Create a multi-tone input to see how different bands behave
    duration = 0.02
    time = np.arange(0, duration, 1 / fs)
    tones = np.array([1e3, 5e3, 25e3, 30e3, 40e3, 45e3])
    
    x = np.zeros_like(time)
    for f in tones:
        x += np.cos(2 * np.pi * f * time)
    
    x = np.random.normal(0.0, 1.0, size=time.shape)

    # Input FFT (reference)
    X = np.fft.fftshift(np.fft.fft(x))
    fX = np.fft.fftshift(np.fft.fftfreq(x.size, 1 / fs))
    X_dB = 20 * np.log10(np.abs(X) / (np.max(np.abs(X)) + 1e-12) + 1e-12)

    # Helper to compute CIC theoretical response and output spectrum
    def compute_cic(R: int, N: int):
        # Theoretical normalized response
        H = (np.abs(np.sin(np.pi * freqs * R) / (M * np.sin(np.pi * freqs) + 1e-12))) ** N
        H_dB = 20 * np.log10(H / (np.max(H) + 1e-12) + 1e-12)

        # Apply current CIC to the multi-tone input
        rx._cic_decimation = R
        rx._cic_stages = N
        cic_out, cic_time = rx._cic_stage(time, x)
        # Output sampling rate is fs / R
        fs_out = fs / R
        Y = np.fft.fftshift(np.fft.fft(cic_out))
        fY = np.fft.fftshift(np.fft.fftfreq(cic_out.size, 1 / fs_out))
        Y_dB = 20 * np.log10(np.abs(Y) / (np.max(np.abs(Y)) + 1e-12) + 1e-12)

        return H_dB, fY, Y_dB

    # Initial compute
    H_dB0, fY0, Y_dB0 = compute_cic(R0, N0)

    # Figure and axes
    fig, (axH, axIn, axOut) = plt.subplots(3, 1, figsize=(12, 10))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.18, hspace=0.4)

    # Plot theoretical CIC response (normalized to kHz scale using current R)
    line_H, = axH.plot(freqs * fs / R0 / 1e3, H_dB0, color="magenta")
    axH.set_title("CIC Theoretical Frequency Response")
    axH.set_ylabel("20 log10 |H(f)| (dB)")

    # Plot input spectrum
    axIn.plot(fX / 1e3, X_dB, color="blue")
    axIn.set_title("Input Multi-Tone Spectrum")
    axIn.set_ylabel("dB relative")

    # Plot initial CIC output spectrum
    line_Y, = axOut.plot(fY0 / 1e3, Y_dB0, color="green")
    axOut.set_title("After CIC (Decimated Output Spectrum)")
    axOut.set_xlabel("Frequency (kHz)")
    axOut.set_ylabel("dB relative")
    # Force all spectra to show ±50 kHz range
    axH.set_xlim(-50, 50)
    axIn.set_xlim(-50, 50)
    axOut.set_xlim(-50, 50)

    # Slider axes
    axcolor = 'lightgoldenrodyellow'
    axR = fig.add_axes([0.1, 0.08, 0.8, 0.03], facecolor=axcolor)
    axN = fig.add_axes([0.1, 0.03, 0.8, 0.03], facecolor=axcolor)

    # R: decimation factor (integer)
    sR = Slider(axR, 'R (decimation)', 2, 64, valinit=R0, valstep=1)
    # N: number of stages (integer)
    sN = Slider(axN, 'N (stages)', 1, 5, valinit=N0, valstep=1)

    def update(_):
        R = int(sR.val)
        N = int(sN.val)
        rx.reset()
        H_dB, fY, Y_dB = compute_cic(R, N)

        # Update theoretical response x-axis because it depends on R
        line_H.set_xdata(freqs * fs / R / 1e3)
        line_H.set_ydata(H_dB)

        # Update output spectrum
        line_Y.set_xdata(fY / 1e3)
        line_Y.set_ydata(Y_dB)

        axH.relim(); axH.autoscale_view()
        axOut.relim(); axOut.autoscale_view()
        fig.canvas.draw_idle()
        axH.set_xlim(-50, 50)
        axOut.set_xlim(-50, 50)


    sR.on_changed(update)
    sN.on_changed(update)

    plt.show()

def plot_fir_example() -> None:
    cfg = RxConfig()
    cfg.pipeline_idx = 2
    rx = Receiver(cfg, mode="full")

    fs = 1e6
    duration = 20e-3
    time = np.arange(0.0, duration, 1.0 / fs, dtype=np.float64)

    rf_offset = 2e3
    rf_freq = cfg.carrier_freq + rf_offset
    rf_samples = np.cos(2.0 * np.pi * rf_freq * time)
    noise_std = 0.1
    rf_samples += np.random.normal(0, noise_std, size=rf_samples.shape)

    chunk_size = 5000
    mixed_parts, mixed_time, cic_parts, cic_time, fir_parts, fir_time = [], [], [], [], [], []
    for start in range(0, rf_samples.size, chunk_size):
        stop = min(start + chunk_size, rf_samples.size)
        chunk = rf_samples[start:stop]
        t_chunk = time[start:stop]
        mixed = rx._mix_stage(t_chunk, chunk)
        mixed_parts.append(mixed)
        mixed_time.append(t_chunk)
        cic_output, cic_time_chunk = rx._cic_stage(t_chunk, mixed)
        cic_parts.append(cic_output)
        cic_time.append(cic_time_chunk)
        fir_output, fir_time_chunk = rx._fir_stage(cic_time_chunk, cic_output)
        fir_parts.append(fir_output)
        fir_time.append(fir_time_chunk)


    mixed_all = np.concatenate(mixed_parts)
    mixed_time_all = np.concatenate(mixed_time)
    cic_all = np.concatenate(cic_parts)
    cic_time_all = np.concatenate(cic_time)
    fir_all = np.concatenate(fir_parts)
    fir_time_all = np.concatenate(fir_time)


    _plot_complex_spectrum(fir_all, fir_time_all, title="FIR Spectrum", input_data=cic_all, input_time=cic_time_all)
    plt.show()

def plot_fir_response() -> None:
    cfg = RxConfig()
    cfg.pipeline_idx = 2
    rx = Receiver(cfg, mode="full")

    fs = 62500
    duration = 0.02
    time = np.arange(0.0, duration, 1.0 / fs, dtype=np.float64)

    # White noise input
    x = np.random.normal(0.0, 1.0, size=time.shape)

    # --- (A) Without Decimation FIR filtering ---
    h = rx._fir_coefficients
    fir_no_dec = np.convolve(x, h, mode="same")
    fir_time_no_dec = time

    # --- (B) With Decimation FIR stage ---
    fir_dec, fir_time_dec = rx._fir_stage(time, x)

    # --- FFT for both ---
    # No-decimation
    Y_no_dec = np.fft.fftshift(np.fft.fft(fir_no_dec))
    f_no_dec = np.fft.fftshift(np.fft.fftfreq(fir_no_dec.size, 1/fs))

    # Decimated
    Y_dec = np.fft.fftshift(np.fft.fft(fir_dec))
    f_dec = np.fft.fftshift(np.fft.fftfreq(fir_dec.size, np.mean(np.diff(fir_time_dec))))

    # --- Plot ---
    fig, (axH, axA, axB) = plt.subplots(3, 1, figsize=(12, 10))

    # (0) Theoretical FIR response
    N_fft = 4096
    H = np.fft.fftshift(np.fft.fft(h, N_fft))
    fH = np.fft.fftshift(np.fft.fftfreq(N_fft, 1/fs))
    axH.plot(fH/1e3, 20*np.log10(np.abs(H)+1e-12))
    axH.set_title("Theoretical FIR Frequency Response")
    axH.set_xlim(-fs/2/1e3, fs/2/1e3)
    axH.grid(True)

    # (1) No-decimation FIR
    axA.plot(f_no_dec/1e3, 20*np.log10(np.abs(Y_no_dec)+1e-12), color="orange")
    axA.set_title("FIR Output (No Decimation)")
    axA.grid(True)
    axA.set_xlim(-fs/2/1e3, fs/2/1e3)

    # (2) Decimated FIR
    fs_out = 1.0 / np.mean(np.diff(fir_time_dec))
    axB.plot(f_dec/1e3, 20*np.log10(np.abs(Y_dec)+1e-12), color="blue")
    axB.set_title(f"FIR Output (After Decimation, fs_out={fs_out:.1f} Hz)")
    axB.grid(True)
    axB.set_xlim(-fs/2/1e3, fs/2/1e3)

    plt.tight_layout()
    plt.show()

def plot_iir_example() -> None:
    cfg = RxConfig()
    cfg.pipeline_idx = 2
    rx = Receiver(cfg, mode="full")

    fs = 1e6
    duration = 20e-3
    time = np.arange(0.0, duration, 1.0 / fs)

    rf_offset = 2e3
    rf_freq = cfg.carrier_freq + rf_offset
    rf_samples = np.cos(2 * np.pi * rf_freq * time)
    rf_samples += np.random.normal(0, 0.1, size=rf_samples.shape)

    chunk_size = 5000
    mixed_parts, mixed_time = [], []
    cic_parts, cic_time = [], []
    iir_parts, iir_time = [], []

    # === Chunk-by-chunk processing ===
    for start in range(0, rf_samples.size, chunk_size):
        stop = min(start + chunk_size, rf_samples.size)
        chunk = rf_samples[start:stop]
        t_chunk = time[start:stop]
        mixed = rx._mix_stage(t_chunk, chunk)
        mixed_parts.append(mixed)
        mixed_time.append(t_chunk)
        cic_out, cic_t = rx._cic_stage(t_chunk, mixed)
        cic_parts.append(cic_out)
        cic_time.append(cic_t)
        iir_out, iir_t = rx._iir_stage(cic_t, cic_out)
        iir_parts.append(iir_out)
        iir_time.append(iir_t)


    mixed_all = np.concatenate(mixed_parts)
    mixed_time_all = np.concatenate(mixed_time)
    cic_all = np.concatenate(cic_parts)
    cic_time_all = np.concatenate(cic_time)
    iir_all = np.concatenate(iir_parts)
    iir_time_all = np.concatenate(iir_time)

    _plot_complex_spectrum(iir_all, iir_time_all, title="IIR Stage Spectrum", input_data=cic_all, input_time=cic_time_all)
    plt.show()

def plot_iir_response() -> None:
    """
    Visualize IIR filter behaviour:
    (A) Theoretical IIR frequency response
    (B) IIR output without decimation
    (C) IIR output with decimation (actual RX stage)
    """
    cfg = RxConfig()
    cfg.pipeline_idx = 1      # IIR stage index (depends on your pipeline)
    rx = Receiver(cfg, mode="full")

    fs = 62500  # IIR input rate in RX pipeline
    duration = 0.02
    time = np.arange(0.0, duration, 1.0 / fs, dtype=np.float64)

    # White noise input → shows full frequency response
    x = np.random.normal(0.0, 1.0, size=time.shape)

    # --- (A) Theoretical IIR Frequency Response ---
    b = rx._iir_b_coefficients
    a = rx._iir_a_coefficients
    order = rx._iir_order

    N_fft = 4096
    # zero-pad numerator & denominator to same length
    b_pad = np.zeros(order + 1)
    a_pad = np.zeros(order + 1)
    b_pad[:len(b)] = b
    a_pad[:len(a)] = a

    H = np.fft.fftshift(np.fft.fft(b_pad, N_fft) / np.fft.fft(a_pad, N_fft))
    fH = np.fft.fftshift(np.fft.fftfreq(N_fft, 1/fs))

    # --- (B) IIR filtering WITHOUT decimation (manual DF2T loop) ---
    # replicate algorithm in _iir_stage
    z = np.zeros(order, dtype=np.complex128)
    y_no_dec = np.zeros_like(x, dtype=np.complex128)

    for idx, sample in enumerate(x):
        acc = b_pad[0] * sample + (z[0] if order > 0 else 0)
        y_no_dec[idx] = acc

        if order > 0:
            for s in range(order - 1):
                z[s] = z[s + 1] + b_pad[s + 1] * sample - a_pad[s + 1] * acc
            z[order - 1] = b_pad[order] * sample - a_pad[order] * acc

    # FFT (no decimation)
    Y_no_dec = np.fft.fftshift(np.fft.fft(y_no_dec))
    f_no_dec = np.fft.fftshift(np.fft.fftfreq(y_no_dec.size, 1/fs))

    # --- (C) Actual IIR stage WITH decimation ---
    rx.reset()
    y_dec, t_dec = rx._iir_stage(time, x)
    fs_dec = 1.0 / np.mean(np.diff(t_dec))

    Y_dec = np.fft.fftshift(np.fft.fft(y_dec))
    f_dec = np.fft.fftshift(np.fft.fftfreq(y_dec.size, 1/fs_dec))

    # --- Plot ---
    fig, (axH, axA, axB) = plt.subplots(3, 1, figsize=(12, 10))

    # (0) Theoretical IIR response
    axH.plot(fH/1e3, 20*np.log10(np.abs(H)+1e-12), color="purple")
    axH.set_title("Theoretical IIR Frequency Response")
    axH.set_xlim(-fs/2/1e3, fs/2/1e3)
    axH.grid(True)

    # (1) No-decimation IIR output
    axA.plot(f_no_dec/1e3, 20*np.log10(np.abs(Y_no_dec)+1e-12), color="orange")
    axA.set_title("IIR Output (No Decimation)")
    axA.grid(True)
    axA.set_xlim(-fs/2/1e3, fs/2/1e3)

    # (2) Decimated IIR output
    axB.plot(f_dec/1e3, 20*np.log10(np.abs(Y_dec)+1e-12), color="blue")
    axB.set_title(f"IIR Output (After Decimation, fs_out={fs_dec:.1f} Hz)")
    axB.grid(True)
    axB.set_xlim(-fs/2/1e3, fs/2/1e3)

    plt.tight_layout()
    plt.show()

def plot_czt_example() -> None:
    """Quick demo for visualizing the CZT output."""
    fs = 25000
    N = 2048
    duration = N/fs
    time = np.arange(0.0, duration, 1.0 / fs, dtype=np.float64)

    rf_freq = 457e3
    rf_samples = np.cos(2.0 * np.pi * rf_freq * time)
    rf_freq2 = 457e3 + 50
    rf_samples += 0.5 * np.cos(2.0 * np.pi * rf_freq2 * time)
    noise_std = 0.1
    rf_samples += np.random.normal(0, noise_std, size=rf_samples.shape)
    
    nco = np.exp(-1j * 2 * np.pi * 457e3 * time)
    bb_samples = rf_samples * nco

    # Define zoom band
    f_start = -100
    f_end   = 100
    M = 128

    # Compute A and W for the CZT
    W = np.exp(-1j * 2 * np.pi * (f_end - f_start) / (M * fs)) # the ratio between points in each step
    A = np.exp(1j * 2 * np.pi * f_start / fs) # the starting point

    # SciPy CZT
    transformer = ScipyCZT(n=len(bb_samples), m=M, w=W, a=A)
    spec = transformer(bb_samples)
    #spec = _czt_fft(rf_samples, m=M, w=W, a=A) # Custom CZT implementation
    freqs = np.linspace(f_start, f_end, M)

    # --- FFT (reference) ---
    X_fft = np.fft.fftshift(np.fft.fft(bb_samples, N))
    freqs_fft = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

    # FFT magnitude
    mag_fft = 20 * np.log10(np.abs(X_fft) + 1e-12)

    # CZT band mask
    fft_band_mask = (freqs_fft >= f_start) & (freqs_fft <= f_end)

    freqs_fft_zoom = freqs_fft[fft_band_mask]
    mag_fft_zoom = mag_fft[fft_band_mask]

    plt.figure(figsize=(10, 4))
    plt.plot(freqs / 1e3, 20 * np.log10(np.abs(spec) + 1e-12),
         label="CZT", linewidth=2)
    plt.plot(freqs_fft_zoom / 1e3, mag_fft_zoom,
         label="FFT (zoomed)", linestyle="--", linewidth=1)
    plt.title("CZT (SciPy) Zoom Spectrum")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _czt_fft(x, m=None, w=None, a=1.0 + 0j):
    """Compute the Chirp Z-Transform (CZT) of a 1D array x.
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]

    if m is None:
        m = n
    if w is None:
        w = np.exp(-1j * 2.0 * np.pi / m)

    # Indices as float for exponent arithmetic
    n_idx = np.arange(n, dtype=np.float64)
    m_idx = np.arange(m, dtype=np.float64)

    # Step 1: Pre-multiply input by chirps
    # y[n] = x[n] * a^{-n} * w^{n^2 / 2}
    y = x * (a ** (-n_idx)) * (w ** (n_idx * n_idx / 2.0))

    # Step 2: Build convolution kernel v[k] = w^{-k^2 / 2}
    # and wrap it so that linear convolution is implemented via FFT.
    L = int(2 ** np.ceil(np.log2(n + m - 1)))
    v = np.zeros(L, dtype=np.complex128)

    # First part: k = 0 .. m-1
    v[:m] = w ** (-m_idx * m_idx / 2.0)

    # Tail part: k = -(n-1) .. -1 mapped to indices L-(n-1) .. L-1
    if n > 1:
        tail_idx = np.arange(1, n, dtype=np.float64)
        v[L - (n - 1):] = w ** (-tail_idx[::-1] * tail_idx[::-1] / 2.0)

    # Step 3: FFT-based convolution
    Y = np.fft.fft(y, L)
    V = np.fft.fft(v, L)
    G = Y * V
    g = np.fft.ifft(G, L)

    # Step 4: Take first m outputs and apply final chirp
    k_idx = np.arange(m, dtype=np.float64)
    X = g[:m] * (w ** (k_idx * k_idx / 2.0))

    return X


def _plot_time_domain(data: np.ndarray, data_time: np.ndarray, title: str, input_data: np.ndarray, input_time: np.ndarray) -> None:
    """Plot time-domain waveform of a complex baseband signal."""
    if data.size == 0:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(data_time * 1e3, data.real, label="I")
    plt.plot(data_time * 1e3, data.imag, label="Q")
    plt.plot(input_time * 1e3, input_data, label="Input RF", alpha=0.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 1)
    plt.show(block=False)

def _plot_real_spectrum(data: np.ndarray, data_time: np.ndarray, title: str, input_data: np.ndarray, input_time: np.ndarray) -> None:
    """Plot magnitude spectrum (frequency domain) of a complex baseband signal."""
    if data.size == 0:
        return
    spec = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(data.size, d=np.mean(np.diff(data_time)))
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

    input_spec = np.fft.rfft(input_data)
    input_freqs = np.fft.rfftfreq(input_data.size, d=np.mean(np.diff(input_time)))
    input_mag_dB = 20 * np.log10(np.abs(input_spec) + 1e-12)
    plt.plot(input_freqs / 1e3, input_mag_dB, label="Input", alpha=0.5)
    plt.legend()

    plt.show(block=False)

def _plot_complex_spectrum(data: np.ndarray, data_time: np.ndarray, title: str, input_data: np.ndarray, input_time: np.ndarray) -> None:
    """Plot magnitude spectrum (frequency domain) of a complex baseband signal."""
    if data.size == 0:
        return
    spec = np.fft.fftshift(np.fft.fft(data))
    freqs = np.fft.fftshift(np.fft.fftfreq(data.size, d=np.mean(np.diff(data_time))))
    mag_dB = 20 * np.log10(np.abs(spec) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs / 1e3, mag_dB)
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

    input_spec = np.fft.fftshift(np.fft.fft(input_data))
    input_freqs = np.fft.fftshift(np.fft.fftfreq(input_data.size, d=np.mean(np.diff(input_time))))
    input_mag_dB = 20 * np.log10(np.abs(input_spec) + 1e-12)
    plt.plot(input_freqs / 1e3, input_mag_dB, label="Input", alpha=0.5)
    plt.legend()

    plt.show(block=False)



if __name__ == "__main__":
    #plot_mix_example()
    #plot_cic_example()
    #plot_cic_interactive()
    #plot_fir_example()
    #plot_fir_response()
    #plot_iir_example()
    #plot_iir_response()
    plot_czt_example()