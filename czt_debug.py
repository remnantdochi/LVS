import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import czt
from stage_demo import _czt_fft


FS = 25000.0
CZT_N = 1024
CZT_M = 256

f_center = 457000.0
span     = 200.0
f_start  = f_center - span / 2
f_end    = f_center + span / 2

with open("input.txt", "r") as f:
    lines = [l.strip() for l in f if l.strip()]

adc_raw = np.array(
    [float(v) for v in lines[:CZT_N]],
    dtype=np.float64
)

czt_mag_mcu = []
for l in lines[CZT_N:CZT_N + CZT_M]:
    # Expected format: X[k] = <real> <imag>
    parts = l.replace('=', ' ').split()
    if len(parts) < 3:
        continue
    try:
        re = float(parts[1])
        im = float(parts[2])
    except ValueError:
        continue
    czt_mag_mcu.append(np.hypot(re, im))

czt_mag_mcu = np.array(czt_mag_mcu, dtype=np.float64)


M = CZT_M

print(f"ADC samples : {len(adc_raw)}")
print(f"CZT bins    : {M}")

# Remove DC offset
adc_raw -= np.mean(adc_raw)


# CZT parameters
W = np.exp(
    -1j * 2 * np.pi * (f_end - f_start) / (M * FS)
)
A = np.exp(
     1j * 2 * np.pi * f_start / FS
)

czt_py = czt(adc_raw, m=M, w=W, a=A)
czt_mag_py = np.abs(czt_py)
czt_custom = _czt_fft(adc_raw, m=M, w=W, a=A)
czt_mag_custom = np.abs(czt_custom)

freqs = np.linspace(f_start, f_end, M, endpoint=False)


plt.figure(figsize=(10, 5))
plt.plot(freqs, czt_mag_mcu, label="MCU CZT", linewidth=2)
plt.plot(freqs, czt_mag_py, "--", label="SciPy CZT", linewidth=2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("CZT Comparison (MCU vs SciPy)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

diff = czt_mag_mcu - czt_mag_py
print("Max abs error :", np.max(np.abs(diff)))
print("Mean abs error:", np.mean(np.abs(diff)))

# Magnitude values from MCU (in bin order)
mag_mcu = np.array(czt_mag_mcu, dtype=np.float64)

# Find peak bin
peak_bin_mcu = np.argmax(mag_mcu)
peak_bin_py  = np.argmax(czt_mag_py)

# Convert bin to frequency
freq_res = (f_end - f_start) / CZT_M

peak_freq_mcu = f_start + peak_bin_mcu * freq_res
peak_freq_py  = f_start + peak_bin_py  * freq_res

print("=== Peak comparison ===")
print(f"MCU  : bin={peak_bin_mcu:3d}, freq={peak_freq_mcu:10.2f} Hz, mag={mag_mcu[peak_bin_mcu]:.3f}")
print(f"SciPy: bin={peak_bin_py:3d}, freq={peak_freq_py:10.2f} Hz, mag={czt_mag_py[peak_bin_py]:.3f}")
print(f"Δbin = {peak_bin_mcu - peak_bin_py}")
print(f"Δfreq = {peak_freq_mcu - peak_freq_py:.2f} Hz")

plt.figure(figsize=(8, 5))

plt.hist(
    czt_mag_mcu,
    bins=50,
    alpha=0.6,
    label="MCU CZT",
    density=True
)

plt.hist(
    czt_mag_py,
    bins=50,
    alpha=0.6,
    label="SciPy CZT",
    density=True
)

plt.xlabel("Magnitude")
plt.ylabel("Probability density")
plt.title("CZT Magnitude Histogram Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()