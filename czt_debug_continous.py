import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import czt

FS = 25000.0
CZT_N = 1024
CZT_M = 256

f_center = 457000.0
span     = 200.0
f_start  = f_center - span / 2
f_end    = f_center + span / 2

LOG_FILE = "input.txt"

def parse_frames(filename):
    """
    Read log file and return a list of {adc_data, mcu_czt_data} dictionaries per frame
    """
    frames = []
    current_adc = []
    current_mcu_czt = []
    in_frame = False
    
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue

            if "=== FRAME" in line and "START ===" in line:
                in_frame = True
                current_adc = []
                current_mcu_czt = []
                continue
            
            if "=== FRAME" in line and "END ===" in line:
                in_frame = False
                if len(current_adc) > 0:
                    frames.append({
                        "adc": np.array(current_adc, dtype=np.float64),
                        "mcu_czt": np.array(current_mcu_czt, dtype=np.float64)
                    })
                continue

            if in_frame:
                # 1. Parse ADC data (lines with only numbers)
                #    or lines without "X[k] =" pattern
                if "X[" not in line and "TIME_US" not in line:
                    try:
                        val = float(line)
                        current_adc.append(val)
                    except ValueError:
                        pass  # Ignore headers, etc.
                
                # 2. Parse CZT data ("X[k] = real imag")
                elif "X[" in line:
                    # format: X[0] = 123.4 567.8
                    parts = line.replace('=', ' ').replace('[', ' ').replace(']', ' ').split()
                    # parts example: ['X', '0', '123.4', '567.8']
                    if len(parts) >= 4:
                        try:
                            re = float(parts[2])
                            im = float(parts[3])
                            current_mcu_czt.append(np.hypot(re, im))  # Calculate and store magnitude
                        except ValueError:
                            pass
    return frames

def process_frame(idx, frame_data):
    adc_raw = frame_data["adc"]
    czt_mag_mcu = frame_data["mcu_czt"]

    # Validate data count
    if len(adc_raw) < CZT_N:
        print(f"[Frame {idx}] Warning: Not enough ADC samples ({len(adc_raw)}/{CZT_N}). Using zero padding.")
        adc_raw = np.pad(adc_raw, (0, CZT_N - len(adc_raw)))
    else:
        adc_raw = adc_raw[:CZT_N]

    # Remove DC offset (same as MCU)
    adc_raw -= np.mean(adc_raw)

    # --- Calculate CZT using Python (SciPy) ---
    W = np.exp(-1j * 2 * np.pi * (f_end - f_start) / (CZT_M * FS))
    A = np.exp( 1j * 2 * np.pi * f_start / FS)
    
    czt_py = czt(adc_raw, m=CZT_M, w=W, a=A)
    czt_mag_py = np.abs(czt_py)

    # --- Compare and print results ---
    freqs = np.linspace(f_start, f_end, CZT_M, endpoint=False)
    
    diff = czt_mag_mcu - czt_mag_py
    max_err = np.max(np.abs(diff))
    mean_err = np.mean(np.abs(diff))

    # Find peaks
    peak_bin_mcu = np.argmax(czt_mag_mcu)
    peak_freq_mcu = freqs[peak_bin_mcu]
    
    peak_bin_py = np.argmax(czt_mag_py)
    peak_freq_py = freqs[peak_bin_py]

    print(f"\n>>> FRAME {idx} REPORT <<<")
    print(f"Max Abs Error : {max_err:.6f}")
    print(f"MCU Peak      : {peak_freq_mcu:.2f} Hz (Bin {peak_bin_mcu})")
    print(f"SciPy Peak    : {peak_freq_py:.2f} Hz (Bin {peak_bin_py})")
    
    # --- Plot graphs ---
    plt.figure(figsize=(10, 4))
    
    # 1. Waveform comparison
    plt.subplot(1, 2, 1)
    plt.plot(freqs, czt_mag_mcu, label="MCU", linewidth=1.5)
    plt.plot(freqs, czt_mag_py, "--", label="SciPy", linewidth=1.5)
    plt.title(f"Frame {idx}: Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    # 2. ADC input waveform (for continuity verification)
    plt.subplot(1, 2, 2)
    plt.plot(adc_raw)
    plt.title(f"Frame {idx}: ADC Input (Time Domain)")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ==== Main execution ====
frames = parse_frames(LOG_FILE)
print(f"Total Frames Parsed: {len(frames)}")

if len(frames) == 0:
    print("No frames found! Check the log file format.")
else:
    for i, frame in enumerate(frames):
        process_frame(i, frame)