import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import czt

import numpy as np
import matplotlib.pyplot as plt

def plot_stage_compare(stage_name, mcu, py, use_ratio=False):
    """
    mcu, py: complex or real numpy arrays
    """

    mcu = np.asarray(mcu)
    py  = np.asarray(py)

    mag_mcu = np.abs(mcu)
    mag_py  = np.abs(py)

    min_len = min(len(mag_mcu), len(mag_py))
    mag_mcu = mag_mcu[:min_len]
    mag_py  = mag_py[:min_len]

    diff = np.abs(mag_mcu - mag_py)

    mean_diff = np.mean(diff)

    if use_ratio:
        ratio = np.mean(mag_mcu) / np.mean(mag_py)
        text = f"ratio = {ratio:.6f}"
    else:
        text = f"mean |diff| = {mean_diff:.6e}"

    plt.figure(figsize=(10, 4))
    plt.plot(mag_mcu, label="MCU", linewidth=1.2)
    plt.plot(mag_py,  label="Python", linestyle="--", linewidth=1.2)

    plt.title(f"{stage_name} comparison")
    plt.xlabel("index")
    plt.ylabel("magnitude")
    plt.legend()
    plt.grid(True)

    plt.text(
        0.02, 0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.2)
    )

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

# === 디버그용 단계별 CZT 함수 ===
def _czt_fft_debug(x, m, w, a):
    """
    main.c의 stage 출력과 1:1 비교를 위한
    CZT 단계별 결과를 모두 반환
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]

    n_idx = np.arange(n, dtype=np.float64)
    m_idx = np.arange(m, dtype=np.float64)

    # ---------- Stage 1 ----------
    y = x * (a ** (-n_idx)) * (w ** (n_idx * n_idx / 2.0))

    # ---------- Stage 2 ----------
    L = int(2 ** np.ceil(np.log2(n + m - 1)))
    v = np.zeros(L, dtype=np.complex128)

    v[:m] = w ** (-m_idx * m_idx / 2.0)

    if n > 1:
        tail_idx = np.arange(1, n, dtype=np.float64)
        v[L - (n - 1):] = w ** (-tail_idx[::-1] * tail_idx[::-1] / 2.0)

    # ---------- Stage 3 ----------
    Y = np.fft.fft(y, L)
    V = np.fft.fft(v, L)
    G = Y * V
    g = np.fft.ifft(G, L)

    # ---------- Stage 4 ----------
    k_idx = np.arange(m, dtype=np.float64)
    X = g[:m] * (w ** (k_idx * k_idx / 2.0))

    return {
        "y": y,
        "v": v,
        "Y": Y,
        "V": V,
        "G": G,
        "g": g,
        "X": X,
        "L": L,
    }

def reconstruct_from_czt(
    adc_raw,
    X_py,
    X_mcu,
    Fs,
    f_start,
    df,
    n_plot=300
):
    """
    adc_raw : mean-removed ADC input (float)
    X_py    : Python CZT result (complex)
    X_mcu   : MCU CZT result (complex)
    Fs      : sampling frequency
    f_start : CZT start frequency
    df      : CZT bin spacing
    """

    # ----------------------------
    # 1. Peak bin 선택 (Python 기준)
    # ----------------------------
    mag = np.abs(X_py)
    k0 = np.argmax(mag)

    f0 = f_start + k0 * df

    print("=== Dominant CZT bin ===")
    print(f"bin index = {k0}")
    print(f"frequency ≈ {f0:.2f} Hz")

    # ----------------------------
    # 2. 시간축
    # ----------------------------
    N = len(adc_raw)
    n = np.arange(N)

    # ----------------------------
    # 3. Python CZT 기반 재구성
    # ----------------------------
    C_py = X_py[k0]
    amp_py = np.abs(C_py) / N
    phase_py = np.angle(C_py)

    x_recon_py = amp_py * np.cos(
        2 * np.pi * f0 * n / Fs + phase_py
    )

    # ----------------------------
    # 4. MCU CZT 기반 재구성
    # ----------------------------
    C_mcu = X_mcu[k0]
    amp_mcu = np.abs(C_mcu) / N
    phase_mcu = np.angle(C_mcu)

    x_recon_mcu = amp_mcu * np.cos(
        2 * np.pi * f0 * n / Fs + phase_mcu
    )

    # ----------------------------
    # 5. 비교 plot
    # ----------------------------
    plt.figure(figsize=(11, 5))

    plt.plot(
        adc_raw[:n_plot],
        label="Original ADC (mean removed)",
        alpha=0.5
    )

    plt.plot(
        x_recon_py[:n_plot],
        label="Reconstructed (Python CZT)",
        linewidth=2
    )

    plt.plot(
        x_recon_mcu[:n_plot],
        label="Reconstructed (MCU CZT)",
        linestyle="--",
        linewidth=2
    )

    plt.title(
        f"Time-domain reconstruction from CZT peak\n"
        f"f ≈ {f0:.2f} Hz (bin {k0})"
    )
    plt.xlabel("sample index")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 6. 정량 비교 출력
    # ----------------------------
    diff_py_mcu = np.abs(x_recon_py - x_recon_mcu)

    print("=== Reconstruction difference ===")
    print(f"mean |py - mcu| = {np.mean(diff_py_mcu):.6e}")
    print(f"max  |py - mcu| = {np.max(diff_py_mcu):.6e}")

def plot_czt_freq_compare(X_mcu, X_py, f_start, df, title="CZT Comparison (MCU vs SciPy)"):
    """
    X_mcu : MCU CZT 결과 (complex)
    X_py  : Python CZT 결과 (complex)
    f_start : CZT 시작 주파수 (Hz)
    df      : CZT bin spacing (Hz)
    """

    X_mcu = np.asarray(X_mcu)
    X_py  = np.asarray(X_py)

    M = min(len(X_mcu), len(X_py))

    freqs = f_start + np.arange(M) * df

    mag_mcu = np.abs(X_mcu[:M])
    mag_py  = np.abs(X_py[:M])

    plt.figure(figsize=(10, 4.5))

    plt.plot(freqs, mag_mcu, label="MCU CZT", linewidth=2)
    plt.plot(freqs, mag_py,  "--", label="SciPy CZT", linewidth=2)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

FS = 1000000.0
CZT_N = 1024

f_center = 457000.0
span     = 200.0
f_start  = f_center - span / 2
f_end    = f_center + span / 2
df = span / CZT_N

# === MCU UART 로그 파싱 함수 ===
import re
import numpy as np

float_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')

def parse_mcu_stage_log(lines):
    stages = {
        "W": None,
        "A": None,
        "adc": [],
        "x": [],
        "y": [],
        "v": [],
        "Y": [],
        "V": [],
        "G": [],
        "g": [],
        "X": [],
    }

    current = None

    for line in lines:
        line = line.strip()
        # ---------- stage0: W, A ----------
        if line.startswith("W ="):
            nums = float_re.findall(line)
            if len(nums) == 2:
                stages["W"] = complex(float(nums[0]), float(nums[1]))
            continue

        if line.startswith("A ="):
            nums = float_re.findall(line)
            if len(nums) == 2:
                stages["A"] = complex(float(nums[0]), float(nums[1]))
            continue

        # ---------- stage markers ----------
        if "ADC" in line:
            current = "adc"
            continue
        if "stage0" in line:
            current = "x"
            continue
        if "stage1" in line:
            current = "y"
            continue
        if "stage2" in line:
            current = "v"
            continue
        if "FFT(Y), FFT(V)" in line:
            current = "YV"
            continue
        if "G = Y*V" in line:
            current = "G"
            continue
        if "IFFT(G)" in line:
            current = "g"
            continue
        if "stage4" in line:
            current = "X"
            continue

        # ---------- data lines ----------
        nums = float_re.findall(line)
        
        if current == "y" and line.startswith("y[") and len(nums) == 2:
            stages["y"].append(complex(float(nums[0]), float(nums[1])))

        elif current == "v" and line.startswith("v[") and len(nums) == 2:
            stages["v"].append(complex(float(nums[0]), float(nums[1])))

        elif current == "YV" and line.startswith("Y[") and len(nums) == 4:
            Yr, Yi, Vr, Vi = map(float, nums)
            stages["Y"].append(complex(Yr, Yi))
            stages["V"].append(complex(Vr, Vi))

        elif current == "G" and line.startswith("G[") and len(nums) == 2:
            stages["G"].append(complex(float(nums[0]), float(nums[1])))

        elif current == "g" and line.startswith("g[") and len(nums) == 2:
            stages["g"].append(complex(float(nums[0]), float(nums[1])))

        elif current == "adc" and line.startswith("adc["):
            m = re.search(r"adc\[\d+\]\s*=\s*(-?\d+)", line)
            if m:
                stages["adc"].append(float(m.group(1)))

        elif current == "x" and line.startswith("x[") and len(nums) == 1:
            stages["x"].append(float(nums[0]))

        elif current == "X" and line.startswith("X[") and len(nums) == 2:
            stages["X"].append(complex(float(nums[0]), float(nums[1])))

    # NumPy 배열로 변환

    for k in ["y", "v", "Y", "V", "G", "g", "X"]:
        stages[k] = np.array(stages[k], dtype=np.complex128)
    # x[n]은 float64
    stages["x"] = np.array(stages["x"], dtype=np.float64)
    stages["adc"] = np.array(stages["adc"], dtype=np.float64)

    return stages

# === 단계별 비교 메인 코드 ===
with open("input_08_04.txt", "r") as f:
    lines = [l.strip() for l in f if l.strip()]



# CZT 파라미터
M = 128  # MCU의 CZT_M
W = np.exp(-1j * 2 * np.pi * (f_end - f_start) / (M * FS))
A = np.exp( 1j * 2 * np.pi * f_start / FS)



# MCU 로그 파싱
mcu = parse_mcu_stage_log(lines)
adc_raw = mcu["adc"]
# Python 단계별 계산
adc_raw -= np.mean(adc_raw)  # DC 오프셋 제거
dbg = _czt_fft_debug(adc_raw, m=M, w=W, a=A)




print("W (Python) =", W)
print("A (Python) =", A)

print("=== Input parameter compare (W, A) ===")
if mcu["W"] is not None and mcu["A"] is not None:
    print("W diff =", abs(mcu["W"] - W))
    print("A diff =", abs(mcu["A"] - A))
else:
    print("MCU W/A not found in log")

print("=== Stage 0 x[n] 비교 ===")
print("max |diff| =", np.max(np.abs(mcu["adc"] - mcu["x"])))

print("=== Stage 1 y[n] 비교 ===")
print("max |diff| =", np.max(np.abs(dbg["y"][:len(mcu["y"])] - mcu["y"])))

print("=== Stage 2 v[k] 비교 ===")
print("max |diff| =", np.max(np.abs(dbg["v"][:len(mcu["v"])] - mcu["v"])))

print("=== Stage 3 Y 비교 ===")
print("ratio ~", np.mean(np.abs(mcu["Y"]) / np.abs(dbg["Y"][:len(mcu["Y"])])))

print("=== Stage 3 G 비교 ===")
print("ratio ~", np.mean(np.abs(mcu["G"]) / np.abs(dbg["G"][:len(mcu["G"])])))

print("=== Stage 3 g 비교 ===")
print("ratio ~", np.mean(np.abs(mcu["g"]) / np.abs(dbg["g"][:len(mcu["g"])])))

# === Stage 4 비교 ===
# MCU g[k]에 Python과 동일한 final chirp 적용
k_idx = np.arange(len(mcu["g"]), dtype=np.float64)
#mcu_X = mcu["g"][:M] * (W ** (k_idx[:M] * k_idx[:M] / 2.0))
mcu_X = mcu["X"][:len(dbg["X"])]

print("=== Stage 4 X[k] 비교 ===")
print("ratio ~", np.mean(np.abs(mcu_X) / np.abs(dbg["X"])))

plot_stage_compare(
    "Stage 1: y[n]",
    mcu["y"],
    dbg["y"],
    use_ratio=False
)

plot_stage_compare(
    "Stage 2: v[k]",
    mcu["v"],
    dbg["v"],
    use_ratio=False
)

plot_stage_compare(
    "Stage 3: Y[k]",
    mcu["Y"],
    dbg["Y"],
    use_ratio=False
)

plot_stage_compare(
    "Stage 3: V[k]",
    mcu["V"],
    dbg["V"],
    use_ratio=False
)

plot_stage_compare(
    "Stage 3: G[k]",
    mcu["G"],
    dbg["G"],
    use_ratio=False
)

plot_stage_compare(
    "Stage 4: X[k]",
    mcu_X,
    dbg["X"],
    use_ratio=False 
)

reconstruct_from_czt(
    adc_raw=adc_raw,
    X_py=dbg["X"],
    X_mcu=mcu["X"],
    Fs=FS,
    f_start=f_start,
    df=df,
    n_plot=300
)
