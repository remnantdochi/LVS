import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import czt

# ---- 설정 (STM32 코드와 동일하게 맞춰주세요) ----
FS = 25000.0
CZT_N = 1024  # ADC 버퍼 크기 (또는 처리 크기)
CZT_M = 256   # 출력 Bin 개수

f_center = 457000.0
span     = 200.0
f_start  = f_center - span / 2
f_end    = f_center + span / 2

# 파일 이름 (TeraTerm 등으로 저장한 로그 파일)
LOG_FILE = "input_11_01.txt"

def parse_frames(filename):
    """
    로그 파일을 읽어서 프레임 별로 (adc_data, mcu_czt_data) 딕셔너리 리스트를 반환
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
                # 1. ADC 데이터 파싱 (숫자만 있는 줄)
                #    혹은 "X[k] =" 패턴이 없는 줄
                if "X[" not in line and "TIME_US" not in line:
                    try:
                        val = float(line)
                        current_adc.append(val)
                    except ValueError:
                        pass # 헤더 등 무시
                
                # 2. CZT 데이터 파싱 ("X[k] = real imag")
                elif "X[" in line:
                    # format: X[0] = 123.4 567.8
                    parts = line.replace('=', ' ').replace('[', ' ').replace(']', ' ').split()
                    # parts 예시: ['X', '0', '123.4', '567.8']
                    if len(parts) >= 4:
                        try:
                            re = float(parts[2])
                            im = float(parts[3])
                            current_mcu_czt.append(np.hypot(re, im)) # Magnitude 계산해서 저장
                        except ValueError:
                            pass
    return frames

def process_frame(idx, frame_data):
    adc_raw = frame_data["adc"]
    czt_mag_mcu = frame_data["mcu_czt"]

    # 데이터 개수 검증
    if len(adc_raw) < CZT_N:
        print(f"[Frame {idx}] Warning: Not enough ADC samples ({len(adc_raw)}/{CZT_N}). Using Zero padding.")
        adc_raw = np.pad(adc_raw, (0, CZT_N - len(adc_raw)))
    else:
        adc_raw = adc_raw[:CZT_N]

    # DC Offset 제거 (MCU에서도 함)
    adc_raw -= np.mean(adc_raw)

    # --- Python(SciPy) CZT 계산 ---
    W = np.exp(-1j * 2 * np.pi * (f_end - f_start) / (CZT_M * FS))
    A = np.exp( 1j * 2 * np.pi * f_start / FS)
    
    czt_py = czt(adc_raw, m=CZT_M, w=W, a=A)
    czt_mag_py = np.abs(czt_py)

    # --- 비교 및 출력 ---
    freqs = np.linspace(f_start, f_end, CZT_M, endpoint=False)
    
    diff = czt_mag_mcu - czt_mag_py
    max_err = np.max(np.abs(diff))
    mean_err = np.mean(np.abs(diff))

    # Peak 찾기
    peak_bin_mcu = np.argmax(czt_mag_mcu)
    peak_freq_mcu = freqs[peak_bin_mcu]
    
    peak_bin_py = np.argmax(czt_mag_py)
    peak_freq_py = freqs[peak_bin_py]

    print(f"\n>>> FRAME {idx} REPORT <<<")
    print(f"Max Abs Error : {max_err:.6f}")
    print(f"MCU Peak      : {peak_freq_mcu:.2f} Hz (Bin {peak_bin_mcu})")
    print(f"SciPy Peak    : {peak_freq_py:.2f} Hz (Bin {peak_bin_py})")
    
    # --- 그래프 그리기 ---
    plt.figure(figsize=(10, 4))
    
    # 1. 파형 비교
    plt.subplot(1, 2, 1)
    plt.plot(freqs, czt_mag_mcu, label="MCU", linewidth=1.5)
    plt.plot(freqs, czt_mag_py, "--", label="SciPy", linewidth=1.5)
    plt.title(f"Frame {idx}: Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    # 2. ADC 입력 파형 (연속성 확인용)
    plt.subplot(1, 2, 2)
    plt.plot(adc_raw)
    plt.title(f"Frame {idx}: ADC Input (Time Domain)")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ==== 메인 실행 ====
frames = parse_frames(LOG_FILE)
print(f"Total Frames Parsed: {len(frames)}")

if len(frames) == 0:
    print("No frames found! Check the log file format.")
else:
    for i, frame in enumerate(frames):
        process_frame(i, frame)