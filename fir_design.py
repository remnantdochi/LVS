from __future__ import annotations
import numpy as np
from scipy.signal import firwin
from scipy.signal import iirfilter
from config import RxConfig

def design_fir_filter(
    num_taps: int = RxConfig.fir_taps,
    cutoff_hz: float = RxConfig.fir_cutoff_hz,
    sample_rate_hz: float = RxConfig.fir_sample_rate_hz,
    window: str = "hann",
) -> np.ndarray:

    return firwin(
        numtaps=num_taps,
        cutoff=cutoff_hz,
        window=window,
        pass_zero="lowpass",
        fs=sample_rate_hz,
    )

def design_iir_filter(
    order: int = RxConfig.iir_order,
    cutoff_hz: float = RxConfig.iir_cutoff_hz,
    sample_rate_hz: float = RxConfig.iir_sample_rate_hz,
    ftype: str = "butter",
) -> tuple[np.ndarray, np.ndarray]:
        # Normalize cutoff (0~1)
    normalized_cutoff = cutoff_hz / (sample_rate_hz / 2.0)

    # butterworth lowpass IIR filter
    b, a = iirfilter(
        N=order,
        Wn=normalized_cutoff,
        btype="low",
        ftype=ftype,
        output="ba"
    )

    # Normalize by a[0] just like the Receiver code
    b = b / a[0]
    a = a / a[0]

    return b, a

if __name__ == "__main__":
    taps = design_fir_filter()
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    #print(f"FIR taps (len={len(taps)}):")
    #print(taps)
    with open("fir_tap.txt", "w") as f:
        f.write("(\n")
        for t in taps:
            f.write(f"    {t:.10f},\n")
        f.write(")\n")

    with open("iir_tap.txt", "w") as f:
        b, a = design_iir_filter()
        f.write("Numerator coefficients (b):\n")
        f.write("(\n")
        for coeff in b:
            f.write(f"    {coeff:.10f},\n")
        f.write(")\n\n")
        f.write("Denominator coefficients (a):\n")
        f.write("(\n")
        for coeff in a:
            f.write(f"    {coeff:.10f},\n")
        f.write(")\n")
    print("Saved FIR taps to fir_tap.txt")
    print("Saved IIR coefficients to iir_tap.txt")
