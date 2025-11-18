from __future__ import annotations
import numpy as np
from scipy.signal import firwin

def design_lowpass(
    num_taps: int = 127,
    cutoff_hz: float = 20e3,
    sample_rate_hz: float = 62500.0,
    window: str = "hann",
) -> np.ndarray:

    return firwin(
        numtaps=num_taps,
        cutoff=cutoff_hz,
        window=window,
        pass_zero="lowpass",
        fs=sample_rate_hz,
    )

if __name__ == "__main__":
    taps = design_lowpass()
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    #print(f"FIR taps (len={len(taps)}):")
    #print(taps)
    with open("fir_tap.txt", "w") as f:
        f.write("(\n")
        for t in taps:
            f.write(f"    {t:.10f},\n")
        f.write(")\n")
    print("Saved FIR taps to fir_tap.txt")
