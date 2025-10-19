# main.py
import numpy as np
import matplotlib.pyplot as plt
from lvs_tx import Transmitter
from lvs_rx import Receiver

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power = np.mean(signal**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise

def run_sim(duration=1.0, fs=5e6, snr_db=-10):
    tx = Transmitter()
    # TBD - add several TXs with different params
    #rx = Receiver()
    t, tx_signal = tx.generate(duration, fs)
    channel = add_awgn(tx_signal, snr_db)
    #_, detection = rx.detect(t, channel, fs)
    return {
        "time": t,
        "tx_signal": tx_signal,
        "rx_signal": channel,
        #"detection": detection,
    }

if __name__ == "__main__":
    results = run_sim()
    t = results["time"]
    tx_signal = results["tx_signal"]

    plt.figure(figsize=(12, 8))
    plt.plot(t * 1e3, tx_signal, label="Transmitted Signal", alpha=0.7)
    plt.title("Transmitted Signal")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim(0, 500)  # Show first 500 ms
    plt.tight_layout()
    plt.show()

    #print(f"Detected signal power peak: {results['detection'].max():.3f}")
