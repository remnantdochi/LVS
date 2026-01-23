import struct
import serial
import time
import numpy as np
import matplotlib.pyplot as plt

# ===== User settings =====
PORT = "/dev/cu.usbmodem11202"          # e.g. "COM7" on Windows or "/dev/tty.usbmodemXXXX" on macOS
BAUD = 2000000
TIMEOUT = 0.2
REPORT_EVERY = 50   # print stats every N frames

# Your firmware constants (must match MCU)
CZT_M = 256
CENTER_FREQ_HZ = 457_000.0
SPAN_HZ = 200.0

# ===== Frame format =====
MAGIC = 0x3053564C  # 'LVS0' little-endian
HEADER_FMT = "<5I"  # magic, seq, time, dropADC, dropFrame
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 20
PAYLOAD_FMT = f"<{CZT_M}f"
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FMT)  # 1024
FRAME_SIZE = HEADER_SIZE + PAYLOAD_SIZE      # 1044

MAGIC_BYTES = b"LVS0"  # appears in the byte stream
MAGIC_LEN = 4


def find_magic_and_align(ser: serial.Serial, buf: bytearray) -> None:
    """
    Ensure buf starts with MAGIC_BYTES.
    Reads from serial until MAGIC is found; discards preceding bytes.
    """
    while True:
        idx = buf.find(MAGIC_BYTES)
        if idx == 0:
            return
        if idx > 0:
            del buf[:idx]
            return

        # not found: keep last few bytes in case magic is split across reads
        keep = MAGIC_LEN - 1
        if len(buf) > keep:
            buf[:] = buf[-keep:]

        chunk = ser.read(4096)
        if chunk:
            buf.extend(chunk)
        else:
            time.sleep(0.01)


def read_exact(ser: serial.Serial, buf: bytearray, n: int):
    """Ensure `buf` contains at least `n` bytes by reading from serial. Returns (data_bytes, bytes_read_from_serial)."""
    read_from_serial = 0
    while len(buf) < n:
        need = n - len(buf)
        n_avail = ser.in_waiting
        if n_avail > 0:
            chunk = ser.read(min(need, n_avail))
            if chunk:
                buf.extend(chunk)
                read_from_serial += len(chunk)
        else:
            time.sleep(0.001)
    out = bytes(buf[:n])
    del buf[:n]
    return out, read_from_serial


def read_frame(ser: serial.Serial, buf: bytearray):
    """
    Returns:
      (seq, czt_time_ms, dropADC, dropFrame, mag, frame_start_s)
    """
    t0 = time.perf_counter()
    find_magic_and_align(ser, buf)
    t1 = time.perf_counter()

    frame, _ = read_exact(ser, buf, FRAME_SIZE)
    t2 = time.perf_counter()

    magic, seq, czt_time_ms, drop_adc, drop_frame = struct.unpack_from(HEADER_FMT, frame, 0)
    if magic != MAGIC:
        # push back and resync
        buf[:0] = frame[1:]
        return None

    mag = np.array(struct.unpack_from(PAYLOAD_FMT, frame, HEADER_SIZE), dtype=np.float32)

    return seq, czt_time_ms, drop_adc, drop_frame, mag, t1


def make_frequency_axis():
    # Your CZT spans CENTER ± SPAN/2 over CZT_M bins
    f_start = CENTER_FREQ_HZ - SPAN_HZ * 0.5
    f_end   = CENTER_FREQ_HZ + SPAN_HZ * 0.5
    return np.linspace(f_start, f_end, CZT_M, endpoint=False)


def main():
    if not PORT:
        raise RuntimeError("Set PORT to your serial device (e.g., COM7 or /dev/tty.usbmodemXXXX).")

    freqs = make_frequency_axis()

    with serial.Serial(PORT, BAUD, timeout=0) as ser:
        ser.reset_input_buffer()
        buf = bytearray()

        fig, ax = plt.subplots()
        line, = ax.plot(freqs, np.zeros_like(freqs))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("LVS ADC CZT Magnitude (LVS0 frames)")
        ax.grid(True)

        plt.ion()
        fig.show()

        frames = 0

        # Timing stats over a REPORT_EVERY window
        czt_acc_ms = 0.0

        prev_frame_start_s = None
        period_acc_s = 0.0
        period_count = 0

        try:
            while True:
                fr = read_frame(ser, buf)
                if fr is None:
                    continue

                seq, czt_time_ms, drop_adc, drop_frame, mag, frame_start_s = fr

                # Update plot
                line.set_ydata(mag)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                # Timing accumulators
                frames += 1
                czt_acc_ms += float(czt_time_ms)

                if prev_frame_start_s is not None:
                    dt = frame_start_s - prev_frame_start_s
                    if dt > 0:
                        period_acc_s += dt
                        period_count += 1
                prev_frame_start_s = frame_start_s

                if frames % REPORT_EVERY == 0:
                    czt_avg_ms = czt_acc_ms / REPORT_EVERY

                    # Period is measured between consecutive frame starts, so we have REPORT_EVERY-1 intervals
                    if period_count > 0:
                        period_avg_ms = (period_acc_s / period_count) * 1000.0
                    else:
                        period_avg_ms = float('nan')

                    print(
                        f"{REPORT_EVERY} frames | last_seq={seq} "
                        f"czt_ms={czt_avg_ms:.2f} period_ms={period_avg_ms:.2f} | "
                        f"dropADC={drop_adc} dropFrame={drop_frame}"
                    )

                    # reset window
                    czt_acc_ms = 0.0
                    period_acc_s = 0.0
                    period_count = 0

        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C).")
            plt.ioff()
            return


if __name__ == "__main__":
    main()