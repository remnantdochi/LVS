# tx.py
from dataclasses import dataclass
import numpy as np


@dataclass
class Transmitter:
    pulse_length_max: float = 0.10  # 100 ms maximum on-time (user selectable)

    center_freq: float = 457e3  # 457 kHz nominal
    freq_tolerance: float = 100.0  # ±100 Hz tolerance
    pulse_length_min: float = 0.07  # 70 ms minimum on-time
    pulse_off_min: float = 0.4      # 400 ms minimum off-time
    pulse_period_center: float = 1.0  # 1000 ms nominal period
    pulse_period_tolerance: float = 0.3  # ±300 ms tolerance
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

        if self.pulse_length_min > self.pulse_length_max:
            raise ValueError("pulse_length_min cannot be greater than pulse_length_max")

    def _draw_carrier_freq(self) -> float:
        return self._rng.uniform(
            self.center_freq - self.freq_tolerance,
            self.center_freq + self.freq_tolerance,
        )

    def _draw_pulse_period(self) -> float:
        return self._rng.uniform(
            self.pulse_period_center - self.pulse_period_tolerance,
            self.pulse_period_center + self.pulse_period_tolerance,
        )

    def generate(
        self,
        duration: float,
        fs: float,
        randomize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        carrier_freq = self._draw_carrier_freq() if randomize else self.center_freq
        pulse_period = self._draw_pulse_period() if randomize else self.pulse_period_center

        pulse_length = float(self._rng.uniform(self.pulse_length_min, self.pulse_length_max)) if randomize else self.pulse_length_min
        
        # Ensure OFF region meets minimum requirement; if not, force ON to minimum
        if (pulse_period - pulse_length) < self.pulse_off_min:
            pulse_length = self.pulse_length_min

        t = np.arange(0, duration, 1 / fs)
        burst = ((t % pulse_period) < pulse_length).astype(float)
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        # use cosine to have a peak at t=0
        return t, burst * carrier
