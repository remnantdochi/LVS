"""Configuration dataclasses for the LVS simulation pipeline."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TxConfig:
    """Transmitter operating parameters."""

    fs: float = 5e6  # Sample rate in Hz
    chunk_size: int = 5000  # Samples per generated chunk - match to DMA buffer size
    center_freq: float = 457e3  # Nominal beacon carrier frequency
    freq_tolerance: float = 100.0  # ± tolerance in Hz
    pulse_length: float = 0.07  # Pulse ON duration in seconds
    pulse_period: float = 1.0  # Pulse repetition period in seconds
    pulse_period_tolerance: float = 0.3  # ± tolerance in seconds
    use_pulse_envelope: bool = False

    seed: int | None = None
    randomize: bool = False
    use_awgn: bool = True
    awgn_snr: float = 30.0  # dB


@dataclass
class AdcConfig:
    """MCU ADC sampling parameters."""
    mode: str = "full"  # "full" or "subsample"
    fs_full: float = 1.2e6
    fs_subsample: float = 100e3  # TBD


@dataclass
class RxConfig:
    """Receiver DSP parameters."""

    carrier_freq: float = 457e3


@dataclass
class SimulationConfig:
    """Top-level simulation control parameters."""

    tx: TxConfig = field(default_factory=TxConfig)
    adc: AdcConfig = field(default_factory=AdcConfig)
    rx: RxConfig = field(default_factory=RxConfig)

    duration: float = 0.05  # Total simulation time in seconds
    plot_enabled: bool = True
    plot_stages: tuple[str, ...] = ("tx", "adc")  # ("tx", "adc", "rx")
    time_scale: float = 1e3
