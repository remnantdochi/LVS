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
    fs_full: float = 1e6
    fs_subsample: float = 100e3  # TBD


@dataclass
class RxConfig:
    """Receiver DSP parameters."""

    carrier_freq: float = 457e3
    pipeline_idx: int = 1 #[0 - nco mixer, 1 - cic filter, 2 - fir filter]
    cic_stages: int = 3
    cic_decimation_full: int = 16
    cic_decimation_subsample: int = 1
    fir_taps_num: int = 127
    fir_decimation_full: int = 6
    fir_decimation_subsample: int = 1
    fir_taps: tuple[float, ...] = (
        0.0000000000,
        -0.0000026943,
        -0.0000016250,
        0.0000281771,
        -0.0000366133,
        -0.0000312366,
        0.0001238044,
        -0.0000825709,
        -0.0001335602,
        0.0002867057,
        -0.0000909437,
        -0.0003459821,
        0.0004906458,
        -0.0000000000,
        -0.0006875791,
        0.0006820018,
        0.0002541244,
        -0.0011497956,
        0.0007827133,
        0.0007253590,
        -0.0016898209,
        0.0006974857,
        0.0014450071,
        -0.0022277665,
        0.0003246100,
        0.0024100259,
        -0.0026482061,
        -0.0004307966,
        0.0035737409,
        -0.0028059368,
        -0.0016427409,
        0.0048400235,
        -0.0025349981,
        -0.0033513257,
        0.0060612109,
        -0.0016590437,
        -0.0055525495,
        0.0070388718,
        -0.0000000000,
        -0.0081924777,
        0.0075246696,
        0.0026198682,
        -0.0111668392,
        0.0072151719,
        0.0063911075,
        -0.0143264644,
        0.0057268337,
        0.0115638221,
        -0.0174882527,
        0.0025162838,
        0.0185766626,
        -0.0204506567,
        -0.0033607557,
        0.0284297436,
        -0.0230120668,
        -0.0140721223,
        0.0440146080,
        -0.0249900754,
        -0.0368409852,
        0.0773929374,
        -0.0262394366,
        -0.1223262052,
        0.2878361582,
        0.6399994998,
        0.2878361582,
        -0.1223262052,
        -0.0262394366,
        0.0773929374,
        -0.0368409852,
        -0.0249900754,
        0.0440146080,
        -0.0140721223,
        -0.0230120668,
        0.0284297436,
        -0.0033607557,
        -0.0204506567,
        0.0185766626,
        0.0025162838,
        -0.0174882527,
        0.0115638221,
        0.0057268337,
        -0.0143264644,
        0.0063911075,
        0.0072151719,
        -0.0111668392,
        0.0026198682,
        0.0075246696,
        -0.0081924777,
        -0.0000000000,
        0.0070388718,
        -0.0055525495,
        -0.0016590437,
        0.0060612109,
        -0.0033513257,
        -0.0025349981,
        0.0048400235,
        -0.0016427409,
        -0.0028059368,
        0.0035737409,
        -0.0004307966,
        -0.0026482061,
        0.0024100259,
        0.0003246100,
        -0.0022277665,
        0.0014450071,
        0.0006974857,
        -0.0016898209,
        0.0007253590,
        0.0007827133,
        -0.0011497956,
        0.0002541244,
        0.0006820018,
        -0.0006875791,
        -0.0000000000,
        0.0004906458,
        -0.0003459821,
        -0.0000909437,
        0.0002867057,
        -0.0001335602,
        -0.0000825709,
        0.0001238044,
        -0.0000312366,
        -0.0000366133,
        0.0000281771,
        -0.0000016250,
        -0.0000026943,
        0.0000000000,
)



@dataclass
class SimulationConfig:
    """Top-level simulation control parameters."""

    tx: TxConfig = field(default_factory=TxConfig)
    adc: AdcConfig = field(default_factory=AdcConfig)
    rx: RxConfig = field(default_factory=RxConfig)

    duration: float = 0.00005  # Total simulation time in seconds
    plot_enabled: bool = True
    plot_stages: tuple[str, ...] = ("tx", "adc", "rx")  # ("tx", "adc", "rx")
    time_scale: float = 1e3
