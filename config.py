"""Configuration dataclasses for the LVS simulation pipeline."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TxConfig:
    """Transmitter operating parameters."""

    fs: float = 5e6  # Sample rate in Hz
    chunk_size: int = 204800  # Samples per generated chunk - match to DMA buffer size
    # 25khz subsample rate * 1024 samples = 40.96ms per CZT frame
    # TX sample rate 5Mhz * 40.96ms = 204800 samples per chunk
    center_freq: float = 457e3  # Nominal beacon carrier frequency
    freq_tolerance: float = 100.0  # ± tolerance in Hz
    pulse_length: float = 0.07  # Pulse ON duration in seconds
    pulse_period: float = 1.0  # Pulse repetition period in seconds
    pulse_period_tolerance: float = 0.3  # ± tolerance in seconds
    use_pulse_envelope: bool = False
    extra_carriers: int = 2
    tx_fft_resolution_hz: float = 1
    extra_carrier_offsets: tuple[float, ...] = (-30,80)

    seed: int | None = None
    randomize: bool = False
    use_awgn: bool = False
    awgn_snr: float = 30.0  # dB


@dataclass
class AdcConfig:
    """MCU ADC sampling parameters."""
    mode: str = "subsample"  # "full" or "subsample"
    fs_full: float = 1e6
    fs_subsample: float = 25000.0


@dataclass
class RxConfig:
    """Receiver DSP parameters."""

    carrier_freq: float = 457e3
    pipeline_idx: int = 4 #[0 - nco mixer, 1 - cic filter, 2 - fir/iir filter, 3 - fft/czt detection, 4 - czt only]
    filter_type: Literal["fir", "iir"] = "fir"
    
    cic_stages: int = 3
    cic_decimation_full: int = 16
    cic_decimation_subsample: int = 3

    fir_taps: int = 127
    fir_cutoff_hz: float = 500
    fir_sample_full: float = 1e6 / 16
    fir_sample_subsample: float = 25000.0 / 3

    fir_decimation_full: int = 6
    fir_decimation_subsample: int = 1
    fir_coefficients_full: tuple[float, ...] = (
        0.0000000000,
        -0.0000004791,
        -0.0000056647,
        -0.0000202718,
        -0.0000465208,
        -0.0000828864,
        -0.0001235126,
        -0.0001584821,
        -0.0001750002,
        -0.0001594057,
        -0.0000997701,
        0.0000112789,
        0.0001739527,
        0.0003793857,
        0.0006088312,
        0.0008342818,
        0.0010206563,
        0.0011294868,
        0.0011238159,
        0.0009738001,
        0.0006623539,
        0.0001900678,
        -0.0004213727,
        -0.0011279462,
        -0.0018647236,
        -0.0025501812,
        -0.0030932824,
        -0.0034028560,
        -0.0033984612,
        -0.0030216510,
        -0.0022463634,
        -0.0010871212,
        0.0003961888,
        0.0020979384,
        0.0038713911,
        0.0055384658,
        0.0069042153,
        0.0077750500,
        0.0079792459,
        0.0073878892,
        0.0059341855,
        0.0036290311,
        0.0005709245,
        -0.0030513085,
        -0.0069639369,
        -0.0108221356,
        -0.0142316529,
        -0.0167770111,
        -0.0180542148,
        -0.0177053840,
        -0.0154523870,
        -0.0111264623,
        -0.0046910058,
        0.0037448365,
        0.0139234367,
        0.0254475976,
        0.0378009138,
        0.0503783832,
        0.0625251478,
        0.0735804822,
        0.0829236234,
        0.0900178036,
        0.0944489264,
        0.0959557265,
        0.0944489264,
        0.0900178036,
        0.0829236234,
        0.0735804822,
        0.0625251478,
        0.0503783832,
        0.0378009138,
        0.0254475976,
        0.0139234367,
        0.0037448365,
        -0.0046910058,
        -0.0111264623,
        -0.0154523870,
        -0.0177053840,
        -0.0180542148,
        -0.0167770111,
        -0.0142316529,
        -0.0108221356,
        -0.0069639369,
        -0.0030513085,
        0.0005709245,
        0.0036290311,
        0.0059341855,
        0.0073878892,
        0.0079792459,
        0.0077750500,
        0.0069042153,
        0.0055384658,
        0.0038713911,
        0.0020979384,
        0.0003961888,
        -0.0010871212,
        -0.0022463634,
        -0.0030216510,
        -0.0033984612,
        -0.0034028560,
        -0.0030932824,
        -0.0025501812,
        -0.0018647236,
        -0.0011279462,
        -0.0004213727,
        0.0001900678,
        0.0006623539,
        0.0009738001,
        0.0011238159,
        0.0011294868,
        0.0010206563,
        0.0008342818,
        0.0006088312,
        0.0003793857,
        0.0001739527,
        0.0000112789,
        -0.0000997701,
        -0.0001594057,
        -0.0001750002,
        -0.0001584821,
        -0.0001235126,
        -0.0000828864,
        -0.0000465208,
        -0.0000202718,
        -0.0000056647,
        -0.0000004791,
        0.0000000000,
)

    fir_coefficients_subsample: tuple[float, ...] = (
        -0.0000000000,
        -0.0000031344,
        -0.0000109467,
        -0.0000174141,
        -0.0000133011,
        0.0000106348,
        0.0000597603,
        0.0001320615,
        0.0002161019,
        0.0002912955,
        0.0003308825,
        0.0003073766,
        0.0001996141,
        -0.0000000000,
        -0.0002797343,
        -0.0006059024,
        -0.0009245867,
        -0.0011682027,
        -0.0012664389,
        -0.0011601175,
        -0.0008156755,
        -0.0002374656,
        0.0005249509,
        0.0013768161,
        0.0021867577,
        0.0028037737,
        0.0030808675,
        0.0029020911,
        0.0022086623,
        0.0010193576,
        -0.0005592866,
        -0.0023362764,
        -0.0040544000,
        -0.0054224826,
        -0.0061582448,
        -0.0060361380,
        -0.0049329824,
        -0.0028636910,
        0.0000000000,
        0.0033330234,
        0.0066850485,
        0.0095319287,
        0.0113456092,
        0.0116742291,
        0.0102217459,
        0.0069153757,
        0.0019497545,
        -0.0042009748,
        -0.0108081826,
        -0.0169511212,
        -0.0216117004,
        -0.0237918660,
        -0.0226399645,
        -0.0175703007,
        -0.0083599620,
        0.0047909866,
        0.0212458247,
        0.0399683786,
        0.0596091278,
        0.0786319212,
        0.0954675656,
        0.1086767460,
        0.1171031153,
        0.1199982184,
        0.1171031153,
        0.1086767460,
        0.0954675656,
        0.0786319212,
        0.0596091278,
        0.0399683786,
        0.0212458247,
        0.0047909866,
        -0.0083599620,
        -0.0175703007,
        -0.0226399645,
        -0.0237918660,
        -0.0216117004,
        -0.0169511212,
        -0.0108081826,
        -0.0042009748,
        0.0019497545,
        0.0069153757,
        0.0102217459,
        0.0116742291,
        0.0113456092,
        0.0095319287,
        0.0066850485,
        0.0033330234,
        0.0000000000,
        -0.0028636910,
        -0.0049329824,
        -0.0060361380,
        -0.0061582448,
        -0.0054224826,
        -0.0040544000,
        -0.0023362764,
        -0.0005592866,
        0.0010193576,
        0.0022086623,
        0.0029020911,
        0.0030808675,
        0.0028037737,
        0.0021867577,
        0.0013768161,
        0.0005249509,
        -0.0002374656,
        -0.0008156755,
        -0.0011601175,
        -0.0012664389,
        -0.0011682027,
        -0.0009245867,
        -0.0006059024,
        -0.0002797343,
        -0.0000000000,
        0.0001996141,
        0.0003073766,
        0.0003308825,
        0.0002912955,
        0.0002161019,
        0.0001320615,
        0.0000597603,
        0.0000106348,
        -0.0000133011,
        -0.0000174141,
        -0.0000109467,
        -0.0000031344,
        -0.0000000000,
)

    iir_order: int = 2
    iir_cutoff_hz: float = 20e3
    iir_sample_rate_hz: float = 62500.0
    iir_b_coefficients: tuple[float, ...] = (
        0.4347393483,
        0.8694786966,
        0.4347393483,
    )
    iir_a_coefficients: tuple[float, ...] = (
        1.0000000000,
        0.5193034092,
        0.2196539839,
    )
    iir_decimation_full: int = 6
    iir_decimation_subsample: int = 1

    fft_size_full: int = 1024
    fft_size_subsample: int = 1024
    fft_threshold_db: float = 8
    fft_min_magnitude: float = 1e-9
    
    czt_bins: int = 256
    czt_span_hz: float = 200.0
    czt_center_hz: float = 457000.0
    czt_sample_rate: float = 25000.0
    czt_window: str = "none"



@dataclass
class SimulationConfig:
    """Top-level simulation control parameters."""

    tx: TxConfig = field(default_factory=TxConfig)
    adc: AdcConfig = field(default_factory=AdcConfig)
    rx: RxConfig = field(default_factory=RxConfig)

    duration: float = 10  # Total simulation time in seconds
    plot_enabled: bool = True
    plot_stages: tuple[str, ...] = ("tx", "rx", "czt")
    time_scale: float = 1e3
    plot_peak_count: int = 3
