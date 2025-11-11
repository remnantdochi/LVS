"""Entry point for the LVS simulation."""
from __future__ import annotations

import argparse
import math
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from config import SimulationConfig
from lvs_adc import LvsAdc
from lvs_rx import Receiver, ReceiverOutputs
from lvs_tx import Transmitter
from observer import NullObserver, Observer, PyQtGraphObserver


DEFAULT_CONFIG = SimulationConfig()


class SimulationEngine:
    """Coordinates the TX → ADC → RX processing flow."""

    def __init__(
        self,
        config: SimulationConfig,
        observers: Sequence[Observer] | None = None,
    ) -> None:
        self.config = config
        self.tx = Transmitter(config.tx)
        self.adc = LvsAdc(config.adc)
        self.rx = Receiver(config.rx)
        self.observers: List[Observer] = list(observers) if observers else [NullObserver()]

    def run(self) -> List[ReceiverOutputs]:
        """Execute the simulation for the configured duration."""
        total_samples = int(math.ceil(self.config.duration * self.config.tx.fs))
        chunk_size = self.config.tx.chunk_size
        remaining = total_samples

        outputs: List[ReceiverOutputs] = []

        while remaining > 0:
            request = min(chunk_size, remaining)
            tx_chunk = self.tx.generate_chunk(num_samples=request)
            self._notify_tx(tx_chunk.plot_time, tx_chunk.plot_samples)

            adc_time, adc_chunk = self.adc.process_chunk(tx_chunk)
            self._notify_adc(adc_time, adc_chunk)

            # 👇👇 디버깅 블록 시작 👇👇
            # ADC가 실제로 샘플한 시각에 맞춰 TX/ADC 값을 비교한다.
            for t_debug, adc_val in zip(adc_time, adc_chunk):
                tx_val = tx_chunk.sample(np.array([t_debug]))[0]
                print(
                    f"[{t_debug*1e6:8.3f} ms] TX={tx_val:+.4f}, ADC={adc_val:+.4f}, diff={tx_val-adc_val:+.4e}"
                )
            # 👆👆 디버깅 블록 끝 👆👆

            rx_outputs = self.rx.process_chunk(adc_time, adc_chunk)
            self._notify_rx(rx_outputs)

            outputs.append(rx_outputs)
            remaining -= len(tx_chunk.plot_samples)

        return outputs

    def _notify_tx(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None:
        for obs in self.observers:
            obs.on_tx(time, samples)

    def _notify_adc(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None:
        for obs in self.observers:
            obs.on_adc(time, samples)

    def _notify_rx(self, outputs: ReceiverOutputs) -> None:
        for obs in self.observers:
            obs.on_rx(outputs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LVS beacon simulation.")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_CONFIG.duration,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_CONFIG.tx.fs,
        help="Sample rate (Hz) used by the transmitter.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CONFIG.tx.chunk_size,
        help="Number of samples processed per chunk.",
    )
    parser.add_argument(
        "--plot-stages",
        nargs="*",
        choices=("tx", "adc", "rx"),
        default=None,
        help="Subset of stages to plot when --plot is enabled.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1e3,
        help="Scale factor applied to time axis in plots (default ms).",
    )
    return parser


def set_config_from_arg(args: argparse.Namespace) -> SimulationConfig:
    # Start with defaults from config.py
    # Then override with any command-line arguments.
    config = SimulationConfig()

    if getattr(args, "duration", None) is not None:
        config.duration = args.duration
    if getattr(args, "fs", None) is not None:
        config.tx.fs = args.fs
    if getattr(args, "chunk_size", None) is not None:
        config.tx.chunk_size = args.chunk_size
    if getattr(args, "plot_stages", None):
        config.plot_stages = tuple(args.plot_stages)
    if getattr(args, "time_scale", None) is not None:
        config.time_scale = args.time_scale

    return config


def create_observers_from_config(config: SimulationConfig) -> List[Observer] | None:
    stages = getattr(config, "plot_stages", ("tx", "adc", "rx"))
    print(f"Enabling plots for stages: {stages}")
    try:
        observer = PyQtGraphObserver(
            show_tx="tx" in stages,
            show_adc="adc" in stages,
            show_rx="rx" in stages,
            time_scale=config.time_scale,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Unable to start PyQtGraph observer: {exc}") from exc

    return [observer]


def run_simulation(args: argparse.Namespace) -> List[ReceiverOutputs]:
    config = set_config_from_arg(args)
    observers = create_observers_from_config(config)
    engine = SimulationEngine(config=config, observers=observers)
    outputs = engine.run()

    # Allow interactive observers to keep their windows open.
    if observers:
        for obs in observers:
            if hasattr(obs, "exec"):
                obs.exec()  # type: ignore[attr-defined]

    return outputs


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_simulation(args)

    if not outputs:
        print("No data produced by the simulation.")
        return
    total_samples = sum(len(out.processed) for out in outputs)
    print(f"Simulation complete: {total_samples} samples processed.")


if __name__ == "__main__":
    main()
