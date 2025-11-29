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
        self.rx = Receiver(config.rx, mode =config.adc.mode)
        self.observers: List[Observer] = list(observers) if observers else [NullObserver()]
        self._collected_outputs: List[ReceiverOutputs] = []

    def run(self) -> List[ReceiverOutputs]:
        """Execute the simulation for the configured duration."""
        total_samples = int(math.ceil(self.config.duration * self.config.tx.fs))
        chunk_size = self.config.tx.chunk_size
        last_outputs: List[ReceiverOutputs] = []
        self._collected_outputs = []

        try:
            while True:
                self._reset_chain()
                remaining = total_samples
                outputs: List[ReceiverOutputs] = []
                restart_requested = False

                while remaining > 0:
                    self._wait_if_paused()
                    if self._restart_requested():
                        restart_requested = True
                        break

                    request = min(chunk_size, remaining)
                    tx_chunk = self.tx.generate_chunk(num_samples=request)
                    self._notify_tx(tx_chunk.plot_time, tx_chunk.plot_samples)

                    adc_time, adc_chunk = self.adc.process_chunk(tx_chunk)
                    self._notify_adc(adc_time, adc_chunk)

                    rx_outputs = self.rx.process_chunk(adc_time, adc_chunk)
                    self._notify_rx(rx_outputs)

                    outputs.append(rx_outputs)
                    self._collected_outputs = outputs
                    remaining -= len(tx_chunk.plot_samples)

                if restart_requested:
                    continue

                last_outputs = outputs
                self._collected_outputs = outputs

                if not self._wait_for_restart_after_completion():
                    return last_outputs
        except RuntimeError as exc:
            print(f"Simulation stopped early: {exc}")
            return self._collected_outputs

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

    def _wait_if_paused(self) -> None:
        for obs in self.observers:
            wait_fn = getattr(obs, "wait_if_paused", None)
            if callable(wait_fn):
                wait_fn()

    def _restart_requested(self) -> bool:
        restart = False
        for obs in self.observers:
            consume_fn = getattr(obs, "consume_restart_request", None)
            if callable(consume_fn) and consume_fn():
                restart = True
        return restart

    def _wait_for_restart_after_completion(self) -> bool:
        waited = False
        for obs in self.observers:
            wait_fn = getattr(obs, "wait_for_restart_after_completion", None)
            if callable(wait_fn):
                waited = True
                if wait_fn():
                    return True
        return False if waited else False

    def _reset_chain(self) -> None:
        tx_reset = getattr(self.tx, "reset", None)
        if callable(tx_reset):
            tx_reset()
        adc_reset = getattr(self.adc, "reset", None)
        if callable(adc_reset):
            adc_reset()
        rx_reset = getattr(self.rx, "reset", None)
        if callable(rx_reset):
            rx_reset()


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


def _wait_for_observers_to_start(observers: Sequence[Observer] | None) -> None:
    if not observers:
        return
    for obs in observers:
        wait_fn = getattr(obs, "wait_for_play", None)
        if callable(wait_fn):
            wait_fn()


def run_simulation(args: argparse.Namespace) -> List[ReceiverOutputs]:
    config = set_config_from_arg(args)
    observers = create_observers_from_config(config)
    _wait_for_observers_to_start(observers)
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

    snr_values = [out.snr_db for out in outputs if getattr(out, "snr_db", None) is not None]
    if snr_values:
        avg_snr = float(np.mean(snr_values))
        print(f"Average detected SNR: {avg_snr:.2f} dB")


if __name__ == "__main__":
    main()
