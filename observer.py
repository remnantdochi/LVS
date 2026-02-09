"""Observer interfaces for tapping into the simulation pipeline."""
from __future__ import annotations

import sys
import threading
from typing import Dict, Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from lvs_rx import ReceiverOutputs


class Observer(Protocol):
    """Interface for receiving callbacks from the simulation engine."""

    def on_tx(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None: 
        ...

    def on_adc(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None: 
        ...

    def on_rx(self, outputs: ReceiverOutputs) -> None: 
        ...

    def on_czt(self, outputs: ReceiverOutputs) -> None:
        ...


class NullObserver:
    """Default observer that drops all callbacks."""

    def on_tx(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None:
        pass

    def on_adc(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None:
        pass

    def on_rx(self, outputs: ReceiverOutputs) -> None:
        pass

    def on_czt(self, outputs: ReceiverOutputs) -> None:
        pass


class PyQtGraphObserver:
    """
    Real-time plotter for the simulation stages using pyqtgraph.

    Features
    --------
    - Native zoom/pan via mouse wheel and drag (pyqtgraph defaults).
    - Space bar or toolbar button toggles pause/resume.
    - Toolbar button switches between time-domain waveform and FFT magnitude views.
    - Mouse hover shows the current time/value readback.
    - Optional down-sampling to keep refresh responsive.
    """

    def __init__(
        self,
        show_tx: bool = True,
        show_adc: bool = True,
        show_rx: bool = True,
        show_czt: bool = False,
        max_points: int = 5000,
        time_scale: float = 1e3,
        view_domain: Literal["time", "frequency"] = "time",
        freq_view_max: float | None = 1e6,
        freq_view_center: float | None = None,
        freq_view_span: float | None = None,
        freq_view_tx_center: float | None = None,
        freq_view_tx_span: float | None = None,
        tx_fft_resolution_hz: float | None = None,
        peak_count: int = 3,
        link_x_axes: bool = True,
        window_title: str = "LVS Simulation Monitor",
        plot_height: int = 220,
        row_spacing: int = 32,
    ) -> None:
        try:
            from PyQt5 import QtCore, QtWidgets  # type: ignore
            import pyqtgraph as pg  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires GUI libs
            raise RuntimeError(
                "PyQtGraphObserver requires PyQt5 and pyqtgraph to be installed."
            ) from exc

        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.pg = pg
        # Use a light theme for the plot background.
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self.show_tx = show_tx
        self.show_adc = show_adc
        self.show_rx = show_rx
        self.show_czt = show_czt
        self.max_points = max_points
        self.time_scale = time_scale
        self.view_domain = view_domain
        self.freq_view_max = freq_view_max
        self.freq_view_center = freq_view_center
        self.freq_view_span = freq_view_span
        self.freq_view_tx_center = freq_view_tx_center
        self.freq_view_tx_span = freq_view_tx_span
        self.tx_fft_resolution_hz = tx_fft_resolution_hz
        self.peak_count = max(1, int(peak_count))
        self.link_x_axes = link_x_axes
        self.plot_height = plot_height
        self.row_spacing = row_spacing

        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication(sys.argv or ["lvs-sim"])
            self._owns_app = True
        else:
            self._owns_app = False

        self._last_data: Dict[str, NDArray[np.float_]] = {}
        self._last_time: Dict[str, NDArray[np.float_]] = {}
        self._last_fft_freqs: Dict[str, NDArray[np.float_]] = {}
        self._last_fft_mag: Dict[str, NDArray[np.float_]] = {}
        self._peak_labels: Dict[str, list["pyqtgraph.TextItem"]] = {}
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle(window_title)
        self._build_ui()
        self._active = False
        self._paused = False
        self._resume_event = threading.Event()
        self._restart_event = threading.Event()

        if not any((show_tx, show_adc, show_rx, show_czt)):
            self.show_tx = True

        self._window.show()
        self._process_events()

    # region UI setup -----------------------------------------------------
    def _build_ui(self) -> None:
        QtWidgets = self.QtWidgets
        pg = self.pg

        central = QtWidgets.QWidget()
        central.setStyleSheet("background-color: white;")
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._toolbar = QtWidgets.QToolBar("Controls", parent=self._window)
        self._window.addToolBar(self._toolbar)

        self._view_action = self._toolbar.addAction("Frequency View")
        self._view_action.setCheckable(True)
        self._view_action.toggled.connect(
            lambda checked: self._set_view_domain("frequency" if checked else "time")
        )

        self._play_pause_action = self._toolbar.addAction("Play")
        self._play_pause_action.setCheckable(True)
        self._play_pause_action.toggled.connect(self._toggle_play_pause)
        self._view_action.setChecked(self.view_domain == "frequency")

        self._toolbar.addSeparator()
        self._toolbar.addAction("Clear", self._clear_plots)

        self._status_label = QtWidgets.QLabel("Press Play to start streaming.")
        layout.addWidget(self._status_label)

        self._plot_container = pg.GraphicsLayoutWidget(show=False)
        self._plot_container.ci.layout.setSpacing(self.row_spacing)
        layout.addWidget(self._plot_container)

        self._window.setCentralWidget(central)

        self._plots: Dict[str, "pyqtgraph.PlotItem"] = {}
        self._curves: Dict[str, "pyqtgraph.PlotDataItem"] = {}

        keys: Sequence[str] = []
        if self.show_tx:
            keys = (*keys, "tx")
        if self.show_adc:
            keys = (*keys, "adc")
        if self.show_rx:
            keys = (*keys, "rx")
        if self.show_czt:
            keys = (*keys, "czt")

        for idx, key in enumerate(keys):
            plot = self._plot_container.addPlot(title=key.upper())
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setMouseEnabled(x=True, y=False)
            plot.setLabel("bottom", self._x_axis_label(), units=self._x_axis_units())
            plot.setLabel("left", "Amplitude" if self.view_domain == "time" else "|X(f)|")
            plot.setMinimumHeight(self.plot_height)
            curve = plot.plot([], [], pen=pg.mkPen(width=1.2), symbol="o", symbolSize=3)
            self._plots[key] = plot
            self._curves[key] = curve
            labels: list["pyqtgraph.TextItem"] = []
            for _ in range(self.peak_count):
                label = pg.TextItem(color=(255, 180, 0), anchor=(0, 1))
                label.setVisible(False)
                plot.addItem(label)
                labels.append(label)
            self._peak_labels[key] = labels
            if idx < len(keys) - 1:
                self._plot_container.nextRow()

        self._plot_container.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._window.keyPressEvent = self._key_press  # type: ignore

        if self.link_x_axes and len(self._plots) > 1:
            first_plot = list(self._plots.values())[0]
            for plot in list(self._plots.values())[1:]:
                plot.setXLink(first_plot)
        self._apply_x_range_limits()

    # endregion -----------------------------------------------------------

    # region Qt helpers ---------------------------------------------------
    def _process_events(self) -> None:
        self._app.processEvents(self.QtCore.QEventLoop.AllEvents, 50)

    def wait_for_play(self, poll_ms: int = 50) -> None:
        """
        Block the caller until the Play button is pressed or the window closes.
        """
        if self._resume_event.is_set():
            return
        self._wait_for_resume(poll_ms, "Observer closed before Play was pressed.")

    def wait_if_paused(self, poll_ms: int = 50) -> None:
        """
        Block the caller while the observer is paused or inactive.
        """
        if self._resume_event.is_set():
            return
        self._wait_for_resume(poll_ms, "Observer closed while paused.")

    def wait_for_restart_after_completion(self, poll_ms: int = 50) -> bool:
        """
        After a run completes, wait for Clear to be pressed or the window to close.
        Returns True if a restart was requested, False if the window closed.
        """
        if self.consume_restart_request():
            return True
        while self._window.isVisible():
            if self.consume_restart_request():
                return True
            self._process_events()
            self.QtCore.QThread.msleep(poll_ms)
        return False

    def _wait_for_resume(self, poll_ms: int, closed_message: str) -> None:
        while self._window.isVisible():
            if self._resume_event.is_set():
                return
            self._process_events()
            self.QtCore.QThread.msleep(poll_ms)
        raise RuntimeError(closed_message)

    def _key_press(self, event):
        if event.key() == self.QtCore.Qt.Key_Space:
            if self._active:
                self._play_pause_action.toggle()
        else:
            self.QtWidgets.QMainWindow.keyPressEvent(self._window, event)

    def _toggle_play_pause(self, checked: bool) -> None:
        if checked:
            self._active = True
            self._paused = False
            self._resume_event.set()
            self._play_pause_action.setText("Pause")
            self._status_label.setText(
                "Streaming running. Hover a plot for time/value (Space toggles pause)."
            )
        else:
            if not self._active:
                return
            self._paused = True
            self._resume_event.clear()
            self._play_pause_action.setText("Play")
            self._status_label.setText("Streaming paused. Press button or Space to resume.")

    def _clear_plots(self) -> None:
        for curve in self._curves.values():
            curve.setData([], [])
        self._last_data.clear()
        self._last_time.clear()
        self._last_fft_freqs.clear()
        self._last_fft_mag.clear()
        self._resume_event.clear()
        self._restart_event.set()
        self._active = False
        self._paused = False
        with self.QtCore.QSignalBlocker(self._play_pause_action):
            self._play_pause_action.setChecked(False)
            self._play_pause_action.setText("Play")
        self._status_label.setText("Plots cleared. Press Play to restart.")
        self._process_events()

    def consume_restart_request(self) -> bool:
        """
        Return True once when the user has requested a restart via Clear.
        """
        if self._restart_event.is_set():
            self._restart_event.clear()
            return True
        return False

    def _time_unit(self) -> str:
        if self.time_scale == 1:
            return "s"
        if self.time_scale == 1e3:
            return "ms"
        if self.time_scale == 1e6:
            return "µs"
        return "scaled"

    def _on_mouse_moved(self, pos):
        for key, plot in self._plots.items():
            if plot.sceneBoundingRect().contains(pos):
                vb = plot.getViewBox()
                mouse_point = vb.mapSceneToView(pos)
                self._status_label.setText(
                    self._format_hover_status(
                        key,
                        mouse_point.x(),
                        mouse_point.y(),
                    )
                )
                break

    # endregion -----------------------------------------------------------

    # region Observer callbacks -------------------------------------------
    def on_tx(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None:
        if not self.show_tx:
            return
        self._update_plot("tx", time, samples)

    def on_adc(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None:
        if not self.show_adc:
            return
        self._update_plot("adc", time, samples)

    def on_rx(self, outputs: ReceiverOutputs) -> None:
        if not self.show_rx:
            return
        if outputs.fft_freqs is not None and outputs.fft_magnitude is not None:
            self._last_fft_freqs["rx"] = outputs.fft_freqs
            self._last_fft_mag["rx"] = outputs.fft_magnitude
        self._update_plot("rx", outputs.time, outputs.processed)

    def on_czt(self, outputs: ReceiverOutputs) -> None:
        if not self.show_czt:
            return
        if outputs.fft_freqs is not None and outputs.fft_magnitude is not None:
            self._last_fft_freqs["czt"] = outputs.fft_freqs
            self._last_fft_mag["czt"] = outputs.fft_magnitude
        self._update_plot("czt", outputs.time, outputs.processed)

    # endregion -----------------------------------------------------------

    def _update_plot(
        self,
        key: str,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None:
        if not self._active or self._paused:
            return
        if not self._render_curve(key, time, samples):
            return
        self._last_time[key] = time
        self._last_data[key] = samples
        self._process_events()

    def _render_curve(
        self,
        key: str,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> bool:
        curve = self._curves.get(key)
        if curve is None:
            return False

        x_vals, y_vals = self._transform_data(time, samples, key=key)

        if self.max_points and len(x_vals) > self.max_points:
            idx = np.linspace(0, len(x_vals) - 1, self.max_points).astype(int)
            x_vals = x_vals[idx]
            y_vals = y_vals[idx]

        curve.setData(x_vals, y_vals)
        self._update_peak_label(key, x_vals, y_vals)
        return True

    def _set_view_domain(self, domain: Literal["time", "frequency"]) -> None:
        if domain == self.view_domain:
            return
        self.view_domain = domain
        if self._view_action is not None:
            self._view_action.blockSignals(True)
            self._view_action.setChecked(domain == "frequency")
            self._view_action.blockSignals(False)
        self._update_axis_labels()
        self._apply_x_range_limits()
        self._update_peak_visibility()
        self._replot_all_cached()

    def _update_axis_labels(self) -> None:
        for plot in self._plots.values():
            plot.setLabel("bottom", self._x_axis_label(), units=self._x_axis_units())
            plot.setLabel("left", "Amplitude" if self.view_domain == "time" else "|X(f)|")

    def _apply_x_range_limits(self) -> None:
        if not self._plots:
            return
        for key, plot in self._plots.items():
            if self.view_domain != "frequency":
                plot.enableAutoRange("x", True)
                continue

            if key == "czt" and self.freq_view_center is not None and self.freq_view_span:
                half_span = self.freq_view_span / 2.0
                plot.enableAutoRange("x", False)
                plot.setXRange(
                    self.freq_view_center - half_span,
                    self.freq_view_center + half_span,
                    padding=0.01,
                )
                continue

            if key == "rx" and self.freq_view_span:
                half_span = self.freq_view_span / 2.0
                plot.enableAutoRange("x", False)
                plot.setXRange(-half_span, half_span, padding=0.01)
                continue

            if (
                key == "tx"
                and self.freq_view_tx_center is not None
                and self.freq_view_tx_span
            ):
                half_span = self.freq_view_tx_span / 2.0
                plot.enableAutoRange("x", False)
                plot.setXRange(
                    self.freq_view_tx_center - half_span,
                    self.freq_view_tx_center + half_span,
                    padding=0.01,
                )
                continue

            if self.freq_view_max:
                plot.enableAutoRange("x", False)
                plot.setXRange(0, self.freq_view_max, padding=0.01)
            else:
                plot.enableAutoRange("x", True)

    def _replot_all_cached(self) -> None:
        updated = False
        for key in self._curves:
            time = self._last_time.get(key)
            data = self._last_data.get(key)
            if time is None or data is None:
                continue
            if self._render_curve(key, time, data):
                updated = True
        if updated:
            self._process_events()

    def _transform_data(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
        *,
        key: str,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        if self.view_domain == "frequency":
            if key == "rx":
                fft_freqs = self._last_fft_freqs.get(key)
                fft_mag = self._last_fft_mag.get(key)
                if fft_freqs is not None and fft_mag is not None:
                    return fft_freqs, fft_mag
                return self._compute_fft(time, samples)
            if key == "tx":
                return self._compute_fft_high_res(time, samples)
            fft_freqs = self._last_fft_freqs.get(key)
            fft_mag = self._last_fft_mag.get(key)
            if fft_freqs is not None and fft_mag is not None:
                return fft_freqs, fft_mag
            return self._compute_fft(time, samples)
        return time * self.time_scale, samples

    def _compute_fft(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        if len(samples) == 0:
            empty = np.array([], dtype=float)
            return empty, empty
        if len(time) < 2:
            freqs = np.array([0.0], dtype=float)
            magnitudes = np.abs(samples[:1])
            return freqs, magnitudes
        spacing = float(np.mean(np.diff(time)))
        if spacing <= 0:
            spacing = 1.0
        fft_vals = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), d=spacing)
        fft_vals = np.fft.fftshift(fft_vals)
        freqs = np.fft.fftshift(freqs)
        magnitudes = np.abs(fft_vals)
        if self.freq_view_max is not None:
            mask = np.abs(freqs) <= self.freq_view_max
            if not np.any(mask):
                mask = np.ones_like(freqs, dtype=bool)
            freqs = freqs[mask]
            magnitudes = magnitudes[mask]
        return freqs, magnitudes

    def _compute_fft_full(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        if len(samples) == 0:
            empty = np.array([], dtype=float)
            return empty, empty
        if len(time) < 2:
            freqs = np.array([0.0], dtype=float)
            magnitudes = np.abs(samples[:1])
            return freqs, magnitudes
        spacing = float(np.mean(np.diff(time)))
        if spacing <= 0:
            spacing = 1.0
        fft_vals = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), d=spacing)
        fft_vals = np.fft.fftshift(fft_vals)
        freqs = np.fft.fftshift(freqs)
        magnitudes = np.abs(fft_vals)
        if self.freq_view_span is not None and self.freq_view_center is not None:
            half_span = self.freq_view_span / 2.0
            mask = (freqs >= -half_span) & (freqs <= half_span)
            if np.any(mask):
                freqs = freqs[mask]
                magnitudes = magnitudes[mask]
        return freqs, magnitudes

    def _compute_fft_high_res(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        if len(samples) == 0:
            empty = np.array([], dtype=float)
            return empty, empty
        if len(time) < 2:
            freqs = np.array([0.0], dtype=float)
            magnitudes = np.abs(samples[:1])
            return freqs, magnitudes
        spacing = float(np.mean(np.diff(time)))
        if spacing <= 0:
            spacing = 1.0
        sample_rate = 1.0 / spacing if spacing > 0 else 0.0
        target_res = self.tx_fft_resolution_hz
        if not target_res or target_res <= 0.0 or sample_rate <= 0.0:
            return self._compute_fft(time, samples)
        target_n = int(round(sample_rate / target_res))
        target_n = max(target_n, len(samples))
        fft_vals = np.fft.rfft(samples, n=target_n)
        freqs = np.fft.rfftfreq(target_n, d=spacing)
        magnitudes = np.abs(fft_vals)
        if self.freq_view_tx_center is not None and self.freq_view_tx_span:
            half_span = self.freq_view_tx_span / 2.0
            mask = (
                (freqs >= self.freq_view_tx_center - half_span)
                & (freqs <= self.freq_view_tx_center + half_span)
            )
            if np.any(mask):
                freqs = freqs[mask]
                magnitudes = magnitudes[mask]
        return freqs, magnitudes

    def _x_axis_label(self) -> str:
        return "Frequency" if self.view_domain == "frequency" else "Time"

    def _x_axis_units(self) -> str:
        return "Hz" if self.view_domain == "frequency" else self._time_unit()

    def _format_hover_status(self, key: str, x_val: float, y_val: float) -> str:
        if self.view_domain == "frequency":
            return f"{key.upper()}  f={x_val:.1f}Hz  |X|={y_val:.3f}"
        time_value = x_val / self.time_scale
        return f"{key.upper()}  t={time_value:.6f}s  value={y_val:.6f}"

    def _update_peak_label(
        self,
        key: str,
        x_vals: NDArray[np.float_],
        y_vals: NDArray[np.float_],
    ) -> None:
        labels = self._peak_labels.get(key)
        if not labels:
            return
        if self.view_domain != "frequency" or len(x_vals) == 0:
            for label in labels:
                label.setVisible(False)
            return
        peaks = self._find_peaks(x_vals, y_vals, len(labels))
        for idx, label in enumerate(labels):
            if idx >= len(peaks):
                label.setVisible(False)
                continue
            freq, mag = peaks[idx]
            display_freq = freq
            if key == "rx" and self.freq_view_center is not None:
                display_freq = freq + self.freq_view_center
            label.setVisible(True)
            label.setText(f"peak {display_freq:,.2f} Hz")
            label.setPos(freq, mag * 0.7)

    def _update_peak_visibility(self) -> None:
        if self.view_domain != "frequency":
            for labels in self._peak_labels.values():
                for label in labels:
                    label.setVisible(False)

    def _find_peaks(
        self,
        x_vals: NDArray[np.float_],
        y_vals: NDArray[np.float_],
        count: int,
    ) -> list[tuple[float, float]]:
        if y_vals.size == 0 or count <= 0:
            return []
        if y_vals.size < 3:
            idx = int(np.argmax(y_vals))
            return [(float(x_vals[idx]), float(y_vals[idx]))]
        peak_mask = (y_vals[1:-1] > y_vals[:-2]) & (y_vals[1:-1] >= y_vals[2:])
        peak_indices = np.where(peak_mask)[0] + 1
        if peak_indices.size == 0:
            idx = int(np.argmax(y_vals))
            return [(float(x_vals[idx]), float(y_vals[idx]))]
        ordered = peak_indices[np.argsort(y_vals[peak_indices])[::-1]]
        top = ordered[:count]
        return [(float(x_vals[i]), float(y_vals[i])) for i in top]

    # region Lifecycle ----------------------------------------------------
    def exec(self) -> None:
        """Enter the Qt event loop to keep the window open."""
        if self._owns_app:
            self._app.exec()
        else:
            # If we don't own the app, ensure events keep flowing.
            self._process_events()

    def close(self) -> None:
        """Close the observer window."""
        self._window.close()
        self._process_events()

    # endregion -----------------------------------------------------------
