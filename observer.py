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
        max_points: int = 5000,
        time_scale: float = 1e3,
        view_domain: Literal["time", "frequency"] = "time",
        freq_view_max: float | None = 1e6,
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

        self.show_tx = show_tx
        self.show_adc = show_adc
        self.show_rx = show_rx
        self.max_points = max_points
        self.time_scale = time_scale
        self.view_domain = view_domain
        self.freq_view_max = freq_view_max
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
        self._peak_labels: Dict[str, "pyqtgraph.TextItem"] = {}
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle(window_title)
        self._build_ui()
        self._active = False
        self._paused = False
        self._resume_event = threading.Event()
        self._restart_event = threading.Event()

        if not any((show_tx, show_adc, show_rx)):
            self.show_tx = True

        self._window.show()
        self._process_events()

    # region UI setup -----------------------------------------------------
    def _build_ui(self) -> None:
        QtWidgets = self.QtWidgets
        pg = self.pg

        central = QtWidgets.QWidget()
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
            label = pg.TextItem(color=(255, 180, 0), anchor=(0, 1))
            label.setVisible(False)
            plot.addItem(label)
            self._peak_labels[key] = label
            if idx < len(keys) - 1:
                self._plot_container.nextRow()

        self._plot_container.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._window.keyPressEvent = self._key_press  # type: ignore

        for plot in list(self._plots.values())[1:]:
            plot.setXLink(list(self._plots.values())[0])
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
        self._update_plot("rx", outputs.time, outputs.processed)

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

        x_vals, y_vals = self._transform_data(time, samples)

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
        for plot in self._plots.values():
            if self.view_domain == "frequency" and self.freq_view_max:
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
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        if self.view_domain == "frequency":
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
        fft_vals = np.fft.rfft(samples)
        freqs = np.fft.rfftfreq(len(samples), d=spacing)
        magnitudes = np.abs(fft_vals)
        if self.freq_view_max is not None:
            mask = freqs <= self.freq_view_max
            if not np.any(mask):
                mask = np.ones_like(freqs, dtype=bool)
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
        label = self._peak_labels.get(key)
        if label is None:
            return
        if self.view_domain != "frequency" or len(x_vals) == 0:
            label.setVisible(False)
            return
        idx = int(np.argmax(y_vals))
        freq = float(x_vals[idx])
        mag = float(y_vals[idx])
        label.setVisible(True)
        label.setText(f"peak {freq:,.0f} Hz")
        label.setPos(freq, mag * 0.7)

    def _update_peak_visibility(self) -> None:
        if self.view_domain != "frequency":
            for label in self._peak_labels.values():
                label.setVisible(False)

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
