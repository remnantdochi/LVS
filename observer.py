"""Observer interfaces for tapping into the simulation pipeline."""
from __future__ import annotations

import sys
from typing import Dict, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from lvs_rx import ReceiverOutputs


class Observer(Protocol):
    """Interface for receiving callbacks from the simulation engine."""

    def on_tx(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None:  # pragma: no cover - interface definition
        ...

    def on_adc(
        self,
        time: NDArray[np.float_],
        samples: NDArray[np.float_],
    ) -> None:  # pragma: no cover - interface definition
        ...

    def on_rx(self, outputs: ReceiverOutputs) -> None:  # pragma: no cover
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
        self.plot_height = plot_height
        self.row_spacing = row_spacing

        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication(sys.argv or ["lvs-sim"])
            self._owns_app = True
        else:
            self._owns_app = False

        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle(window_title)
        self._build_ui()
        self._paused = False
        self._last_data: Dict[str, NDArray[np.float_]] = {}
        self._last_time: Dict[str, NDArray[np.float_]] = {}

        if not any((show_tx, show_adc, show_rx)):
            self.show_rx = True

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

        self._pause_action = self._toolbar.addAction("Pause")
        self._pause_action.setCheckable(True)
        self._pause_action.toggled.connect(self._toggle_pause)

        self._toolbar.addSeparator()
        self._toolbar.addAction("Clear", self._clear_plots)

        self._toolbar.addSeparator()
        self._toolbar.addAction("Measure TX Period", lambda: self._measure_period("tx"))
        self._toolbar.addAction("Measure ADC Period", lambda: self._measure_period("adc"))

        self._status_label = QtWidgets.QLabel("Hover a plot to read time/value.")
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
        if not keys:
            keys = ("rx",)

        for idx, key in enumerate(keys):
            plot = self._plot_container.addPlot(title=key.upper())
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setMouseEnabled(x=True, y=False)
            plot.setLabel("bottom", "Time", units=self._time_unit())
            plot.setLabel("left", "Amplitude")
            plot.setMinimumHeight(self.plot_height)
            curve = plot.plot([], [], pen=pg.mkPen(width=1.2), symbol="o", symbolSize=3)
            self._plots[key] = plot
            self._curves[key] = curve
            if idx < len(keys) - 1:
                self._plot_container.nextRow()

        self._plot_container.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._window.keyPressEvent = self._key_press  # type: ignore

        for plot in list(self._plots.values())[1:]:
            plot.setXLink(list(self._plots.values())[0])

    # endregion -----------------------------------------------------------

    # region Qt helpers ---------------------------------------------------
    def _process_events(self) -> None:
        self._app.processEvents(self.QtCore.QEventLoop.AllEvents, 50)

    def _key_press(self, event):  # pragma: no cover - GUI callback
        if event.key() == self.QtCore.Qt.Key_Space:
            self._pause_action.toggle()
        else:
            self.QtWidgets.QMainWindow.keyPressEvent(self._window, event)

    def _toggle_pause(self, checked: bool) -> None:
        self._paused = checked
        state = "paused" if checked else "running"
        self._status_label.setText(f"Streaming {state}. (Space toggles pause)")

    def _clear_plots(self) -> None:
        for curve in self._curves.values():
            curve.setData([], [])
        self._last_data.clear()
        self._last_time.clear()
        self._status_label.setText("Plots cleared.")
        self._process_events()

    def _time_unit(self) -> str:
        if self.time_scale == 1:
            return "s"
        if self.time_scale == 1e3:
            return "ms"
        if self.time_scale == 1e6:
            return "µs"
        return "scaled"

    def _on_mouse_moved(self, pos):  # pragma: no cover - GUI feedback
        for key, plot in self._plots.items():
            if plot.sceneBoundingRect().contains(pos):
                vb = plot.getViewBox()
                mouse_point = vb.mapSceneToView(pos)
                time_value = mouse_point.x() / self.time_scale
                amp_value = mouse_point.y()
                self._status_label.setText(
                    f"{key.upper()}  t={time_value:.6f}s  value={amp_value:.6f}"
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
        if self._paused:
            return
        curve = self._curves.get(key)
        if curve is None:
            return

        if len(time) > self.max_points:
            idx = np.linspace(0, len(time) - 1, self.max_points).astype(int)
            time = time[idx]
            samples = samples[idx]

        curve.setData(time * self.time_scale, samples)
        self._last_time[key] = time
        self._last_data[key] = samples
        self._process_events()

    def _measure_period(self, key: str, level: float = 0.0):
        import numpy as np
        time = self._last_time.get(key)
        data = self._last_data.get(key)
        if time is None or data is None:
            self._status_label.setText(f"No {key.upper()} data to measure.")
            return

        sign_changes = np.where(np.diff(np.signbit(data - level)))[0]
        if len(sign_changes) < 2:
            self._status_label.setText(f"{key.upper()}: Not enough level crossings.")
            return

        # Compute precise crossing times by linear interpolation
        t_cross = []
        for i in sign_changes:
            t1, t2 = time[i], time[i + 1]
            y1, y2 = data[i], data[i + 1]
            if y2 == y1:
                continue
            t_cross.append(t1 + (level - y1) * (t2 - t1) / (y2 - y1))

        if len(t_cross) < 2:
            self._status_label.setText(f"{key.upper()}: Not enough crossings for measurement.")
            return

        diffs = np.diff(t_cross)
        avg = np.mean(diffs)
        freq = 1.0 / (2 * avg) if avg > 0 else 0.0
        self._status_label.setText(f"{key.upper()} mean Δt={avg*1e6:.3f} µs  ({freq:.1f} Hz)")

    # region Lifecycle ----------------------------------------------------
    def exec(self) -> None:
        """Enter the Qt event loop to keep the window open."""
        if self._owns_app:  # pragma: no cover - GUI only
            self._app.exec()
        else:
            # If we don't own the app, ensure events keep flowing.
            self._process_events()

    def close(self) -> None:
        """Close the observer window."""
        self._window.close()
        self._process_events()

    # endregion -----------------------------------------------------------
