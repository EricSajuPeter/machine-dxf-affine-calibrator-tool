from __future__ import annotations

import csv
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from affine_core import (
    AffineResult,
    apply_inverse_transform,
    apply_transform,
    build_affine_result,
    build_compensation_transform,
    compose_user_adjustment,
    decompose_affine,
    extract_dxf_paths_for_plot,
    transform_dxf_with_compensation,
)

Point = Tuple[float, float]


def _attach_plot_scroll_pan(canvas: object, ax: object, toolbar: Optional[object] = None) -> None:
    """Scroll wheel zooms around cursor; left-drag pans. For Matplotlib Qt canvas + axes."""
    # Pan uses figure pixel deltas (event.x/y), not data coords — avoids feedback jitter when
    # limits change each frame (data-space deltas were inconsistent with the new transform).
    _pan: dict[str, object] = {"active": False, "last_px": None}

    def _toolbar_busy() -> bool:
        if toolbar is None:
            return False
        mode = getattr(toolbar, "mode", "") or getattr(toolbar, "_active", "")
        return bool(mode)

    def on_scroll(event: object) -> None:
        if _toolbar_busy():
            return
        if getattr(event, "inaxes", None) is not ax or event.xdata is None or event.ydata is None:
            return
        step = getattr(event, "step", None)
        if step is not None and step != 0:
            factor = 0.9 if step > 0 else 1.1
        elif getattr(event, "button", None) == "up":
            factor = 0.9
        elif getattr(event, "button", None) == "down":
            factor = 1.1
        else:
            return
        xdata, ydata = float(event.xdata), float(event.ydata)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        w = xlim[1] - xlim[0]
        h = ylim[1] - ylim[0]
        if w <= 0 or h <= 0:
            return
        new_w = w * factor
        new_h = h * factor
        relx = (xdata - xlim[0]) / w
        rely = (ydata - ylim[0]) / h
        ax.set_xlim(xdata - relx * new_w, xdata + (1.0 - relx) * new_w)
        ax.set_ylim(ydata - rely * new_h, ydata + (1.0 - rely) * new_h)
        canvas.draw_idle()

    def on_press(event: object) -> None:
        if _toolbar_busy():
            return
        if getattr(event, "inaxes", None) is not ax:
            return
        if getattr(event, "button", None) != 1:
            return
        if event.x is None or event.y is None:
            return
        _pan["active"] = True
        _pan["last_px"] = (float(event.x), float(event.y))

    def on_motion(event: object) -> None:
        if _toolbar_busy():
            return
        if not _pan["active"] or _pan["last_px"] is None:
            return
        if event.x is None or event.y is None:
            return
        lx, ly = _pan["last_px"]
        dx_px = float(event.x) - lx
        dy_px = float(event.y) - ly
        _pan["last_px"] = (float(event.x), float(event.y))
        if dx_px == 0.0 and dy_px == 0.0:
            return
        bbox = ax.bbox
        bw = float(bbox.width)
        bh = float(bbox.height)
        if bw < 2.0 or bh < 2.0:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xd = xlim[1] - xlim[0]
        yd = ylim[1] - ylim[0]
        # Figure/canvas pixel motion -> data shift (same convention as Matplotlib's pan tool).
        dx_data = -(dx_px / bw) * xd
        dy_data = -(dy_px / bh) * yd
        ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
        ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
        canvas.draw_idle()

    def on_release(event: object) -> None:
        if getattr(event, "button", None) != 1:
            return
        _pan["active"] = False
        _pan["last_px"] = None

    canvas.mpl_connect("scroll_event", on_scroll)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("button_release_event", on_release)


# Keys for full-view plot layer toggles (maximized dialog only).
PLOT_LAYER_IDEAL = "ideal"
PLOT_LAYER_MEASURED = "measured"
PLOT_LAYER_COMPENSATED = "compensated"
PLOT_LAYER_PREDICTED_PRINT = "predicted_print"
PLOT_LAYER_COORDINATES = "coordinates"


def _default_plot_visibility() -> Dict[str, bool]:
    return {
        PLOT_LAYER_IDEAL: True,
        PLOT_LAYER_MEASURED: True,
        PLOT_LAYER_COMPENSATED: True,
        PLOT_LAYER_PREDICTED_PRINT: True,
        PLOT_LAYER_COORDINATES: False,
    }


def _style_coord_field(edit: QLineEdit) -> None:
    """Keep numeric fields readable at any window width (avoid huge stretched boxes)."""
    edit.setMinimumWidth(100)
    edit.setMaximumWidth(220)
    edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


def _apply_affine_to_paths(
    paths: List[np.ndarray], a_mat: np.ndarray, t_vec: np.ndarray
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for arr in paths:
        if arr.size == 0:
            continue
        transformed = (a_mat @ arr.T).T + t_vec
        if transformed.shape[0] < 2:
            continue
        out.append(np.asarray(transformed, dtype=float))
    return out


class PairCardWidget(QFrame):
    def __init__(self, index: int, validator: QDoubleValidator) -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet("QFrame { border: 1px solid #c9c9c9; border-radius: 6px; padding: 4px; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title = QLabel(f"Pair {index}")
        self.title.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.title)

        self.ideal_x = QLineEdit()
        self.ideal_y = QLineEdit()
        self.measured_x = QLineEdit()
        self.measured_y = QLineEdit()
        for edit in (self.ideal_x, self.ideal_y, self.measured_x, self.measured_y):
            edit.setValidator(validator)
            edit.setPlaceholderText("0.0")
            _style_coord_field(edit)

        ideal_row = QHBoxLayout()
        ideal_row.setSpacing(10)
        ideal_row.addWidget(QLabel("Ideal X"))
        ideal_row.addWidget(self.ideal_x)
        ideal_row.addSpacing(12)
        ideal_row.addWidget(QLabel("Ideal Y"))
        ideal_row.addWidget(self.ideal_y)
        ideal_row.addStretch(1)
        layout.addLayout(ideal_row)

        measured_row = QHBoxLayout()
        measured_row.setSpacing(10)
        measured_row.addWidget(QLabel("Measured X"))
        measured_row.addWidget(self.measured_x)
        measured_row.addSpacing(12)
        measured_row.addWidget(QLabel("Measured Y"))
        measured_row.addWidget(self.measured_y)
        measured_row.addStretch(1)
        layout.addLayout(measured_row)

    def connect_plot_updates(self, callback) -> None:
        for w in (self.ideal_x, self.ideal_y, self.measured_x, self.measured_y):
            w.textChanged.connect(callback)

    def set_index(self, index: int) -> None:
        self.title.setText(f"Pair {index}")

    def values(self) -> Tuple[str, str, str, str]:
        return (
            self.ideal_x.text().strip(),
            self.ideal_y.text().strip(),
            self.measured_x.text().strip(),
            self.measured_y.text().strip(),
        )


class CalibrationPlotDialog(QDialog):
    """Fullscreen-style plot with labels, legend, zoom/pan toolbar, and Close."""

    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__(main_window)
        self._main = main_window
        self.setWindowTitle("Calibration preview")
        self.setModal(False)
        self._adj_group: Optional[QGroupBox] = None
        self._adj_value_edits: Dict[str, Tuple[QLineEdit, str]] = {}
        self._reference_group: Optional[QGroupBox] = None
        self._reference_point: Optional[Point] = None
        self._dimension_mode = False
        self._pickable_points: List[Dict[str, Any]] = []
        self._dimension_records: List[Dict[str, Any]] = []
        self._hover_pick: Optional[Dict[str, Any]] = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_v = QVBoxLayout(left_panel)
        left_v.setSpacing(10)

        layers_box = QGroupBox("Layers")
        layers_l = QVBoxLayout(layers_box)
        self._chk_ideal = QCheckBox("Ideal")
        self._chk_ideal.setChecked(True)
        self._chk_ideal.setToolTip("Computer / CAD polyline (blue)")
        self._chk_measured = QCheckBox("Measured")
        self._chk_measured.setChecked(True)
        self._chk_measured.setToolTip("As-measured polyline (red)")
        self._chk_comp = QCheckBox("Compensated")
        self._chk_comp.setChecked(True)
        self._chk_comp.setToolTip("Inverse-affine corrected path for machine feed (green)")
        self._chk_predicted_print = QCheckBox("Predicted print")
        self._chk_predicted_print.setChecked(True)
        self._chk_predicted_print.setToolTip(
            "Purple: for each vertex, the midpoint between as-measured (red) and compensated "
            "machine feed (green): (measured + compensated) / 2."
        )
        self._chk_coordinates = QCheckBox("Coordinates")
        self._chk_coordinates.setChecked(False)
        self._chk_coordinates.setToolTip(
            "Show bracketed coordinate labels next to plotted points."
        )
        for cb in (
            self._chk_ideal,
            self._chk_measured,
            self._chk_comp,
            self._chk_predicted_print,
            self._chk_coordinates,
        ):
            cb.stateChanged.connect(lambda _s: self.refresh(preserve_view=True))
            layers_l.addWidget(cb)
        left_v.addWidget(layers_box)

        self._adj_group = self._build_adjustment_panel()
        left_v.addWidget(self._adj_group)
        self._reference_group = self._build_reference_panel()
        left_v.addWidget(self._reference_group)

        left_v.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.setToolTip("Return to normal view")
        btn_close.clicked.connect(self.close)
        left_v.addWidget(btn_close)

        outer.addWidget(left_panel)

        right_panel = QWidget()
        right_l = QVBoxLayout(right_panel)
        right_l.setContentsMargins(0, 0, 0, 0)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            self._figure = Figure(figsize=(10, 8))
            self._ax = self._figure.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            right_l.addWidget(self._toolbar)
            right_l.addWidget(self._canvas, stretch=1)
            outer.addWidget(right_panel, stretch=1)
            _attach_plot_scroll_pan(self._canvas, self._ax, self._toolbar)
            self._canvas.mpl_connect("button_press_event", self._on_plot_pick)
            self._canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
        except Exception as exc:
            self._figure = None
            self._ax = None
            self._canvas = None
            self._toolbar = None
            self._chk_ideal.setEnabled(False)
            self._chk_measured.setEnabled(False)
            self._chk_comp.setEnabled(False)
            self._chk_predicted_print.setEnabled(False)
            self._chk_coordinates.setEnabled(False)
            if self._reference_group is not None:
                self._reference_group.setEnabled(False)
            right_l.addWidget(QLabel(f"Plot tools unavailable: {exc}"))
            outer.addWidget(right_panel, stretch=1)

    def _build_adjustment_panel(self) -> QGroupBox:
        box = QGroupBox("Manual corrections (additive to solved transform)")
        outer = QVBoxLayout(box)
        main = self._main
        vali = main._validator
        specs: List[Tuple[str, str, float, str]] = [
            ("Translation ΔX", "_adj_dtx", 0.05, "%+.4f"),
            ("Translation ΔY", "_adj_dty", 0.05, "%+.4f"),
            ("Rotation Δ (deg)", "_adj_drot", 0.1, "%+.4f"),
            ("Scale ΔX", "_adj_dsx", 0.001, "%+.6f"),
            ("Scale ΔY", "_adj_dsy", 0.001, "%+.6f"),
            ("Shear Δ", "_adj_dshear", 0.001, "%+.6f"),
        ]
        for label_text, attr, step, fmt in specs:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            btn_m = QPushButton("−")
            btn_p = QPushButton("+")
            btn_m.setFixedWidth(36)
            btn_p.setFixedWidth(36)
            # Single click: one step; hold: repeat until release (Qt auto-repeat).
            for b in (btn_m, btn_p):
                b.setAutoRepeat(True)
                b.setAutoRepeatDelay(400)
                b.setAutoRepeatInterval(75)
            edit = QLineEdit()
            edit.setValidator(vali)
            edit.setText(fmt % getattr(main, attr))
            edit.setMinimumWidth(108)
            edit.setMaximumWidth(140)
            edit.setAlignment(Qt.AlignmentFlag.AlignRight)

            def commit(attr_name: str = attr, f: str = fmt, ed: QLineEdit = edit) -> None:
                raw = ed.text().strip().replace(",", ".")
                if raw == "":
                    self._sync_one_adjustment(attr_name, ed, f)
                    return
                try:
                    v = float(raw)
                except ValueError:
                    self._sync_one_adjustment(attr_name, ed, f)
                    return
                setattr(main, attr_name, v)
                ed.blockSignals(True)
                ed.setText(f % getattr(main, attr_name))
                ed.blockSignals(False)
                main.update_shape_plot()

            def bump(sign: int, attr_name: str = attr, st: float = step, f: str = fmt, ed: QLineEdit = edit) -> None:
                cur = getattr(main, attr_name)
                setattr(main, attr_name, cur + sign * st)
                ed.blockSignals(True)
                ed.setText(f % getattr(main, attr_name))
                ed.blockSignals(False)
                main.update_shape_plot()

            edit.editingFinished.connect(commit)
            btn_m.clicked.connect(partial(bump, -1))
            btn_p.clicked.connect(partial(bump, 1))
            row.addWidget(btn_m)
            row.addWidget(edit, stretch=1)
            row.addWidget(btn_p)
            outer.addLayout(row)
            self._adj_value_edits[attr] = (edit, fmt)

        reset_row = QHBoxLayout()
        btn_reset = QPushButton("Reset all corrections")
        btn_reset.setToolTip("Clear all manual deltas (does not change the last Solve result)")
        btn_reset.clicked.connect(self._on_reset_corrections)
        reset_row.addWidget(btn_reset)
        reset_row.addStretch(1)
        outer.addLayout(reset_row)
        return box

    def _sync_one_adjustment(self, attr: str, edit: QLineEdit, fmt: str) -> None:
        edit.blockSignals(True)
        edit.setText(fmt % getattr(self._main, attr))
        edit.blockSignals(False)

    def _on_reset_corrections(self) -> None:
        self._main._reset_user_adjustments()
        self._sync_adjustment_labels()
        self._main.update_shape_plot()

    def _sync_adjustment_labels(self) -> None:
        for attr, (edit, fmt) in self._adj_value_edits.items():
            self._sync_one_adjustment(attr, edit, fmt)

    def _build_reference_panel(self) -> QGroupBox:
        box = QGroupBox("Reference point")
        root = QVBoxLayout(box)
        root.setSpacing(8)
        row = QHBoxLayout()
        row.addWidget(QLabel("X"))
        self._ref_x = QLineEdit()
        self._ref_x.setValidator(self._main._validator)
        self._ref_x.setPlaceholderText("0.0")
        row.addWidget(self._ref_x)
        row.addWidget(QLabel("Y"))
        self._ref_y = QLineEdit()
        self._ref_y.setValidator(self._main._validator)
        self._ref_y.setPlaceholderText("0.0")
        row.addWidget(self._ref_y)
        root.addLayout(row)

        btn_grid = QGridLayout()
        btn_grid.setHorizontalSpacing(6)
        btn_grid.setVerticalSpacing(6)
        self._btn_set_reference = QPushButton("Set reference")
        self._btn_set_reference.clicked.connect(self._on_set_reference)
        self._btn_add_dimensions = QPushButton("Add dimensions")
        self._btn_add_dimensions.setCheckable(True)
        self._btn_add_dimensions.setEnabled(False)
        self._btn_add_dimensions.clicked.connect(self._on_toggle_dimensions)
        self._btn_clear_reference = QPushButton("Clear reference")
        self._btn_clear_reference.clicked.connect(self._on_clear_reference)
        self._btn_clear_dimensions = QPushButton("Clear dimensions")
        self._btn_clear_dimensions.clicked.connect(self._on_clear_dimensions)
        btn_grid.addWidget(self._btn_set_reference, 0, 0)
        btn_grid.addWidget(self._btn_add_dimensions, 0, 1)
        btn_grid.addWidget(self._btn_clear_reference, 1, 0)
        btn_grid.addWidget(self._btn_clear_dimensions, 1, 1)
        root.addLayout(btn_grid)

        self._dimension_log = QPlainTextEdit()
        self._dimension_log.setReadOnly(True)
        self._dimension_log.setPlaceholderText("Dimension picks will appear here.")
        self._dimension_log.setMaximumBlockCount(400)
        self._dimension_log.setMinimumHeight(100)
        root.addWidget(self._dimension_log)
        return box

    def _on_set_reference(self) -> None:
        sx = self._ref_x.text().strip()
        sy = self._ref_y.text().strip()
        if not sx or not sy:
            QMessageBox.warning(self, "Reference point", "Provide both reference X and Y.")
            return
        try:
            x = float(sx)
            y = float(sy)
        except ValueError:
            QMessageBox.warning(self, "Reference point", "Reference coordinates must be numeric.")
            return
        self._reference_point = (x, y)
        self._dimension_records.clear()
        self._hover_pick = None
        self._btn_add_dimensions.setEnabled(True)
        self._append_dimension_text(f"Reference set to [{x:.4f}, {y:.4f}]")
        self.refresh()

    def _on_toggle_dimensions(self, checked: bool) -> None:
        if checked and self._reference_point is None:
            self._btn_add_dimensions.setChecked(False)
            self._btn_add_dimensions.setEnabled(False)
            return
        self._dimension_mode = checked
        if not checked:
            self._hover_pick = None
        self._btn_add_dimensions.setText("Stop adding" if checked else "Add dimensions")
        if checked:
            self._append_dimension_text("Dimension pick mode ON. Click a plotted point.")
        else:
            self._append_dimension_text("Dimension pick mode OFF.")
        self.refresh(preserve_view=True)

    def _on_clear_dimensions(self) -> None:
        self._dimension_records.clear()
        self._hover_pick = None
        self._dimension_log.clear()
        self.refresh(preserve_view=True)

    def _on_clear_reference(self) -> None:
        self._reference_point = None
        self._dimension_mode = False
        self._dimension_records.clear()
        self._hover_pick = None
        self._dimension_log.clear()
        self._ref_x.clear()
        self._ref_y.clear()
        self._btn_add_dimensions.setChecked(False)
        self._btn_add_dimensions.setText("Add dimensions")
        self._btn_add_dimensions.setEnabled(False)
        self.refresh(preserve_view=True)

    def clear_dimension_state(self) -> None:
        self._dimension_mode = False
        self._dimension_records.clear()
        self._pickable_points.clear()
        self._hover_pick = None
        self._dimension_log.clear()
        if hasattr(self, "_btn_add_dimensions"):
            self._btn_add_dimensions.setChecked(False)
            self._btn_add_dimensions.setText("Add dimensions")
            self._btn_add_dimensions.setEnabled(self._reference_point is not None)
        self.refresh()

    def _append_dimension_text(self, text: str) -> None:
        self._dimension_log.appendPlainText(text)

    @staticmethod
    def _pick_distance_threshold(ax: object) -> float:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        span = max(float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]))
        return max(1e-9, span * 0.03)

    def _nearest_pick(self, xdata: float, ydata: float) -> Optional[Dict[str, Any]]:
        if self._ax is None or not self._pickable_points:
            return None
        best: Optional[Dict[str, Any]] = None
        best_d2 = float("inf")
        for item in self._pickable_points:
            px, py = item["point"]
            d2 = (xdata - px) ** 2 + (ydata - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = item
        if best is None:
            return None
        threshold = self._pick_distance_threshold(self._ax)
        if (best_d2 ** 0.5) > threshold:
            return None
        return best

    def _on_plot_hover(self, event: object) -> None:
        if self._ax is None:
            return
        if not self._dimension_mode or self._reference_point is None:
            if self._hover_pick is not None:
                self._hover_pick = None
                self.refresh(preserve_view=True)
            return
        if getattr(event, "inaxes", None) is not self._ax:
            if self._hover_pick is not None:
                self._hover_pick = None
                self.refresh(preserve_view=True)
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        candidate = self._nearest_pick(float(event.xdata), float(event.ydata))
        old_label = self._hover_pick["label"] if self._hover_pick is not None else None
        new_label = candidate["label"] if candidate is not None else None
        if old_label == new_label:
            return
        self._hover_pick = candidate
        self.refresh(preserve_view=True)

    def _on_plot_pick(self, event: object) -> None:
        if self._ax is None or self._canvas is None:
            return
        if not self._dimension_mode or self._reference_point is None:
            return
        if getattr(event, "inaxes", None) is not self._ax:
            return
        if getattr(event, "button", None) != 1:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        best = self._nearest_pick(float(event.xdata), float(event.ydata))
        if best is None:
            return

        px, py = best["point"]
        rx, ry = self._reference_point
        dx = px - rx
        dy = py - ry
        self._dimension_records.append(
            {
                "label": best["label"],
                "point": (px, py),
                "reference": (rx, ry),
                "dx": dx,
                "dy": dy,
            }
        )
        idx = len(self._dimension_records)
        self._append_dimension_text(
            f"#{idx} {best['label']} @ [{px:.4f}, {py:.4f}] -> ΔX={dx:+.4f}, ΔY={dy:+.4f}"
        )
        self.refresh()

    def _visibility(self) -> Dict[str, bool]:
        return {
            PLOT_LAYER_IDEAL: self._chk_ideal.isChecked(),
            PLOT_LAYER_MEASURED: self._chk_measured.isChecked(),
            PLOT_LAYER_COMPENSATED: self._chk_comp.isChecked(),
            PLOT_LAYER_PREDICTED_PRINT: self._chk_predicted_print.isChecked(),
            PLOT_LAYER_COORDINATES: self._chk_coordinates.isChecked(),
        }

    def refresh(self, *, preserve_view: bool = False) -> None:
        self._sync_adjustment_labels()
        if self._adj_group is not None:
            self._adj_group.setEnabled(self._main.result is not None)
        if self._reference_group is not None:
            self._reference_group.setEnabled(self._main.result is not None)
        if self._ax is None or self._canvas is None or self._figure is None:
            return
        old_xlim = self._ax.get_xlim() if preserve_view else None
        old_ylim = self._ax.get_ylim() if preserve_view else None
        pick_sink: List[Dict[str, Any]] = []
        self._main._render_calibration_axes(
            self._ax,
            minimal=False,
            visibility=self._visibility(),
            pick_sink=pick_sink,
            reference_point=self._reference_point,
            dimension_records=self._dimension_records,
            hover_pick=self._hover_pick,
            preserve_view=preserve_view,
        )
        self._pickable_points = pick_sink
        if preserve_view and old_xlim is not None and old_ylim is not None:
            self._ax.set_xlim(old_xlim)
            self._ax.set_ylim(old_ylim)
        elif self._toolbar is not None:
            # Keep toolbar "Home" synced to the latest full-view frame (including reference).
            try:
                nav = getattr(self._toolbar, "_nav_stack", None)
                if nav is not None:
                    nav.clear()
                    self._toolbar.push_current()
                    self._toolbar.set_history_buttons()
            except Exception:
                pass
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def closeEvent(self, event) -> None:
        self._main._plot_dialog = None
        super().closeEvent(event)


class DxfCompareDialog(QDialog):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__(main_window)
        self._main = main_window
        self.setWindowTitle("DXF comparison")
        self.setModal(False)
        self._input_paths: List[np.ndarray] = []
        self._output_paths: List[np.ndarray] = []
        self._distorted_paths: List[np.ndarray] = []
        self._measure_mode = False
        self._measure_pending_point: Optional[Dict[str, Any]] = None
        self._measure_records: List[Dict[str, Any]] = []
        self._reference_point: Optional[Point] = None
        self._dimension_mode = False
        self._dimension_records: List[Dict[str, Any]] = []
        self._pickable_points: List[Dict[str, Any]] = []
        self._hover_pick: Optional[Dict[str, Any]] = None
        self._bbox_enabled = False
        self._bbox_layer = "Input"

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        left = QWidget()
        left.setFixedWidth(260)
        left_v = QVBoxLayout(left)
        box = QGroupBox("Layers")
        box_l = QVBoxLayout(box)
        self._chk_input = QCheckBox("Input DXF")
        self._chk_input.setChecked(True)
        self._chk_output = QCheckBox("Output DXF")
        self._chk_output.setChecked(True)
        self._chk_distorted = QCheckBox("Distorted (input→machine)")
        self._chk_distorted.setChecked(True)
        self._chk_coords = QCheckBox("Coordinates")
        self._chk_coords.setChecked(False)
        for cb in (self._chk_input, self._chk_output, self._chk_distorted, self._chk_coords):
            cb.stateChanged.connect(lambda _s: self.refresh(preserve_view=True))
            box_l.addWidget(cb)
        left_v.addWidget(box)

        bbox_box = QGroupBox("Bounding box")
        bbox_l = QVBoxLayout(bbox_box)
        self._chk_bbox = QCheckBox("Show bounding box")
        self._chk_bbox.setChecked(False)
        self._chk_bbox.stateChanged.connect(self._on_bbox_controls_changed)
        bbox_l.addWidget(self._chk_bbox)
        bbox_row = QHBoxLayout()
        bbox_row.addWidget(QLabel("Layer"))
        self._cmb_bbox_layer = QComboBox()
        self._cmb_bbox_layer.addItems(["Input", "Output", "Distorted"])
        self._cmb_bbox_layer.currentIndexChanged.connect(self._on_bbox_controls_changed)
        bbox_row.addWidget(self._cmb_bbox_layer, stretch=1)
        bbox_l.addLayout(bbox_row)
        left_v.addWidget(bbox_box)

        measure_box = QGroupBox("Measure")
        measure_l = QVBoxLayout(measure_box)
        btn_row = QHBoxLayout()
        self._btn_measure = QPushButton("Measure DX/DY")
        self._btn_measure.setCheckable(True)
        self._btn_measure.clicked.connect(self._on_toggle_measure)
        self._btn_clear_measure = QPushButton("Clear measurements")
        self._btn_clear_measure.clicked.connect(self._on_clear_measurements)
        btn_row.addWidget(self._btn_measure)
        btn_row.addWidget(self._btn_clear_measure)
        measure_l.addLayout(btn_row)
        self._measure_log = QPlainTextEdit()
        self._measure_log.setReadOnly(True)
        self._measure_log.setPlaceholderText("Click point A then point B to measure DX/DY.")
        self._measure_log.setMaximumBlockCount(500)
        self._measure_log.setMinimumHeight(120)
        measure_l.addWidget(self._measure_log)
        left_v.addWidget(measure_box)

        ref_box = QGroupBox("Reference point")
        ref_l = QVBoxLayout(ref_box)
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("X"))
        self._ref_x = QLineEdit()
        self._ref_x.setValidator(self._main._validator)
        self._ref_x.setPlaceholderText("0.0")
        ref_row.addWidget(self._ref_x)
        ref_row.addWidget(QLabel("Y"))
        self._ref_y = QLineEdit()
        self._ref_y.setValidator(self._main._validator)
        self._ref_y.setPlaceholderText("0.0")
        ref_row.addWidget(self._ref_y)
        ref_l.addLayout(ref_row)

        ref_btn_grid = QGridLayout()
        ref_btn_grid.setHorizontalSpacing(6)
        ref_btn_grid.setVerticalSpacing(6)
        self._btn_set_reference = QPushButton("Set reference")
        self._btn_set_reference.clicked.connect(self._on_set_reference)
        self._btn_add_dimensions = QPushButton("Add dimensions")
        self._btn_add_dimensions.setCheckable(True)
        self._btn_add_dimensions.setEnabled(False)
        self._btn_add_dimensions.clicked.connect(self._on_toggle_dimensions)
        self._btn_clear_reference = QPushButton("Clear reference")
        self._btn_clear_reference.clicked.connect(self._on_clear_reference)
        self._btn_clear_dimensions = QPushButton("Clear dimensions")
        self._btn_clear_dimensions.clicked.connect(self._on_clear_dimensions)
        ref_btn_grid.addWidget(self._btn_set_reference, 0, 0)
        ref_btn_grid.addWidget(self._btn_add_dimensions, 0, 1)
        ref_btn_grid.addWidget(self._btn_clear_reference, 1, 0)
        ref_btn_grid.addWidget(self._btn_clear_dimensions, 1, 1)
        ref_l.addLayout(ref_btn_grid)

        self._dimension_log = QPlainTextEdit()
        self._dimension_log.setReadOnly(True)
        self._dimension_log.setPlaceholderText(
            "Reference dimensions will appear here."
        )
        self._dimension_log.setMaximumBlockCount(500)
        self._dimension_log.setMinimumHeight(100)
        ref_l.addWidget(self._dimension_log)
        left_v.addWidget(ref_box)

        left_v.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        left_v.addWidget(btn_close)
        outer.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            self._figure = Figure(figsize=(10, 8))
            self._ax = self._figure.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            right_l.addWidget(self._toolbar)
            right_l.addWidget(self._canvas, stretch=1)
            _attach_plot_scroll_pan(self._canvas, self._ax, self._toolbar)
            self._canvas.mpl_connect("button_press_event", self._on_plot_click)
            self._canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
        except Exception as exc:
            self._figure = None
            self._ax = None
            self._canvas = None
            self._toolbar = None
            for cb in (self._chk_input, self._chk_output, self._chk_distorted, self._chk_coords, self._chk_bbox):
                cb.setEnabled(False)
            self._cmb_bbox_layer.setEnabled(False)
            self._btn_measure.setEnabled(False)
            self._btn_clear_measure.setEnabled(False)
            right_l.addWidget(QLabel(f"Plot unavailable: {exc}"))
        outer.addWidget(right, stretch=1)

    @staticmethod
    def _draw_paths(ax: object, paths: List[np.ndarray], *, color: str, label: str) -> bool:
        if not paths:
            return False
        first = True
        for arr in paths:
            if arr.shape[0] < 2:
                continue
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                color=color,
                linewidth=1.2,
                alpha=0.95,
                label=label if first else "_nolegend_",
            )
            first = False
        return not first

    @staticmethod
    def _fmt_xy(pt: Tuple[float, float]) -> str:
        return f"[{pt[0]:.3f}, {pt[1]:.3f}]"

    def _annotate_points(
        self, ax: object, points: List[Tuple[float, float]], *, color: str, prefix: str
    ) -> None:
        for p in points:
            ax.annotate(
                f"{prefix} {self._fmt_xy(p)}",
                xy=(p[0], p[1]),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                zorder=12,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                },
            )

    def _collect_pickables(
        self, paths: List[np.ndarray], *, source: str
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p_idx, arr in enumerate(paths):
            if arr.size == 0:
                continue
            for i, row in enumerate(arr):
                out.append(
                    {
                        "source": source,
                        "path_index": p_idx,
                        "point_index": i,
                        "point": (float(row[0]), float(row[1])),
                        "label": f"{source} p{p_idx+1}:{i+1}",
                    }
                )
        return out

    def _on_bbox_controls_changed(self, _value: object = None) -> None:
        self._bbox_enabled = self._chk_bbox.isChecked()
        self._bbox_layer = self._cmb_bbox_layer.currentText()
        self.refresh(preserve_view=True)

    def _get_layer_paths(self, layer_name: str) -> List[np.ndarray]:
        if layer_name == "Input":
            return self._input_paths
        if layer_name == "Output":
            return self._output_paths
        if layer_name == "Distorted":
            return self._distorted_paths
        return []

    def _is_layer_visible(self, layer_name: str) -> bool:
        if layer_name == "Input":
            return self._chk_input.isChecked()
        if layer_name == "Output":
            return self._chk_output.isChecked()
        if layer_name == "Distorted":
            return self._chk_distorted.isChecked()
        return False

    @staticmethod
    def _compute_bbox_from_paths(paths: List[np.ndarray]) -> Optional[Tuple[float, float, float, float]]:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for arr in paths:
            if arr.size == 0:
                continue
            xs.append(np.asarray(arr[:, 0], dtype=float))
            ys.append(np.asarray(arr[:, 1], dtype=float))
        if not xs or not ys:
            return None
        all_x = np.concatenate(xs)
        all_y = np.concatenate(ys)
        if all_x.size == 0 or all_y.size == 0:
            return None
        return (float(all_x.min()), float(all_x.max()), float(all_y.min()), float(all_y.max()))

    def _draw_bbox_overlay(
        self, ax: object, layer_name: str, bounds: Tuple[float, float, float, float]
    ) -> None:
        min_x, max_x, min_y, max_y = bounds
        corners = self._bbox_corners_from_bounds(bounds)
        poly_x = [min_x, min_x, max_x, max_x, min_x]
        poly_y = [min_y, max_y, max_y, min_y, min_y]
        ax.plot(
            poly_x,
            poly_y,
            linestyle="--",
            linewidth=1.3,
            color="#cc6a00",
            alpha=0.95,
            zorder=9,
            label="_nolegend_",
        )
        ax.scatter(
            [p[0] for p in corners],
            [p[1] for p in corners],
            c="#cc6a00",
            s=34,
            marker="s",
            zorder=10,
            label="_nolegend_",
        )
        if self._chk_coords.isChecked():
            for cx, cy in corners:
                ax.annotate(
                    f"{layer_name} {self._fmt_xy((cx, cy))}",
                    xy=(cx, cy),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="#8a4700",
                    zorder=12,
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "alpha": 0.74,
                        "edgecolor": "none",
                    },
                )

    @staticmethod
    def _bbox_corners_from_bounds(
        bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[float, float]]:
        min_x, max_x, min_y, max_y = bounds
        return [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
        ]

    def _collect_bbox_corner_pickables(
        self, layer_name: str, bounds: Tuple[float, float, float, float]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, (cx, cy) in enumerate(self._bbox_corners_from_bounds(bounds), start=1):
            out.append(
                {
                    "source": f"BBox {layer_name}",
                    "path_index": -1,
                    "point_index": i - 1,
                    "point": (cx, cy),
                    "label": f"BBox {layer_name} C{i}",
                }
            )
        return out

    @staticmethod
    def _pick_distance_threshold(ax: object) -> float:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        span = max(float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]))
        return max(1e-9, span * 0.03)

    def _nearest_pick(self, xdata: float, ydata: float) -> Optional[Dict[str, Any]]:
        if self._ax is None or not self._pickable_points:
            return None
        best: Optional[Dict[str, Any]] = None
        best_d2 = float("inf")
        for item in self._pickable_points:
            px, py = item["point"]
            d2 = (xdata - px) ** 2 + (ydata - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = item
        if best is None:
            return None
        if (best_d2**0.5) > self._pick_distance_threshold(self._ax):
            return None
        return best

    def _on_toggle_measure(self, checked: bool) -> None:
        self._measure_mode = checked
        self._measure_pending_point = None
        if checked and self._dimension_mode:
            self._dimension_mode = False
            self._btn_add_dimensions.blockSignals(True)
            self._btn_add_dimensions.setChecked(False)
            self._btn_add_dimensions.blockSignals(False)
            self._btn_add_dimensions.setText("Add dimensions")
            self._dimension_log.appendPlainText("Dimension pick mode OFF.")
        if checked:
            self._measure_log.appendPlainText("Measure mode ON. Click point A, then point B.")
            self._btn_measure.setText("Stop measure")
        else:
            self._measure_log.appendPlainText("Measure mode OFF.")
            self._btn_measure.setText("Measure DX/DY")
            self._hover_pick = None
        self.refresh(preserve_view=True)

    def _on_clear_measurements(self) -> None:
        self._measure_records.clear()
        self._measure_pending_point = None
        self._hover_pick = None
        self._measure_log.clear()
        self.refresh(preserve_view=True)

    def _append_dimension_text(self, text: str) -> None:
        self._dimension_log.appendPlainText(text)

    def _on_set_reference(self) -> None:
        sx = self._ref_x.text().strip()
        sy = self._ref_y.text().strip()
        if not sx or not sy:
            QMessageBox.warning(self, "Reference point", "Provide both reference X and Y.")
            return
        try:
            x = float(sx)
            y = float(sy)
        except ValueError:
            QMessageBox.warning(
                self, "Reference point", "Reference coordinates must be numeric."
            )
            return
        self._reference_point = (x, y)
        self._dimension_records.clear()
        self._hover_pick = None
        self._btn_add_dimensions.setEnabled(True)
        self._append_dimension_text(f"Reference set to [{x:.4f}, {y:.4f}]")
        self.refresh()

    def _on_toggle_dimensions(self, checked: bool) -> None:
        if checked and self._reference_point is None:
            self._btn_add_dimensions.setChecked(False)
            self._btn_add_dimensions.setEnabled(False)
            return
        self._dimension_mode = checked
        if checked and self._measure_mode:
            self._measure_mode = False
            self._measure_pending_point = None
            self._btn_measure.blockSignals(True)
            self._btn_measure.setChecked(False)
            self._btn_measure.blockSignals(False)
            self._btn_measure.setText("Measure DX/DY")
            self._measure_log.appendPlainText("Measure mode OFF.")
        if not checked:
            self._hover_pick = None
        self._btn_add_dimensions.setText("Stop adding" if checked else "Add dimensions")
        if checked:
            self._append_dimension_text("Dimension pick mode ON. Click a plotted point.")
        else:
            self._append_dimension_text("Dimension pick mode OFF.")
        self.refresh(preserve_view=True)

    def _on_clear_dimensions(self) -> None:
        self._dimension_records.clear()
        self._hover_pick = None
        self._dimension_log.clear()
        self.refresh(preserve_view=True)

    def _on_clear_reference(self) -> None:
        self._reference_point = None
        self._dimension_mode = False
        self._dimension_records.clear()
        self._hover_pick = None
        self._dimension_log.clear()
        self._ref_x.clear()
        self._ref_y.clear()
        self._btn_add_dimensions.setChecked(False)
        self._btn_add_dimensions.setText("Add dimensions")
        self._btn_add_dimensions.setEnabled(False)
        self.refresh(preserve_view=True)

    def _on_plot_hover(self, event: object) -> None:
        if self._ax is None:
            return
        active_pick_mode = self._measure_mode or (
            self._dimension_mode and self._reference_point is not None
        )
        if not active_pick_mode:
            if self._hover_pick is not None:
                self._hover_pick = None
                self.refresh(preserve_view=True)
            return
        if getattr(event, "inaxes", None) is not self._ax:
            if self._hover_pick is not None:
                self._hover_pick = None
                self.refresh(preserve_view=True)
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        candidate = self._nearest_pick(float(event.xdata), float(event.ydata))
        old = self._hover_pick["label"] if self._hover_pick is not None else None
        new = candidate["label"] if candidate is not None else None
        if old == new:
            return
        self._hover_pick = candidate
        self.refresh(preserve_view=True)

    def _on_plot_click(self, event: object) -> None:
        if self._ax is None:
            return
        if getattr(event, "inaxes", None) is not self._ax:
            return
        if getattr(event, "button", None) != 1:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        picked = self._nearest_pick(float(event.xdata), float(event.ydata))
        if picked is None:
            return
        if self._dimension_mode and self._reference_point is not None:
            px, py = picked["point"]
            rx, ry = self._reference_point
            dx = px - rx
            dy = py - ry
            self._dimension_records.append(
                {
                    "label": picked["label"],
                    "point": (px, py),
                    "reference": (rx, ry),
                    "dx": dx,
                    "dy": dy,
                }
            )
            idx = len(self._dimension_records)
            self._append_dimension_text(
                f"#{idx} {picked['label']} @ [{px:.4f}, {py:.4f}] -> ΔX={dx:+.4f}, ΔY={dy:+.4f}"
            )
            self.refresh(preserve_view=True)
            return
        if not self._measure_mode:
            return
        if self._measure_pending_point is None:
            self._measure_pending_point = picked
            ax, ay = picked["point"]
            self._measure_log.appendPlainText(
                f"A: {picked['label']} @ [{ax:.4f}, {ay:.4f}]"
            )
            self.refresh(preserve_view=True)
            return
        a = self._measure_pending_point
        b = picked
        ax_, ay_ = a["point"]
        bx, by = b["point"]
        dx = bx - ax_
        dy = by - ay_
        rec = {"a": a, "b": b, "dx": dx, "dy": dy}
        self._measure_records.append(rec)
        n = len(self._measure_records)
        self._measure_log.appendPlainText(
            f"#{n} {a['label']} -> {b['label']} | DX={dx:+.4f}, DY={dy:+.4f}"
        )
        self._measure_pending_point = None
        self.refresh(preserve_view=True)

    def set_data(
        self, input_paths: List[np.ndarray], output_paths: List[np.ndarray], distorted_paths: List[np.ndarray]
    ) -> None:
        self._input_paths = input_paths
        self._output_paths = output_paths
        self._distorted_paths = distorted_paths
        self._pickable_points.clear()
        self._hover_pick = None
        self._measure_pending_point = None
        self._on_bbox_controls_changed()

    def refresh(self, *, preserve_view: bool = False) -> None:
        if self._ax is None or self._canvas is None or self._figure is None:
            return
        ax = self._ax
        old_xlim = ax.get_xlim() if preserve_view else None
        old_ylim = ax.get_ylim() if preserve_view else None
        ax.clear()
        ax.grid(True, alpha=0.35)
        ax.set_title("DXF comparison")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        any_drawn = False
        pickables: List[Dict[str, Any]] = []

        if self._chk_input.isChecked():
            any_drawn = self._draw_paths(ax, self._input_paths, color="blue", label="Input DXF") or any_drawn
            pickables.extend(self._collect_pickables(self._input_paths, source="Input"))
            if self._chk_coords.isChecked():
                pts = [item["point"] for item in self._collect_pickables(self._input_paths, source="Input")]
                self._annotate_points(ax, pts, color="blue", prefix="In")
        if self._chk_output.isChecked():
            any_drawn = self._draw_paths(ax, self._output_paths, color="green", label="Output DXF") or any_drawn
            pickables.extend(self._collect_pickables(self._output_paths, source="Output"))
            if self._chk_coords.isChecked():
                pts = [item["point"] for item in self._collect_pickables(self._output_paths, source="Output")]
                self._annotate_points(ax, pts, color="green", prefix="Out")
        if self._chk_distorted.isChecked():
            any_drawn = (
                self._draw_paths(
                    ax, self._distorted_paths, color="purple", label="Distorted"
                )
                or any_drawn
            )
            pickables.extend(self._collect_pickables(self._distorted_paths, source="Distorted"))
            if self._chk_coords.isChecked():
                pts = [item["point"] for item in self._collect_pickables(self._distorted_paths, source="Distorted")]
                self._annotate_points(ax, pts, color="purple", prefix="Distorted")

        if self._bbox_enabled and self._is_layer_visible(self._bbox_layer):
            bbox = self._compute_bbox_from_paths(self._get_layer_paths(self._bbox_layer))
            if bbox is not None:
                self._draw_bbox_overlay(ax, self._bbox_layer, bbox)
                pickables.extend(self._collect_bbox_corner_pickables(self._bbox_layer, bbox))
                any_drawn = True

        self._pickable_points = pickables

        if self._reference_point is not None:
            rx, ry = self._reference_point
            ax.scatter(
                [rx],
                [ry],
                c="black",
                s=95,
                marker="x",
                linewidths=1.7,
                zorder=13,
                label="_nolegend_",
            )
            ax.annotate(
                f"Ref [{rx:.3f}, {ry:.3f}]",
                xy=(rx, ry),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8.5,
                color="black",
                zorder=13,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "alpha": 0.78,
                    "edgecolor": "none",
                },
            )

        for rec in self._dimension_records:
            px, py = rec["point"]
            rx, ry = rec["reference"]
            ax.plot(
                [rx, px],
                [ry, py],
                color="#555",
                linewidth=1.15,
                alpha=0.78,
                zorder=10,
                label="_nolegend_",
            )
            ax.plot(
                [rx, px],
                [ry, ry],
                linestyle=":",
                linewidth=1.0,
                color="#666",
                alpha=0.58,
                zorder=9,
                label="_nolegend_",
            )
            ax.plot(
                [px, px],
                [ry, py],
                linestyle=":",
                linewidth=1.0,
                color="#666",
                alpha=0.58,
                zorder=9,
                label="_nolegend_",
            )
            ax.scatter([px], [py], c="#222", s=28, zorder=11, label="_nolegend_")
            ax.annotate(
                f"ΔX={rec['dx']:+.3f}, ΔY={rec['dy']:+.3f}",
                xy=(px, py),
                xytext=(10, -14),
                textcoords="offset points",
                fontsize=8,
                color="black",
                zorder=12,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "alpha": 0.78,
                    "edgecolor": "none",
                },
            )

        if self._hover_pick is not None:
            hx, hy = self._hover_pick["point"]
            ax.scatter(
                [hx],
                [hy],
                c="none",
                s=210,
                marker="o",
                edgecolors="gold",
                linewidths=2.0,
                zorder=14,
                label="_nolegend_",
            )

        if self._measure_pending_point is not None:
            ax_, ay_ = self._measure_pending_point["point"]
            ax.scatter(
                [ax_],
                [ay_],
                c="black",
                s=90,
                marker="x",
                zorder=14,
                linewidths=1.4,
                label="_nolegend_",
            )
            ax.annotate(
                "A",
                xy=(ax_, ay_),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                color="black",
                zorder=14,
            )

        for rec in self._measure_records:
            ax1, ay1 = rec["a"]["point"]
            bx, by = rec["b"]["point"]
            ax.plot([ax1, bx], [ay1, by], color="#444", linewidth=1.2, alpha=0.8, zorder=10, label="_nolegend_")
            ax.plot([ax1, bx], [ay1, ay1], linestyle=":", linewidth=1.0, color="#666", alpha=0.55, zorder=9, label="_nolegend_")
            ax.plot([bx, bx], [ay1, by], linestyle=":", linewidth=1.0, color="#666", alpha=0.55, zorder=9, label="_nolegend_")
            ax.scatter([ax1], [ay1], c="#111", s=28, zorder=11, label="_nolegend_")
            ax.scatter([bx], [by], c="#111", s=28, zorder=11, label="_nolegend_")
            ax.annotate(
                f"DX={rec['dx']:+.3f}, DY={rec['dy']:+.3f}",
                xy=(bx, by),
                xytext=(10, -14),
                textcoords="offset points",
                fontsize=8,
                color="black",
                zorder=12,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "alpha": 0.78,
                    "edgecolor": "none",
                },
            )

        if any_drawn:
            if not preserve_view:
                ax.relim()
                ax.autoscale_view()
            handles, labels = ax.get_legend_handles_labels()
            by_label: Dict[str, object] = {}
            for h, lab in zip(handles, labels):
                if lab and not lab.startswith("_"):
                    by_label[lab] = h
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=9)
        else:
            ax.text(
                0.5,
                0.5,
                "No DXF paths to display.\nEnable a layer or check file contents.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        if preserve_view and old_xlim is not None and old_ylim is not None:
            ax.set_xlim(old_xlim)
            ax.set_ylim(old_ylim)
        ax.set_aspect("equal", adjustable="box")
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def closeEvent(self, event) -> None:
        self._main._dxf_compare_dialog = None
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Affine Calibration GUI")
        self.resize(1100, 780)

        self.result: AffineResult | None = None
        self._validator = QDoubleValidator()
        self._validator.setNotation(QDoubleValidator.StandardNotation)
        self._mpl_canvas: Optional[object] = None
        self._mpl_figure: Optional[object] = None
        self._mpl_ax: Optional[object] = None
        self._mpl_click_cid: Optional[int] = None
        self._plot_dialog: Optional[CalibrationPlotDialog] = None
        self._dxf_compare_dialog: Optional[DxfCompareDialog] = None
        # Additive manual tweaks on top of last Solve (forward map ideal → measured).
        self._adj_dtx = 0.0
        self._adj_dty = 0.0
        self._adj_drot = 0.0
        self._adj_dsx = 0.0
        self._adj_dsy = 0.0
        self._adj_dshear = 0.0
        # Measured-space inputs only; ideal/comp/predicted recomputed on each draw from current affine.
        self._verification_measured_points: List[Point] = []

        self._build_ui()
        self._set_actions_enabled(False)
        self.update_shape_plot()

    def _build_ui(self) -> None:
        container = QWidget()
        root = QVBoxLayout(container)
        root.setSpacing(8)

        # Use top-row split to avoid large unused right-side whitespace.
        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        # Give calibration section room to grow so pair list + buttons do not overlap.
        top_row.addWidget(self._build_calibration_group(), stretch=3)
        top_row.addWidget(self._build_right_column(), stretch=2)
        root.addLayout(top_row, stretch=1)

        mid_row = QHBoxLayout()
        mid_row.setSpacing(10)
        mid_row.addWidget(self._build_rectify_group(), stretch=3)
        mid_row.addWidget(self._build_dxf_group(), stretch=2)
        root.addLayout(mid_row)
        root.addWidget(self._build_logs_group())
        root.addLayout(self._build_button_bar())

        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")
        self._build_menu()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        action_quit = QAction("Quit", self)
        action_quit.triggered.connect(self.close)
        file_menu.addAction(action_quit)

    def _build_right_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_diagnostics_group(), stretch=0)
        layout.addWidget(self._build_shape_plot_group(), stretch=1)
        return column

    def _build_shape_plot_group(self) -> QGroupBox:
        group = QGroupBox("Calibration preview")
        layout = QVBoxLayout(group)
        self._mpl_canvas = None
        self._mpl_figure = None
        self._mpl_ax = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            figure = Figure(figsize=(4.5, 4.0))
            ax = figure.add_subplot(111)
            canvas = FigureCanvasQTAgg(figure)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumHeight(220)
            canvas.setCursor(Qt.CursorShape.PointingHandCursor)
            canvas.setToolTip("Click to open full view (zoom, pan, legend)")
            layout.addWidget(canvas)
            self._mpl_figure = figure
            self._mpl_ax = ax
            self._mpl_canvas = canvas
            self._mpl_click_cid = canvas.mpl_connect(
                "button_press_event", self._on_embedded_plot_click
            )
        except Exception as exc:
            err = QLabel(
                f"Plot unavailable ({exc}).\nInstall matplotlib: pip install matplotlib"
            )
            err.setWordWrap(True)
            layout.addWidget(err)
        return group

    @staticmethod
    def _try_parse_pair(sx: str, sy: str) -> Optional[Point]:
        if sx == "" or sy == "":
            return None
        try:
            return (float(sx), float(sy))
        except ValueError:
            return None

    def _collect_partial_ideal_points(self) -> List[Point]:
        out: List[Point] = []
        for card in self.pair_cards:
            ix, iy, _, _ = card.values()
            p = self._try_parse_pair(ix, iy)
            if p is not None:
                out.append(p)
        return out

    def _collect_partial_measured_points(self) -> List[Point]:
        out: List[Point] = []
        for card in self.pair_cards:
            _, _, lx, ly = card.values()
            p = self._try_parse_pair(lx, ly)
            if p is not None:
                out.append(p)
        return out

    def _collect_complete_pairs(self) -> Tuple[List[Point], List[Point]]:
        ideal_pairs: List[Point] = []
        measured_pairs: List[Point] = []
        for card in self.pair_cards:
            ix, iy, lx, ly = card.values()
            if ix == "" or iy == "" or lx == "" or ly == "":
                continue
            try:
                ideal_pairs.append((float(ix), float(iy)))
                measured_pairs.append((float(lx), float(ly)))
            except ValueError:
                continue
        return ideal_pairs, measured_pairs

    @staticmethod
    def _closed_ring(points: np.ndarray) -> np.ndarray:
        if len(points) < 3:
            return points
        return np.vstack([points, points[0:1]])

    def _plot_point_chain(
        self,
        ax: object,
        points: List[Point],
        color: str,
        label: str,
        zorder: int,
        *,
        minimal: bool,
    ) -> None:
        if not points:
            return
        arr = np.asarray(points, dtype=float)
        leg = "_nolegend_" if minimal else label
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            c=color,
            s=40,
            zorder=zorder + 1,
            edgecolors="white",
            linewidths=0.7,
            label=leg,
        )
        if len(arr) >= 2:
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                color=color,
                linewidth=1.5,
                zorder=zorder,
                label="_nolegend_",
            )
        if len(arr) >= 3:
            ring = self._closed_ring(arr)
            ax.plot(
                ring[:, 0],
                ring[:, 1],
                color=color,
                linewidth=1.5,
                zorder=zorder,
                alpha=0.9,
                label="_nolegend_",
            )

    def _compute_verification_geometry(
        self, measured: Point
    ) -> Tuple[Point, Point, Point, Point]:
        """Ideal from inverse affine; compensated feed for that ideal; predicted = mean(measured, compensated). Uses current effective map."""
        eff = self.get_effective_forward_affine()
        if eff is None:
            raise ValueError("No calibration result.")
        a_eff, t_eff = eff
        ideal_np = apply_inverse_transform(measured, a_eff, t_eff)
        ideal = (float(ideal_np[0]), float(ideal_np[1]))
        comp_a, comp_t = build_compensation_transform(a_eff, t_eff)
        ig = np.asarray([ideal[0], ideal[1]], dtype=float)
        comp_np = comp_a @ ig + comp_t
        measured_np = np.asarray([measured[0], measured[1]], dtype=float)
        pred_np = 0.5 * (measured_np + comp_np)
        measured_pt = (float(measured[0]), float(measured[1]))
        return (
            measured_pt,
            ideal,
            (float(comp_np[0]), float(comp_np[1])),
            (float(pred_np[0]), float(pred_np[1])),
        )

    @staticmethod
    def _fmt_xy(pt: Point) -> str:
        return f"[{pt[0]:.3f}, {pt[1]:.3f}]"

    def _annotate_points(
        self,
        ax: object,
        points: List[Point],
        *,
        color: str,
        prefix: str,
        xoff: float,
        yoff: float,
        zorder: int = 14,
    ) -> None:
        for p in points:
            ax.annotate(
                f"{prefix} {self._fmt_xy(p)}",
                xy=(p[0], p[1]),
                xytext=(xoff, yoff),
                textcoords="offset points",
                fontsize=8,
                color=color,
                zorder=zorder,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                },
            )

    def _draw_verification_marks(
        self,
        ax: object,
        *,
        minimal: bool,
        annotate_coords: bool,
        pick_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Scatter-only overlays; recomputes positions from current manual corrections each draw."""
        v_pts = self._verification_measured_points
        if not v_pts:
            return False
        show_lbl = not minimal
        for i, mpt in enumerate(v_pts):
            try:
                _, ideal, comp, pred = self._compute_verification_geometry(mpt)
            except ValueError:
                continue
            l0 = "Verify measured" if (show_lbl and i == 0) else "_nolegend_"
            l1 = "Verify ideal" if (show_lbl and i == 0) else "_nolegend_"
            l2 = "Verify compensated" if (show_lbl and i == 0) else "_nolegend_"
            l3 = "Verify predicted" if (show_lbl and i == 0) else "_nolegend_"
            ax.scatter(
                [mpt[0]],
                [mpt[1]],
                c="#e6550d",
                s=90,
                marker="o",
                zorder=12,
                edgecolors="white",
                linewidths=1.0,
                label=l0,
            )
            if pick_sink is not None:
                pick_sink.append({"label": f"Verify measured #{i+1}", "point": mpt})
            ax.scatter(
                [ideal[0]],
                [ideal[1]],
                c="#00b8d4",
                s=90,
                marker="s",
                zorder=12,
                edgecolors="white",
                linewidths=1.0,
                label=l1,
            )
            if pick_sink is not None:
                pick_sink.append({"label": f"Verify ideal #{i+1}", "point": ideal})
            ax.scatter(
                [comp[0]],
                [comp[1]],
                c="#c9a227",
                s=90,
                marker="^",
                zorder=12,
                edgecolors="white",
                linewidths=1.0,
                label=l2,
            )
            if pick_sink is not None:
                pick_sink.append({"label": f"Verify compensated #{i+1}", "point": comp})
            ax.scatter(
                [pred[0]],
                [pred[1]],
                c="#c451ce",
                s=90,
                marker="D",
                zorder=12,
                edgecolors="white",
                linewidths=1.0,
                label=l3,
            )
            if pick_sink is not None:
                pick_sink.append({"label": f"Verify predicted #{i+1}", "point": pred})
            if annotate_coords and not minimal:
                self._annotate_points(
                    ax, [mpt], color="#e6550d", prefix="V-Measure", xoff=7, yoff=7
                )
                self._annotate_points(
                    ax, [ideal], color="#00b8d4", prefix="V-Ideal", xoff=7, yoff=-10
                )
                self._annotate_points(
                    ax, [comp], color="#c9a227", prefix="V-Comp", xoff=-55, yoff=7
                )
                self._annotate_points(
                    ax, [pred], color="#c451ce", prefix="V-Pred", xoff=-55, yoff=-10
                )
        return True

    def _render_calibration_axes(
        self,
        ax: object,
        *,
        minimal: bool,
        visibility: Optional[Dict[str, bool]] = None,
        pick_sink: Optional[List[Dict[str, Any]]] = None,
        reference_point: Optional[Point] = None,
        dimension_records: Optional[List[Dict[str, Any]]] = None,
        hover_pick: Optional[Dict[str, Any]] = None,
        preserve_view: bool = False,
    ) -> None:
        ax.clear()
        ax.grid(True, alpha=0.35)
        vis = _default_plot_visibility()
        if not minimal and visibility is not None:
            vis.update(visibility)

        if minimal:
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelbottom=False, labelleft=False)
        else:
            ax.set_title("Calibration preview")
            ax.set_xlabel("X (ideal vs measured overlaid for comparison)")
            ax.set_ylabel("Y")
            ax.tick_params(labelbottom=True, labelleft=True)

        ideal_partial = self._collect_partial_ideal_points()
        measured_partial = self._collect_partial_measured_points()
        ideal_complete, _ = self._collect_complete_pairs()
        n_complete = len(ideal_complete)

        show_ideal = minimal or vis[PLOT_LAYER_IDEAL]
        show_measured = minimal or vis[PLOT_LAYER_MEASURED]
        show_comp = minimal or vis[PLOT_LAYER_COMPENSATED]
        show_predicted_print = (not minimal) and vis[PLOT_LAYER_PREDICTED_PRINT]
        show_coords = (not minimal) and vis[PLOT_LAYER_COORDINATES]
        all_layers_hidden = not (show_ideal or show_measured or show_comp or show_predicted_print)

        any_drawn = False

        if ideal_partial and show_ideal:
            self._plot_point_chain(
                ax, ideal_partial, "blue", "Ideal (computer)", zorder=2, minimal=minimal
            )
            if pick_sink is not None:
                for idx, p in enumerate(ideal_partial, start=1):
                    pick_sink.append({"label": f"Ideal #{idx}", "point": p})
            if show_coords:
                self._annotate_points(
                    ax, ideal_partial, color="blue", prefix="Ideal", xoff=8, yoff=8
                )
            any_drawn = True
        if measured_partial and show_measured:
            self._plot_point_chain(
                ax, measured_partial, "red", "As measured", zorder=2, minimal=minimal
            )
            if pick_sink is not None:
                for idx, p in enumerate(measured_partial, start=1):
                    pick_sink.append({"label": f"Measured #{idx}", "point": p})
            if show_coords:
                self._annotate_points(
                    ax,
                    measured_partial,
                    color="red",
                    prefix="Measured",
                    xoff=8,
                    yoff=-11,
                )
            any_drawn = True

        comp: Optional[np.ndarray] = None
        if self.result is not None and n_complete >= 3:
            ideal_arr = np.asarray(ideal_complete, dtype=float)
            a_eff, t_eff = self.get_effective_forward_affine()
            assert a_eff is not None
            comp_a, comp_t = build_compensation_transform(a_eff, t_eff)
            comp = (comp_a @ ideal_arr.T).T + comp_t
            if show_comp:
                comp_ring = self._closed_ring(comp)
                glab = "_nolegend_" if minimal else "Compensated (machine feed)"
                ax.plot(
                    comp_ring[:, 0],
                    comp_ring[:, 1],
                    color="green",
                    linewidth=2.0,
                    zorder=6,
                    label=glab,
                )
                if show_coords:
                    comp_pts = [
                        (float(row[0]), float(row[1])) for row in np.asarray(comp, dtype=float)
                    ]
                    self._annotate_points(
                        ax, comp_pts, color="green", prefix="Comp", xoff=-56, yoff=8
                    )
                if pick_sink is not None:
                    for idx, row in enumerate(np.asarray(comp, dtype=float), start=1):
                        pick_sink.append(
                            {
                                "label": f"Compensated #{idx}",
                                "point": (float(row[0]), float(row[1])),
                            }
                        )
                any_drawn = True
            if show_predicted_print and comp is not None:
                _, measured_complete = self._collect_complete_pairs()
                measured_arr = np.asarray(measured_complete, dtype=float)
                predicted = 0.5 * (measured_arr + comp)
                pred_ring = self._closed_ring(predicted)
                ax.plot(
                    pred_ring[:, 0],
                    pred_ring[:, 1],
                    color="purple",
                    linewidth=1.8,
                    linestyle="-",
                    zorder=4,
                    label="Predicted print (mean of measured & compensated)",
                )
                if show_coords:
                    pred_pts = [
                        (float(row[0]), float(row[1]))
                        for row in np.asarray(predicted, dtype=float)
                    ]
                    self._annotate_points(
                        ax, pred_pts, color="purple", prefix="Pred", xoff=-56, yoff=-11
                    )
                if pick_sink is not None:
                    for idx, row in enumerate(np.asarray(predicted, dtype=float), start=1):
                        pick_sink.append(
                            {
                                "label": f"Predicted #{idx}",
                                "point": (float(row[0]), float(row[1])),
                            }
                        )
                any_drawn = True

        if self._draw_verification_marks(
            ax,
            minimal=minimal,
            annotate_coords=show_coords,
            pick_sink=pick_sink,
        ):
            any_drawn = True

        if not minimal and reference_point is not None:
            ax.scatter(
                [reference_point[0]],
                [reference_point[1]],
                c="black",
                s=100,
                marker="X",
                zorder=13,
                edgecolors="white",
                linewidths=0.9,
                label="Reference",
            )
            if show_coords:
                self._annotate_points(
                    ax, [reference_point], color="black", prefix="Ref", xoff=10, yoff=10
                )
            if dimension_records:
                for rec in dimension_records:
                    px, py = rec["point"]
                    rx, ry = rec.get("reference", reference_point)
                    ax.plot(
                        [rx, px],
                        [ry, ry],
                        linestyle=":",
                        linewidth=1.2,
                        color="black",
                        alpha=0.55,
                        zorder=8,
                        label="_nolegend_",
                    )
                    ax.plot(
                        [px, px],
                        [ry, py],
                        linestyle=":",
                        linewidth=1.2,
                        color="black",
                        alpha=0.55,
                        zorder=8,
                        label="_nolegend_",
                    )
                    ax.annotate(
                        f"ΔX={rec['dx']:+.3f}, ΔY={rec['dy']:+.3f}",
                        xy=(px, py),
                        xytext=(10, -24),
                        textcoords="offset points",
                        fontsize=8,
                        color="black",
                        zorder=15,
                        bbox={
                            "boxstyle": "round,pad=0.12",
                            "facecolor": "white",
                            "alpha": 0.75,
                            "edgecolor": "none",
                        },
                    )
            if hover_pick is not None:
                hx, hy = hover_pick["point"]
                ax.scatter(
                    [hx],
                    [hy],
                    c="none",
                    s=230,
                    marker="o",
                    edgecolors="gold",
                    linewidths=2.0,
                    zorder=16,
                    label="_nolegend_",
                )
                ax.annotate(
                    f"Pick: {hover_pick['label']}",
                    xy=(hx, hy),
                    xytext=(11, 12),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    zorder=16,
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "#fff7cf",
                        "alpha": 0.85,
                        "edgecolor": "none",
                    },
                )

        has_partial = bool(ideal_partial or measured_partial)
        has_green = self.result is not None and n_complete >= 3
        has_verify = bool(self._verification_measured_points)
        if not has_partial and not has_green and not has_verify:
            if not minimal:
                ax.text(
                    0.5,
                    0.55,
                    "Enter ideal coordinates (blue),\nthen measured coordinates (red).\n"
                    "Lines connect in order; polygons close at 3+ points.\n"
                    "After Solve, green is compensated machine feed; purple is the midpoint\n"
                    "between red (as measured) and green at each vertex.\n"
                    "Use the toolbar below for zoom and pan; use checkboxes to show layers.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="gray",
                )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        elif any_drawn and not preserve_view:
            ax.relim()
            ax.autoscale_view()
        elif not minimal and all_layers_hidden:
            ax.text(
                0.5,
                0.5,
                "All plot layers are hidden.\nCheck one or more boxes above.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

        if not minimal:
            handles, labels = ax.get_legend_handles_labels()
            by_label: dict[str, object] = {}
            for h, lab in zip(handles, labels):
                if lab and not lab.startswith("_"):
                    by_label[lab] = h
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=9)

        ax.set_aspect("equal", adjustable="box")

    def _on_embedded_plot_click(self, event: object) -> None:
        self._open_plot_dialog()

    def _open_plot_dialog(self) -> None:
        if self._mpl_canvas is None:
            return
        if self._plot_dialog is not None:
            self._plot_dialog.refresh()
            self._plot_dialog.showMaximized()
            self._plot_dialog.raise_()
            self._plot_dialog.activateWindow()
            return
        dlg = CalibrationPlotDialog(self)
        self._plot_dialog = dlg
        dlg.refresh()
        dlg.showMaximized()
        dlg.raise_()
        dlg.activateWindow()

    def update_shape_plot(self) -> None:
        if self.result is not None:
            self._refresh_diagnostics_display()
        if self._mpl_ax is None or self._mpl_canvas is None or self._mpl_figure is None:
            return
        self._render_calibration_axes(self._mpl_ax, minimal=True)
        self._mpl_figure.tight_layout()
        self._mpl_canvas.draw_idle()
        if self._plot_dialog is not None:
            self._plot_dialog.refresh()

    def _reset_user_adjustments(self) -> None:
        self._adj_dtx = 0.0
        self._adj_dty = 0.0
        self._adj_drot = 0.0
        self._adj_dsx = 0.0
        self._adj_dsy = 0.0
        self._adj_dshear = 0.0

    def get_effective_forward_affine(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Solved forward map (ideal → measured) with additive manual corrections."""
        if self.result is None:
            return None
        return compose_user_adjustment(
            self.result.matrix,
            self.result.translation,
            dtx=self._adj_dtx,
            dty=self._adj_dty,
            d_rot_deg=self._adj_drot,
            d_scale_x=self._adj_dsx,
            d_scale_y=self._adj_dsy,
            d_shear=self._adj_dshear,
        )

    def _build_calibration_group(self) -> QGroupBox:
        group = QGroupBox("Calibration Inputs (center first + variable pairs)")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 12, 10, 10)

        center_row = QHBoxLayout()
        self.ideal_center_x = QLineEdit("0")
        self.ideal_center_y = QLineEdit("0")
        self.ideal_center_x.setValidator(self._validator)
        self.ideal_center_y.setValidator(self._validator)
        self.ideal_center_x.setReadOnly(True)
        self.ideal_center_y.setReadOnly(True)
        self.ideal_center_x.setStyleSheet("background-color: #f0f0f0;")
        self.ideal_center_y.setStyleSheet("background-color: #f0f0f0;")
        self.measured_center_x = QLineEdit()
        self.measured_center_y = QLineEdit()
        self.measured_center_x.setValidator(self._validator)
        self.measured_center_y.setValidator(self._validator)

        center_row.addWidget(QLabel("Ideal center X"))
        center_row.addWidget(self.ideal_center_x)
        center_row.addWidget(QLabel("Ideal center Y"))
        center_row.addWidget(self.ideal_center_y)
        center_row.addWidget(QLabel("Measured center X (optional)"))
        center_row.addWidget(self.measured_center_x)
        center_row.addWidget(QLabel("Measured center Y (optional)"))
        center_row.addWidget(self.measured_center_y)
        center_row.addStretch(1)
        layout.addLayout(center_row)

        self.pair_cards: List[PairCardWidget] = []
        self.pairs_container = QWidget()
        self.pairs_layout = QVBoxLayout(self.pairs_container)
        self.pairs_layout.setContentsMargins(0, 0, 0, 0)
        self.pairs_layout.setSpacing(8)
        pairs_scroll = QScrollArea()
        pairs_scroll.setWidgetResizable(True)
        pairs_scroll.setFrameShape(QFrame.StyledPanel)
        pairs_scroll.setFrameShadow(QFrame.Plain)
        pairs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        pairs_scroll.setMinimumHeight(220)
        pairs_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pairs_scroll.setWidget(self.pairs_container)
        # Stretch so extra vertical space goes to the pair list, not under the buttons.
        layout.addWidget(pairs_scroll, stretch=1)

        controls_footer = QWidget()
        controls_footer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button_row = QHBoxLayout(controls_footer)
        button_row.setContentsMargins(0, 10, 0, 0)
        self.btn_add_pair = QPushButton("Add Pair")
        self.btn_remove_pair = QPushButton("Remove Pair")
        self.btn_save_coords_csv = QPushButton("Save Coordinates CSV")
        self.btn_load_coords_csv = QPushButton("Load Coordinates CSV")
        self.btn_solve = QPushButton("Solve Calibration")
        self.btn_add_pair.clicked.connect(self.on_add_pair)
        self.btn_remove_pair.clicked.connect(self.on_remove_pair)
        self.btn_save_coords_csv.clicked.connect(self.on_save_coordinates_csv)
        self.btn_load_coords_csv.clicked.connect(self.on_load_coordinates_csv)
        self.btn_solve.clicked.connect(self.on_solve_calibration)
        button_row.addWidget(self.btn_add_pair)
        button_row.addWidget(self.btn_remove_pair)
        button_row.addWidget(self.btn_save_coords_csv)
        button_row.addWidget(self.btn_load_coords_csv)
        button_row.addStretch(1)
        button_row.addWidget(self.btn_solve)
        layout.addWidget(controls_footer, stretch=0)

        for _ in range(3):
            self.on_add_pair()
        return group

    def _build_diagnostics_group(self) -> QGroupBox:
        group = QGroupBox("Calibration Diagnostics (includes manual corrections)")
        form = QFormLayout(group)
        self.lbl_matrix = QLabel("-")
        self.lbl_translation = QLabel("-")
        self.lbl_pred_center = QLabel("-")
        self.lbl_params = QLabel("-")
        self.lbl_rms = QLabel("-")
        for l in (self.lbl_matrix, self.lbl_translation, self.lbl_pred_center, self.lbl_params, self.lbl_rms):
            l.setTextInteractionFlags(Qt.TextSelectableByMouse)
            l.setWordWrap(True)
        form.addRow("A (2x2):", self.lbl_matrix)
        form.addRow("Translation:", self.lbl_translation)
        form.addRow("Predicted measured center:", self.lbl_pred_center)
        form.addRow("TRSS:", self.lbl_params)
        form.addRow("RMS error:", self.lbl_rms)
        return group

    def _build_rectify_group(self) -> QGroupBox:
        group = QGroupBox("Point Rectification")
        root = QVBoxLayout(group)

        title_row_1 = QHBoxLayout()
        title_row_1.addWidget(QLabel("Measured → Ideal"))
        title_row_1.addStretch(1)
        root.addLayout(title_row_1)

        row = QHBoxLayout()
        self.rect_in_x = QLineEdit()
        self.rect_in_y = QLineEdit()
        self.rect_in_x.setValidator(self._validator)
        self.rect_in_y.setValidator(self._validator)
        self.rect_out = QLabel("-")
        self.rect_out.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.btn_rectify = QPushButton("Rectify Point")
        self.btn_rectify.clicked.connect(self.on_rectify_point)

        row.addWidget(QLabel("Measured X:"))
        row.addWidget(self.rect_in_x)
        row.addWidget(QLabel("Measured Y:"))
        row.addWidget(self.rect_in_y)
        row.addWidget(self.btn_rectify)
        row.addWidget(QLabel("Ideal Output:"))
        row.addWidget(self.rect_out, 1)
        root.addLayout(row)

        title_row_2 = QHBoxLayout()
        title_row_2.addWidget(QLabel("Ideal → Measured"))
        title_row_2.addStretch(1)
        root.addLayout(title_row_2)

        row_fwd = QHBoxLayout()
        self.rect_fwd_in_x = QLineEdit()
        self.rect_fwd_in_y = QLineEdit()
        self.rect_fwd_in_x.setValidator(self._validator)
        self.rect_fwd_in_y.setValidator(self._validator)
        self.rect_fwd_out = QLabel("-")
        self.rect_fwd_out.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.btn_rectify_fwd = QPushButton("Verify Forward")
        self.btn_rectify_fwd.clicked.connect(self.on_rectify_forward_point)
        row_fwd.addWidget(QLabel("Ideal X:"))
        row_fwd.addWidget(self.rect_fwd_in_x)
        row_fwd.addWidget(QLabel("Ideal Y:"))
        row_fwd.addWidget(self.rect_fwd_in_y)
        row_fwd.addWidget(self.btn_rectify_fwd)
        row_fwd.addWidget(QLabel("Measured Output:"))
        row_fwd.addWidget(self.rect_fwd_out, 1)
        root.addLayout(row_fwd)

        row2 = QHBoxLayout()
        self.btn_add_verification = QPushButton("Add verification mark")
        self.btn_add_verification.setToolTip(
            "Stores this measured (X,Y) and plots ideal, compensated, and predicted using the current map. "
            "Those three derived points move when you change manual corrections (measured input stays fixed)."
        )
        self.btn_add_verification.clicked.connect(self.on_add_verification_mark)
        self.btn_clear_verification = QPushButton("Clear verification marks")
        self.btn_clear_verification.setToolTip("Remove all verification scatter marks from the plot.")
        self.btn_clear_verification.clicked.connect(self.on_clear_verification_marks)
        row2.addWidget(self.btn_add_verification)
        row2.addWidget(self.btn_clear_verification)
        row2.addStretch(1)
        root.addLayout(row2)
        return group

    def _build_dxf_group(self) -> QGroupBox:
        group = QGroupBox("DXF Compensation (Ideal -> Pre-distorted DXF)")
        layout = QGridLayout(group)

        self.dxf_input = QLineEdit()
        self.dxf_output = QLineEdit()
        self.btn_browse_input = QPushButton("Browse Input")
        self.btn_browse_output = QPushButton("Browse Output")
        self.btn_process_dxf = QPushButton("Process DXF")
        self.btn_show_dxf_compare = QPushButton("Show DXF Comparison Graph")

        self.btn_browse_input.clicked.connect(self.on_browse_input_dxf)
        self.btn_browse_output.clicked.connect(self.on_browse_output_dxf)
        self.btn_process_dxf.clicked.connect(self.on_process_dxf)
        self.btn_show_dxf_compare.clicked.connect(self.on_show_dxf_comparison)

        layout.addWidget(QLabel("Input DXF:"), 0, 0)
        layout.addWidget(self.dxf_input, 0, 1)
        layout.addWidget(self.btn_browse_input, 0, 2)
        layout.addWidget(QLabel("Output DXF:"), 1, 0)
        layout.addWidget(self.dxf_output, 1, 1)
        layout.addWidget(self.btn_browse_output, 1, 2)
        layout.addWidget(self.btn_show_dxf_compare, 2, 1)
        layout.addWidget(self.btn_process_dxf, 2, 2)
        return group

    def _build_logs_group(self) -> QGroupBox:
        group = QGroupBox("Logs")
        layout = QVBoxLayout(group)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        return group

    def _build_button_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.btn_export = QPushButton("Export Report")
        self.btn_reset = QPushButton("Reset")
        self.btn_export.clicked.connect(self.on_export_report)
        self.btn_reset.clicked.connect(self.on_reset)
        bar.addWidget(self.btn_export)
        bar.addWidget(self.btn_reset)
        bar.addStretch(1)
        return bar

    def _set_actions_enabled(self, solved: bool) -> None:
        self.btn_rectify.setEnabled(solved)
        self.btn_rectify_fwd.setEnabled(solved)
        self.btn_add_verification.setEnabled(solved)
        self.btn_clear_verification.setEnabled(solved)
        self.btn_process_dxf.setEnabled(solved)
        self.btn_show_dxf_compare.setEnabled(solved)
        self.btn_export.setEnabled(solved)

    def _restore_maximized_if_needed(self, was_maximized: bool) -> None:
        if not was_maximized:
            return

        def _force_maximized() -> None:
            screen = self.screen()
            if screen is None:
                screen = QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                # Qt can ignore geometry updates while already maximized, so normalize first.
                self.showNormal()
                self.setGeometry(avail)
            self.setWindowState(Qt.WindowNoState)
            self.setWindowState(Qt.WindowMaximized)
            self.showMaximized()

        # Run immediately and once more after layout settles.
        QTimer.singleShot(0, _force_maximized)
        QTimer.singleShot(120, _force_maximized)

    def _log(self, text: str) -> None:
        self.log_box.appendPlainText(text)

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)
        self.statusBar().showMessage("Error")
        self._log(f"[ERROR] {message}")

    def _read_float(self, widget: QLineEdit, label: str) -> float:
        raw = widget.text().strip()
        if raw == "":
            raise ValueError(f"Missing value for {label}.")
        return float(raw)

    def _read_calibration_points(self) -> Tuple[Point, List[Point], Point | None, List[Point]]:
        try:
            ideal_center = (
                self._read_float(self.ideal_center_x, "Ideal center X"),
                self._read_float(self.ideal_center_y, "Ideal center Y"),
            )
            if ideal_center != (0.0, 0.0):
                raise ValueError("Ideal center must be (0,0).")

            measured_center: Point | None = None
            raw_lx = self.measured_center_x.text().strip()
            raw_ly = self.measured_center_y.text().strip()
            if raw_lx or raw_ly:
                if not raw_lx or not raw_ly:
                    raise ValueError("Provide both measured center X and Y, or leave both blank.")
                measured_center = (float(raw_lx), float(raw_ly))

            ideal_pairs: List[Point] = []
            measured_pairs: List[Point] = []
            for row, card in enumerate(self.pair_cards, start=1):
                vals: List[float] = []
                for txt, name in zip(card.values(), ("Ideal X", "Ideal Y", "Measured X", "Measured Y")):
                    if txt == "":
                        vals = []
                        break
                    try:
                        vals.append(float(txt))
                    except ValueError as exc:
                        raise ValueError(f"Invalid number at pair row {row}, {name}.") from exc
                if vals:
                    ideal_pairs.append((vals[0], vals[1]))
                    measured_pairs.append((vals[2], vals[3]))

            if len(ideal_pairs) < 3:
                raise ValueError("At least 3 complete matched pairs are required.")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        return ideal_center, ideal_pairs, measured_center, measured_pairs

    def on_solve_calibration(self) -> None:
        was_maximized = self.isMaximized()
        try:
            ideal_center, ideal_pairs, measured_center, measured_pairs = self._read_calibration_points()
            result = build_affine_result(
                ideal_center=ideal_center,
                ideal_points=ideal_pairs,
                measured_points=measured_pairs,
                provided_measured_center=measured_center,
            )
            self.result = result
            self._reset_user_adjustments()
            self._verification_measured_points.clear()
            self._set_actions_enabled(True)
            self._update_diagnostics(result)
            self.update_shape_plot()
            if self._plot_dialog is not None:
                self._plot_dialog.clear_dimension_state()
            self.statusBar().showMessage("Calibration solved")
            self._log("Calibration solved successfully.")
            if result.rank < 6:
                self._log("WARNING: low rank; calibration may be unstable.")
        except Exception as exc:
            self._error(str(exc))
            self.result = None
            self._reset_user_adjustments()
            self._verification_measured_points.clear()
            self._set_actions_enabled(False)
            self.update_shape_plot()
            if self._plot_dialog is not None:
                self._plot_dialog.clear_dimension_state()
        finally:
            self._restore_maximized_if_needed(was_maximized)

    def _refresh_diagnostics_display(self) -> None:
        """Show effective forward map (A, t) after manual adjustments; RMS stays calibration fit."""
        r = self.result
        if r is None:
            return
        eff = self.get_effective_forward_affine()
        if eff is None:
            return
        a_eff, t_eff = eff
        self.lbl_matrix.setText(np.array2string(a_eff, precision=6))
        self.lbl_translation.setText(f"tx={t_eff[0]:.6f}, ty={t_eff[1]:.6f}")
        rot_deg, sx, sy, shx, shy = decompose_affine(a_eff, t_eff)
        self.lbl_params.setText(
            f"rot={rot_deg:.4f} deg, sx={sx:.6f}, "
            f"sy={sy:.6f}, shx={shx:.6f}, shy={shy:.6f}"
        )
        pred = apply_transform((0.0, 0.0), a_eff, t_eff)
        center_text = f"x={pred[0]:.6f}, y={pred[1]:.6f}"
        if r.provided_measured_center is not None:
            prov = r.provided_measured_center
            delta = pred - prov
            center_text += (
                f" | provided=({prov[0]:.6f},{prov[1]:.6f})"
                f" | delta=({delta[0]:.6f},{delta[1]:.6f})"
            )
        self.lbl_pred_center.setText(center_text)
        self.lbl_rms.setText(f"{r.rms_error:.6f} (calibration fit)")

    def _update_diagnostics(self, result: AffineResult) -> None:
        labels = [f"p{i+1}" for i in range(len(result.residual_vectors))]
        for i, (dx, dy) in enumerate(result.residual_vectors):
            self._log(f"Residual {labels[i]}: dx={dx:.6f}, dy={dy:.6f}")
        self._refresh_diagnostics_display()

    def on_add_pair(self) -> None:
        was_maximized = self.isMaximized()
        card = PairCardWidget(index=len(self.pair_cards) + 1, validator=self._validator)
        card.connect_plot_updates(self.update_shape_plot)
        self.pair_cards.append(card)
        self.pairs_layout.addWidget(card)
        self.update_shape_plot()
        self._restore_maximized_if_needed(was_maximized)

    def on_remove_pair(self) -> None:
        was_maximized = self.isMaximized()
        if not self.pair_cards:
            return
        card = self.pair_cards.pop()
        self.pairs_layout.removeWidget(card)
        card.deleteLater()
        for i, existing in enumerate(self.pair_cards, start=1):
            existing.set_index(i)
        self.update_shape_plot()
        self._restore_maximized_if_needed(was_maximized)

    def on_save_coordinates_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Coordinate Inputs CSV",
            "",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            rows: List[Dict[str, str]] = [
                {
                    "type": "center",
                    "ideal_x": self.ideal_center_x.text().strip(),
                    "ideal_y": self.ideal_center_y.text().strip(),
                    "measured_x": self.measured_center_x.text().strip(),
                    "measured_y": self.measured_center_y.text().strip(),
                }
            ]
            for card in self.pair_cards:
                ix, iy, mx, my = card.values()
                rows.append(
                    {
                        "type": "pair",
                        "ideal_x": ix,
                        "ideal_y": iy,
                        "measured_x": mx,
                        "measured_y": my,
                    }
                )
            fieldnames = ["type", "ideal_x", "ideal_y", "measured_x", "measured_y"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self.statusBar().showMessage("Coordinate CSV saved")
            self._log(f"Saved coordinate inputs CSV: {path}")
        except Exception as exc:
            self._error(f"Failed to save coordinate CSV: {exc}")

    def on_load_coordinates_csv(self) -> None:
        was_maximized = self.isMaximized()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Coordinate Inputs CSV",
            "",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                required = {"type", "ideal_x", "ideal_y", "measured_x", "measured_y"}
                if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                    raise ValueError("CSV missing required columns.")
                rows = list(reader)

            center_rows = [r for r in rows if (r.get("type") or "").strip().lower() == "center"]
            if len(center_rows) != 1:
                raise ValueError("CSV must contain exactly one center row.")
            center = center_rows[0]

            ideal_x = (center.get("ideal_x") or "").strip()
            ideal_y = (center.get("ideal_y") or "").strip()
            if ideal_x == "" or ideal_y == "":
                raise ValueError("Center ideal coordinates cannot be blank.")
            ideal_center = (float(ideal_x), float(ideal_y))
            if ideal_center != (0.0, 0.0):
                raise ValueError("Ideal center in CSV must be (0,0).")

            measured_x = (center.get("measured_x") or "").strip()
            measured_y = (center.get("measured_y") or "").strip()
            if (measured_x == "") != (measured_y == ""):
                raise ValueError("Center measured coordinates must be both filled or both blank.")
            if measured_x:
                float(measured_x)
                float(measured_y)

            pair_rows = [r for r in rows if (r.get("type") or "").strip().lower() == "pair"]
            parsed_pairs: List[Tuple[str, str, str, str]] = []
            for idx, row in enumerate(pair_rows, start=1):
                ix = (row.get("ideal_x") or "").strip()
                iy = (row.get("ideal_y") or "").strip()
                mx = (row.get("measured_x") or "").strip()
                my = (row.get("measured_y") or "").strip()
                vals = [ix, iy, mx, my]
                non_empty = [v != "" for v in vals]
                if any(non_empty) and not all(non_empty):
                    raise ValueError(f"Pair row {idx} must have all four coordinates or be fully blank.")
                if all(non_empty):
                    for v in vals:
                        float(v)
                parsed_pairs.append((ix, iy, mx, my))

            self.ideal_center_x.setText("0")
            self.ideal_center_y.setText("0")
            self.measured_center_x.setText(measured_x)
            self.measured_center_y.setText(measured_y)

            while self.pair_cards:
                card = self.pair_cards.pop()
                self.pairs_layout.removeWidget(card)
                card.deleteLater()

            if not parsed_pairs:
                parsed_pairs = [("", "", "", "") for _ in range(3)]
            for vals in parsed_pairs:
                self.on_add_pair()
                card = self.pair_cards[-1]
                card.ideal_x.setText(vals[0])
                card.ideal_y.setText(vals[1])
                card.measured_x.setText(vals[2])
                card.measured_y.setText(vals[3])

            self.update_shape_plot()
            self.statusBar().showMessage("Coordinate CSV loaded")
            self._log(f"Loaded coordinate inputs CSV: {path}")
        except Exception as exc:
            self._error(f"Failed to load coordinate CSV: {exc}")
        finally:
            self._restore_maximized_if_needed(was_maximized)

    def on_rectify_point(self) -> None:
        if self.result is None:
            self._error("Solve calibration first.")
            return
        try:
            x = self._read_float(self.rect_in_x, "Measured X")
            y = self._read_float(self.rect_in_y, "Measured Y")
            eff = self.get_effective_forward_affine()
            assert eff is not None
            out = apply_inverse_transform((x, y), eff[0], eff[1])
            self.rect_out.setText(f"x={out[0]:.2f}, y={out[1]:.2f}")
            self.statusBar().showMessage("Point rectified")
            self._log(f"Rectified ({x}, {y}) -> ({out[0]:.6f}, {out[1]:.6f})")
        except Exception as exc:
            self._error(str(exc))

    def on_rectify_forward_point(self) -> None:
        if self.result is None:
            self._error("Solve calibration first.")
            return
        try:
            x = self._read_float(self.rect_fwd_in_x, "Ideal X")
            y = self._read_float(self.rect_fwd_in_y, "Ideal Y")
            eff = self.get_effective_forward_affine()
            assert eff is not None
            out = apply_transform((x, y), eff[0], eff[1])
            self.rect_fwd_out.setText(f"x={out[0]:.2f}, y={out[1]:.2f}")
            self.statusBar().showMessage("Forward verification complete")
            self._log(f"Forward verify ({x}, {y}) -> ({out[0]:.6f}, {out[1]:.6f})")
        except Exception as exc:
            self._error(str(exc))

    def on_add_verification_mark(self) -> None:
        if self.result is None:
            self._error("Solve calibration first.")
            return
        try:
            x = self._read_float(self.rect_in_x, "Measured X")
            y = self._read_float(self.rect_in_y, "Measured Y")
            self._verification_measured_points.append((x, y))
            n = len(self._verification_measured_points)
            measured_pt, ideal, comp, pred = self._compute_verification_geometry((x, y))
            self._log(
                f"Verification mark #{n} (measured input fixed; derived points track manual params): "
                f"measured=({measured_pt[0]:.6f},{measured_pt[1]:.6f}) "
                f"ideal=({ideal[0]:.6f},{ideal[1]:.6f}) "
                f"comp=({comp[0]:.6f},{comp[1]:.6f}) "
                f"pred=({pred[0]:.6f},{pred[1]:.6f})"
            )
            self.statusBar().showMessage(f"Verification mark #{n} added to plot")
            self.update_shape_plot()
        except Exception as exc:
            self._error(str(exc))

    def on_clear_verification_marks(self) -> None:
        if not self._verification_measured_points:
            return
        self._verification_measured_points.clear()
        self.statusBar().showMessage("Verification marks cleared")
        self.update_shape_plot()

    def on_browse_input_dxf(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input DXF", "", "DXF Files (*.dxf)")
        if not file_path:
            return
        self.dxf_input.setText(file_path)
        if not self.dxf_output.text().strip():
            in_path = Path(file_path)
            self.dxf_output.setText(str(in_path.with_name(f"{in_path.stem}_compensated.dxf")))

    def on_browse_output_dxf(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output DXF", "", "DXF Files (*.dxf)")
        if file_path:
            self.dxf_output.setText(file_path)

    def on_process_dxf(self) -> None:
        if self.result is None:
            self._error("Solve calibration first.")
            return
        try:
            input_path = Path(self.dxf_input.text().strip())
            output_path = Path(self.dxf_output.text().strip())
            if not input_path.exists():
                raise ValueError("Input DXF path does not exist.")
            if input_path.suffix.lower() != ".dxf":
                raise ValueError("Input must be a .dxf file.")
            if output_path.suffix.lower() != ".dxf":
                raise ValueError("Output must be a .dxf file.")

            eff = self.get_effective_forward_affine()
            assert eff is not None
            comp_a, comp_t = build_compensation_transform(eff[0], eff[1])
            stats = transform_dxf_with_compensation(
                input_path=input_path,
                output_path=output_path,
                comp_a=comp_a,
                comp_t=comp_t,
            )
            self.statusBar().showMessage("DXF processing complete")
            self._log(f"DXF output: {output_path}")
            self._log(f"Transformed entities: {stats.transformed_entities}")
            self._log(f"Converted entities: {stats.converted_entities}")
            self._log(f"Skipped entities: {stats.skipped_entities}")
            for warning in stats.warnings[:20]:
                self._log(f"Warning: {warning}")
            if len(stats.warnings) > 20:
                self._log(f"... and {len(stats.warnings) - 20} additional warnings")
            QMessageBox.information(self, "DXF Complete", f"Saved compensated DXF:\n{output_path}")
        except Exception as exc:
            self._error(str(exc))

    def on_show_dxf_comparison(self) -> None:
        if self.result is None:
            self._error("Solve calibration first.")
            return
        try:
            input_path = Path(self.dxf_input.text().strip())
            output_path = Path(self.dxf_output.text().strip())
            if not input_path.exists():
                raise ValueError("Input DXF path does not exist.")
            if not output_path.exists():
                raise ValueError("Output DXF path does not exist.")
            if input_path.suffix.lower() != ".dxf" or output_path.suffix.lower() != ".dxf":
                raise ValueError("Both paths must be .dxf files.")

            input_paths = extract_dxf_paths_for_plot(input_path)
            output_paths = extract_dxf_paths_for_plot(output_path)
            eff = self.get_effective_forward_affine()
            assert eff is not None
            distorted_paths = _apply_affine_to_paths(input_paths, eff[0], eff[1])

            if self._dxf_compare_dialog is None:
                self._dxf_compare_dialog = DxfCompareDialog(self)
            dlg = self._dxf_compare_dialog
            dlg.set_data(input_paths, output_paths, distorted_paths)
            dlg.refresh()
            dlg.showMaximized()
            dlg.raise_()
            dlg.activateWindow()
        except Exception as exc:
            self._error(str(exc))

    def on_export_report(self) -> None:
        if self.result is None:
            self._error("No calibration result to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Calibration Report", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            eff = self.get_effective_forward_affine()
            assert eff is not None
            a_eff, t_eff = eff
            lines = [
                "Affine Calibration Report",
                "=========================",
                "Solved (least-squares) A:",
                f"{np.array2string(self.result.matrix, precision=8)}",
                (
                    f"Solved t: tx={self.result.translation[0]:.8f}, "
                    f"ty={self.result.translation[1]:.8f}"
                ),
                (
                    f"Solved rotation={self.result.rotation_deg:.8f}, "
                    f"sx={self.result.scale_x:.8f}, sy={self.result.scale_y:.8f}, "
                    f"shx={self.result.shear_x:.8f}, shy={self.result.shear_y:.8f}"
                ),
                f"Solved rms={self.result.rms_error:.8f}",
                "",
                "Manual corrections (additive):",
                f"  dtx={self._adj_dtx:.8f}, dty={self._adj_dty:.8f}",
                f"  drot_deg={self._adj_drot:.8f}",
                f"  dsx={self._adj_dsx:.8f}, dsy={self._adj_dsy:.8f}, dshear={self._adj_dshear:.8f}",
                "",
                "Effective forward map (ideal → measured) with corrections:",
                f"A_eff:\n{np.array2string(a_eff, precision=8)}",
                f"t_eff: tx={t_eff[0]:.8f}, ty={t_eff[1]:.8f}",
                "",
                "Log:",
                self.log_box.toPlainText(),
            ]
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            self.statusBar().showMessage("Report exported")
            self._log(f"Report exported: {path}")
        except Exception as exc:
            self._error(str(exc))

    def on_reset(self) -> None:
        was_maximized = self.isMaximized()
        self.ideal_center_x.setText("0")
        self.ideal_center_y.setText("0")
        self.measured_center_x.clear()
        self.measured_center_y.clear()
        while self.pair_cards:
            card = self.pair_cards.pop()
            self.pairs_layout.removeWidget(card)
            card.deleteLater()
        for _ in range(3):
            self.on_add_pair()
        self.rect_in_x.clear()
        self.rect_in_y.clear()
        self.rect_out.setText("-")
        self.rect_fwd_in_x.clear()
        self.rect_fwd_in_y.clear()
        self.rect_fwd_out.setText("-")
        self.dxf_input.clear()
        self.dxf_output.clear()
        self.lbl_matrix.setText("-")
        self.lbl_translation.setText("-")
        self.lbl_pred_center.setText("-")
        self.lbl_params.setText("-")
        self.lbl_rms.setText("-")
        self.log_box.clear()
        self.result = None
        self._reset_user_adjustments()
        self._verification_measured_points.clear()
        self._set_actions_enabled(False)
        self.statusBar().showMessage("Reset complete")
        self.update_shape_plot()
        if self._plot_dialog is not None:
            self._plot_dialog.clear_dimension_state()
        self._restore_maximized_if_needed(was_maximized)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
