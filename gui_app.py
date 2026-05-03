from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
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
    QMenu,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from affine_core import (
    AffineResult,
    DxfPlottedEntity,
    apply_inverse_transform,
    apply_transform,
    build_affine_result,
    build_compensation_transform,
    compose_user_adjustment,
    decompose_affine,
    extract_dxf_entities_for_plot,
    extract_dxf_paths_for_plot,
    transform_dxf_with_compensation,
)

Point = Tuple[float, float]


def _dist_point_to_segment_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-18:
        return apx * apx + apy * apy
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx, qy = ax + t * abx, ay + t * aby
    dx, dy = px - qx, py - qy
    return dx * dx + dy * dy


def _nearest_entity(
    x: float, y: float, entities: List[DxfPlottedEntity], thresh_data: float
) -> Optional[DxfPlottedEntity]:
    best: Optional[DxfPlottedEntity] = None
    best_d = float("inf")
    t2 = thresh_data * thresh_data
    for ent in entities:
        arr = ent.path
        if arr.shape[0] < 2:
            continue
        for i in range(arr.shape[0] - 1):
            ax_, ay_ = float(arr[i, 0]), float(arr[i, 1])
            bx, by = float(arr[i + 1, 0]), float(arr[i + 1, 1])
            d2 = _dist_point_to_segment_sq(x, y, ax_, ay_, bx, by)
            if d2 < best_d:
                best_d = d2
                best = ent
    if best is None or best_d > t2:
        return None
    return best


def _polyline_chain_endpoints(ent: DxfPlottedEntity, tol2: float) -> List[Tuple[float, float]]:
    """Endpoints used for chain connectivity (closed polylines collapse to one junction)."""
    arr = ent.path
    n = int(arr.shape[0])
    if n < 2:
        return []
    p0 = (float(arr[0, 0]), float(arr[0, 1]))
    p1 = (float(arr[-1, 0]), float(arr[-1, 1]))
    dx, dy = p0[0] - p1[0], p0[1] - p1[1]
    if dx * dx + dy * dy <= tol2:
        return [p0]
    return [p0, p1]


def _build_endpoint_chain_graph_spatial(
    entities: List[DxfPlottedEntity], tol2: float
) -> Dict[str, Set[str]]:
    """
    Adjacency when entity endpoints lie within sqrt(tol2). Spatial grid keeps this ~O(n)
    instead of O(n^2) endpoint pairs (critical for large DXFs / dense polylines).
    """
    graph: Dict[str, Set[str]] = {e.handle: set() for e in entities}
    if not entities:
        return graph
    tol = math.sqrt(max(tol2, 0.0))
    cell = max(tol, 1e-12)
    buckets: Dict[Tuple[int, int], List[Tuple[str, float, float]]] = defaultdict(list)

    for ent in entities:
        for px, py in _polyline_chain_endpoints(ent, tol2):
            ix = int(math.floor(px / cell))
            iy = int(math.floor(py / cell))
            buckets[(ix, iy)].append((ent.handle, px, py))

    for (ix, iy), pts in buckets.items():
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                other = buckets.get((ix + di, iy + dj))
                if not other:
                    continue
                for h1, x1, y1 in pts:
                    for h2, x2, y2 in other:
                        if h1 == h2:
                            continue
                        dx, dy = x1 - x2, y1 - y2
                        if dx * dx + dy * dy <= tol2:
                            graph[h1].add(h2)
                            graph[h2].add(h1)
    return graph


def _bbox_intersects_rect(
    bbox: Tuple[float, float, float, float], rx0: float, rx1: float, ry0: float, ry1: float
) -> bool:
    minx, maxx, miny, maxy = bbox
    sx0, sx1 = (rx0, rx1) if rx0 <= rx1 else (rx1, rx0)
    sy0, sy1 = (ry0, ry1) if ry0 <= ry1 else (ry1, ry0)
    if maxx < sx0 or minx > sx1 or maxy < sy0 or miny > sy1:
        return False
    return True


class _CollapsibleSection(QWidget):
    """Header row toggles visibility of a block of controls (for dense preview sidebars)."""

    def __init__(self, title: str, content: QWidget, *, start_open: bool = True) -> None:
        super().__init__()
        self._content = content
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(start_open)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if start_open else Qt.ArrowType.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.toggled.connect(self._on_toggled)
        root.addWidget(self._toggle)
        root.addWidget(self._content)
        self._content.setVisible(start_open)

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)


def _attach_plot_scroll_pan(
    canvas: object,
    ax: object,
    toolbar: Optional[object] = None,
    skip_pan_begin: Optional[Callable[[object], bool]] = None,
) -> None:
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
        if skip_pan_begin is not None and skip_pan_begin(event):
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
        self.setWindowTitle("Calibration")
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
        left_v.setSpacing(8)

        layers_inner = QWidget()
        layers_l = QVBoxLayout(layers_inner)
        layers_l.setContentsMargins(8, 4, 8, 8)
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
        sec_layers = _CollapsibleSection("Layers", layers_inner, start_open=True)

        self._adj_group = self._build_adjustment_panel()
        self._adj_group.setTitle("")
        self._adj_group.setFlat(True)
        sec_adj = _CollapsibleSection("Manual corrections (Δ on solved)", self._adj_group, start_open=True)

        self._reference_group = self._build_reference_panel()
        self._reference_group.setTitle("")
        self._reference_group.setFlat(True)
        sec_ref = _CollapsibleSection("Reference point & dimensions", self._reference_group, start_open=False)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(2, 2, 2, 2)
        scroll_layout.addWidget(sec_layers)
        scroll_layout.addWidget(sec_adj)
        scroll_layout.addWidget(sec_ref)
        scroll_layout.addStretch(1)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setWidget(scroll_content)
        left_v.addWidget(side_scroll, stretch=1)

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
        self._dimension_log.setMinimumHeight(72)
        self._dimension_log.setMaximumHeight(140)
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
        self._input_entities: List[DxfPlottedEntity] = []
        self._has_output_paths: bool = True
        self._press_px: Optional[Tuple[float, float]] = None
        self._press_xy: Optional[Tuple[float, float]] = None
        self._marquee_mode: bool = False
        self._marquee_corner: Optional[Tuple[float, float]] = None
        self._marquee_remove: bool = False
        self._marquee_rect_artist: Optional[object] = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        left = QWidget()
        left.setFixedWidth(260)
        left_v = QVBoxLayout(left)
        left_v.setSpacing(6)

        layers_inner = QWidget()
        layers_l = QVBoxLayout(layers_inner)
        layers_l.setContentsMargins(8, 4, 8, 8)
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
            layers_l.addWidget(cb)
        sec_layers = _CollapsibleSection("Layers", layers_inner, start_open=True)

        bbox_inner = QWidget()
        bbox_l = QVBoxLayout(bbox_inner)
        bbox_l.setContentsMargins(8, 4, 8, 8)
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
        sec_bbox = _CollapsibleSection("Bounding box", bbox_inner, start_open=False)

        sel_inner = QWidget()
        sel_l = QVBoxLayout(sel_inner)
        sel_l.setContentsMargins(8, 4, 8, 8)
        self._lbl_selection = QLabel("0 / 0 selected")
        sel_l.addWidget(self._lbl_selection)
        self._btn_sel_all = QPushButton("Select all")
        self._btn_sel_none = QPushButton("Deselect all")
        self._btn_sel_invert = QPushButton("Invert selection")
        self._btn_sel_all.clicked.connect(self._on_select_all)
        self._btn_sel_none.clicked.connect(self._on_select_none)
        self._btn_sel_invert.clicked.connect(self._on_select_invert)
        for b in (self._btn_sel_all, self._btn_sel_none, self._btn_sel_invert):
            sel_l.addWidget(b)
        sel_help = QLabel(
            "Main-window Inverse / Forward transform only the entities selected here.\n"
            "Click: replace selection. Ctrl+click: toggle.\n"
            "Shift+click: toggle endpoint-connected chain (off if chain already selected).\n"
            "Shift+drag: box add. Ctrl+Shift+drag: box remove.\n"
            "Ctrl+Shift+click (no drag): toggle one entity.\n"
            "Wheel zoom; drag pan (Shift disables pan for box)."
        )
        sel_help.setWordWrap(True)
        sel_l.addWidget(sel_help)
        sec_sel = _CollapsibleSection("Entity selection (input)", sel_inner, start_open=True)

        measure_inner = QWidget()
        measure_l = QVBoxLayout(measure_inner)
        measure_l.setContentsMargins(8, 4, 8, 8)
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
        self._measure_log.setMinimumHeight(72)
        self._measure_log.setMaximumHeight(150)
        measure_l.addWidget(self._measure_log)
        sec_measure = _CollapsibleSection("Measure", measure_inner, start_open=False)

        ref_inner = QWidget()
        ref_l = QVBoxLayout(ref_inner)
        ref_l.setContentsMargins(8, 4, 8, 8)
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
        self._dimension_log.setMinimumHeight(72)
        self._dimension_log.setMaximumHeight(130)
        ref_l.addWidget(self._dimension_log)
        sec_ref = _CollapsibleSection("Reference point", ref_inner, start_open=False)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)
        scroll_layout.setContentsMargins(2, 2, 2, 2)
        scroll_layout.addWidget(sec_layers)
        scroll_layout.addWidget(sec_bbox)
        scroll_layout.addWidget(sec_sel)
        scroll_layout.addWidget(sec_measure)
        scroll_layout.addWidget(sec_ref)
        scroll_layout.addStretch(1)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setWidget(scroll_content)
        left_v.addWidget(side_scroll, stretch=1)

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
            self._canvas.mpl_connect("button_press_event", self._on_selection_press)
            _attach_plot_scroll_pan(
                self._canvas, self._ax, self._toolbar, skip_pan_begin=self._skip_pan_for_marquee
            )
            self._canvas.mpl_connect("button_press_event", self._on_plot_click)
            self._canvas.mpl_connect("motion_notify_event", self._on_selection_motion)
            self._canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
            self._canvas.mpl_connect("button_release_event", self._on_selection_release)
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
            self._btn_sel_all.setEnabled(False)
            self._btn_sel_none.setEnabled(False)
            self._btn_sel_invert.setEnabled(False)
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

    def _draw_input_entities(self, ax: object) -> bool:
        sel: Set[str] = self._main._dxf_selected_handles
        drawn = False
        first = True
        for ent in self._input_entities:
            arr = ent.path
            if arr.shape[0] < 2:
                continue
            is_sel = ent.handle in sel
            color = "#ff7f0e" if is_sel else "#1f77b4"
            lw = 2.35 if is_sel else 1.15
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                color=color,
                linewidth=lw,
                alpha=0.95,
                label="Input DXF" if first else "_nolegend_",
            )
            first = False
            drawn = True
        return drawn

    def _collect_pickables_from_entities(self, entities: List[DxfPlottedEntity]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for e_idx, ent in enumerate(entities):
            arr = ent.path
            for i, row in enumerate(arr):
                out.append(
                    {
                        "source": "Input",
                        "path_index": e_idx,
                        "point_index": i,
                        "point": (float(row[0]), float(row[1])),
                        "label": f"Input {ent.handle} p{i+1}",
                    }
                )
        return out

    def _update_selection_label(self) -> None:
        n = len(self._main._dxf_selected_handles)
        m = len(self._input_entities)
        self._lbl_selection.setText(f"{n} / {m} selected")

    def _on_select_all(self) -> None:
        self._main._dxf_selected_handles = {e.handle for e in self._input_entities}
        self._update_selection_label()
        self.refresh(preserve_view=True)

    def _on_select_none(self) -> None:
        self._main._dxf_selected_handles.clear()
        self._update_selection_label()
        self.refresh(preserve_view=True)

    def _on_select_invert(self) -> None:
        all_h = {e.handle for e in self._input_entities}
        self._main._dxf_selected_handles = all_h - self._main._dxf_selected_handles
        self._update_selection_label()
        self.refresh(preserve_view=True)

    def _entity_pick_threshold(self) -> float:
        if self._ax is None:
            return 1e-6
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        span = max(float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]))
        return max(1e-9, span * 0.025)

    def _model_xy_extent(self) -> float:
        """Larger axis-aligned span of all input entities (drawing units), for tolerance caps."""
        ents = self._input_entities
        if not ents:
            return 1.0
        min_x = min(e.bbox[0] for e in ents)
        max_x = max(e.bbox[1] for e in ents)
        min_y = min(e.bbox[2] for e in ents)
        max_y = max(e.bbox[3] for e in ents)
        return max(max_x - min_x, max_y - min_y, 1e-12)

    def _chain_endpoint_glue_distance(self) -> float:
        """
        Max distance between endpoints that counts as one chain.

        Uses min(zoom-based pick radius, a tiny fraction of drawing extent). Otherwise a
        zoomed-out view makes the pick radius enormous in data units and unrelated geometry
        (or parallel rails like slot inner/outer) bridges into one component.
        """
        extent = self._model_xy_extent()
        pick = self._entity_pick_threshold()
        rel_cap = extent * 1e-6
        tol = min(pick, rel_cap)
        return max(tol, max(extent * 1e-14, 1e-15))

    def _chain_connection_tol_sq(self) -> float:
        d = self._chain_endpoint_glue_distance()
        return d * d

    def _touching_entity_graph(self) -> Dict[str, Set[str]]:
        return _build_endpoint_chain_graph_spatial(
            self._input_entities, self._chain_connection_tol_sq()
        )

    def _chain_handles_from(self, start_handle: str) -> Set[str]:
        """All entity handles in the same endpoint-touching component as start_handle."""
        graph = self._touching_entity_graph()
        if start_handle not in graph:
            return {start_handle}
        seen: Set[str] = {start_handle}
        stack = [start_handle]
        while stack:
            h = stack.pop()
            for nb in graph[h]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen

    def _shift_toggle_chain(self, hit_handle: str) -> None:
        """Shift+chain-select: add linked endpoints on first use; remove same chain if already fully selected."""
        handles = self._main._dxf_selected_handles
        ch = self._chain_handles_from(hit_handle)
        if ch <= handles:
            handles -= ch
        else:
            handles |= ch

    def _skip_pan_for_marquee(self, event: object) -> bool:
        if not self._input_entities:
            return False
        mods = QApplication.keyboardModifiers()
        return bool(mods & Qt.ShiftModifier)

    def _on_selection_press(self, event: object) -> None:
        if self._ax is None or getattr(event, "inaxes", None) is not self._ax:
            self._press_px = None
            self._press_xy = None
            self._marquee_mode = False
            return
        if getattr(event, "button", None) != 1:
            return
        if event.x is None or event.y is None:
            return
        self._press_px = (float(event.x), float(event.y))
        xd = getattr(event, "xdata", None)
        yd = getattr(event, "ydata", None)
        self._press_xy = (float(xd), float(yd)) if xd is not None and yd is not None else None
        mods = QApplication.keyboardModifiers()
        shift = bool(mods & Qt.ShiftModifier)
        self._marquee_mode = bool(shift and self._press_xy is not None and self._input_entities)
        self._marquee_remove = bool(mods & Qt.ControlModifier) and shift
        if self._marquee_mode and self._press_xy is not None:
            self._marquee_corner = (self._press_xy[0], self._press_xy[1])
            self._clear_marquee_artist()

    def _on_selection_motion(self, event: object) -> None:
        if not self._marquee_mode or self._ax is None or self._figure is None:
            return
        if getattr(event, "inaxes", None) is not self._ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        if self._marquee_corner is None:
            return
        x0, y0 = self._marquee_corner
        x1, y1 = float(event.xdata), float(event.ydata)
        self._clear_marquee_artist()
        try:
            from matplotlib.patches import Rectangle

            xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
            ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
            rect = Rectangle(
                (xmin, ymin),
                max(xmax - xmin, 1e-12),
                max(ymax - ymin, 1e-12),
                fill=False,
                edgecolor="#e377c2",
                linewidth=1.4,
                linestyle="--",
                zorder=20,
            )
            self._ax.add_patch(rect)
            self._marquee_rect_artist = rect
            self._canvas.draw_idle()
        except Exception:
            pass

    def _clear_marquee_artist(self) -> None:
        if self._marquee_rect_artist is not None and self._ax is not None:
            try:
                self._marquee_rect_artist.remove()
            except Exception:
                pass
        self._marquee_rect_artist = None

    def _finalize_marquee(self, x1: float, y1: float) -> None:
        if self._marquee_corner is None:
            return
        x0, y0 = self._marquee_corner
        self._clear_marquee_artist()
        handles = self._main._dxf_selected_handles
        for ent in self._input_entities:
            if not _bbox_intersects_rect(ent.bbox, x0, x1, y0, y1):
                continue
            if self._marquee_remove:
                handles.discard(ent.handle)
            else:
                handles.add(ent.handle)
        self._marquee_mode = False
        self._marquee_corner = None
        self._update_selection_label()
        self.refresh(preserve_view=True)

    def _on_selection_release(self, event: object) -> None:
        if getattr(event, "button", None) != 1:
            return
        if self._ax is None:
            self._press_px = None
            return
        if self._marquee_mode and self._press_xy is not None:
            xd = getattr(event, "xdata", None)
            yd = getattr(event, "ydata", None)
            if xd is not None and yd is not None and self._press_px is not None and event.x is not None and event.y is not None:
                dxp = float(event.x) - self._press_px[0]
                dyp = float(event.y) - self._press_px[1]
                if (dxp * dxp + dyp * dyp) ** 0.5 > 4.0:
                    self._finalize_marquee(float(xd), float(yd))
                else:
                    self._clear_marquee_artist()
                    self._marquee_mode = False
                    self._marquee_corner = None
                    if self._input_entities and not (self._measure_mode or self._dimension_mode):
                        hit = _nearest_entity(float(xd), float(yd), self._input_entities, self._entity_pick_threshold())
                        if hit is not None:
                            mods = QApplication.keyboardModifiers()
                            if (mods & Qt.ControlModifier) and (mods & Qt.ShiftModifier):
                                if hit.handle in self._main._dxf_selected_handles:
                                    self._main._dxf_selected_handles.discard(hit.handle)
                                else:
                                    self._main._dxf_selected_handles.add(hit.handle)
                            else:
                                self._shift_toggle_chain(hit.handle)
                            self._update_selection_label()
                            self.refresh(preserve_view=True)
            else:
                self._clear_marquee_artist()
                self._marquee_mode = False
                self._marquee_corner = None
            self._press_px = None
            return
        if self._measure_mode or self._dimension_mode or not self._input_entities:
            self._press_px = None
            return
        if self._press_px is None or event.x is None or event.y is None:
            return
        dxp = float(event.x) - self._press_px[0]
        dyp = float(event.y) - self._press_px[1]
        if (dxp * dxp + dyp * dyp) ** 0.5 > 5.0:
            self._press_px = None
            return
        xd = getattr(event, "xdata", None)
        yd = getattr(event, "ydata", None)
        if xd is None or yd is None:
            self._press_px = None
            return
        hit = _nearest_entity(float(xd), float(yd), self._input_entities, self._entity_pick_threshold())
        mods = QApplication.keyboardModifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)
        handles = self._main._dxf_selected_handles
        if hit is None:
            if not ctrl and not shift:
                handles.clear()
        elif ctrl:
            if hit.handle in handles:
                handles.discard(hit.handle)
            else:
                handles.add(hit.handle)
        elif shift:
            self._shift_toggle_chain(hit.handle)
        else:
            handles.clear()
            handles.add(hit.handle)
        self._update_selection_label()
        self.refresh(preserve_view=True)
        self._press_px = None

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
        if self._marquee_mode:
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
        self,
        input_paths: List[np.ndarray],
        output_paths: List[np.ndarray],
        distorted_paths: List[np.ndarray],
        *,
        input_entities: Optional[List[DxfPlottedEntity]] = None,
    ) -> None:
        self._input_paths = input_paths
        self._output_paths = output_paths
        self._distorted_paths = distorted_paths
        self._input_entities = list(input_entities or [])
        self._has_output_paths = len(output_paths) > 0
        if self._has_output_paths:
            self._chk_output.setEnabled(True)
        else:
            self._chk_output.setChecked(False)
            self._chk_output.setEnabled(False)
        has_dist = len(distorted_paths) > 0
        self._chk_distorted.setEnabled(has_dist)
        if not has_dist:
            self._chk_distorted.setChecked(False)
        ents = self._input_entities
        for b in (self._btn_sel_all, self._btn_sel_none, self._btn_sel_invert):
            b.setEnabled(bool(ents))
        self._pickable_points.clear()
        self._hover_pick = None
        self._measure_pending_point = None
        self._on_bbox_controls_changed()
        self._update_selection_label()

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
            if self._input_entities:
                any_drawn = self._draw_input_entities(ax) or any_drawn
                pickables.extend(self._collect_pickables_from_entities(self._input_entities))
                if self._chk_coords.isChecked():
                    pts: List[Tuple[float, float]] = []
                    for ent in self._input_entities:
                        for row in ent.path:
                            pts.append((float(row[0]), float(row[1])))
                    self._annotate_points(ax, pts, color="blue", prefix="In")
            else:
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
        self._main.update_dxf_preview_plot()
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
        self._dxf_preview_canvas: Optional[object] = None
        self._dxf_preview_figure: Optional[object] = None
        self._dxf_preview_ax: Optional[object] = None
        self._preview_stack: Optional[QStackedWidget] = None
        self._plot_dialog: Optional[CalibrationPlotDialog] = None
        self._dxf_compare_dialog: Optional[DxfCompareDialog] = None
        self._dxf_entity_catalog: List[DxfPlottedEntity] = []
        self._dxf_selected_handles: Set[str] = set()
        self._logs: List[str] = []
        # Additive manual tweaks on top of last Solve (forward map ideal → measured).
        self._adj_dtx = 0.0
        self._adj_dty = 0.0
        self._adj_drot = 0.0
        self._adj_dsx = 0.0
        self._adj_dsy = 0.0
        self._adj_dshear = 0.0
        # Measured-space inputs only; ideal/comp/predicted recomputed on each draw from current affine.
        self._verification_measured_points: List[Point] = []
        self._did_fit_window_to_screen: bool = False

        self._build_ui()
        self._set_actions_enabled(False)
        self.update_shape_plot()
        self.update_dxf_preview_plot()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._did_fit_window_to_screen:
            return
        self._did_fit_window_to_screen = True
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        margin = 20
        inner_max_w = max(400, avail.width() - margin)
        inner_max_h = max(300, avail.height() - margin)
        cw, ch = self.width(), self.height()
        if cw > inner_max_w or ch > inner_max_h:
            fw, fh = self.frameGeometry().width(), self.frameGeometry().height()
            extra_w = max(0, fw - cw)
            extra_h = max(0, fh - ch)
            self.resize(
                max(520, min(cw, inner_max_w - extra_w)),
                max(400, min(ch, inner_max_h - extra_h)),
            )
        fg = self.frameGeometry()
        dx = dy = 0
        if fg.right() > avail.right():
            dx -= fg.right() - avail.right()
        if fg.bottom() > avail.bottom():
            dy -= fg.bottom() - avail.bottom()
        if fg.left() < avail.left():
            dx += avail.left() - fg.left()
        if fg.top() < avail.top():
            dy += avail.top() - fg.top()
        if dx or dy:
            self.move(fg.x() + dx, fg.y() + dy)

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
        root.addLayout(self._build_button_bar())

        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())
        self.statusBar().hide()

    def _build_right_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_diagnostics_group(), stretch=0)
        layout.addWidget(self._build_shape_plot_group(), stretch=1)
        return column

    def _build_shape_plot_group(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Sunken)
        root = QVBoxLayout(frame)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        title_lbl = QLabel("Preview")
        tf = title_lbl.font()
        tf.setBold(True)
        title_lbl.setFont(tf)
        header.addWidget(title_lbl)
        header.addStretch(1)

        self._btn_preview_calibration = QPushButton("Calibration")
        self._btn_preview_dxf = QPushButton("DXF")
        self._btn_preview_calibration.setCheckable(True)
        self._btn_preview_dxf.setCheckable(True)
        self._btn_preview_calibration.setChecked(True)
        self._preview_mode_group = QButtonGroup(self)
        self._preview_mode_group.setExclusive(True)
        self._preview_mode_group.addButton(self._btn_preview_calibration, 0)
        self._preview_mode_group.addButton(self._btn_preview_dxf, 1)
        self._preview_mode_group.idClicked.connect(self._on_preview_mode_changed)
        header.addWidget(self._btn_preview_calibration)
        header.addWidget(self._btn_preview_dxf)
        root.addLayout(header)

        self._preview_stack = QStackedWidget()
        root.addWidget(self._preview_stack, stretch=1)

        self._mpl_canvas = None
        self._mpl_figure = None
        self._mpl_ax = None
        self._mpl_click_cid = None
        self._dxf_preview_canvas = None
        self._dxf_preview_figure = None
        self._dxf_preview_ax = None

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            cal_page = QWidget()
            cal_layout = QVBoxLayout(cal_page)
            cal_layout.setContentsMargins(0, 0, 0, 0)
            figure = Figure(figsize=(4.5, 4.0))
            ax = figure.add_subplot(111)
            canvas = FigureCanvasQTAgg(figure)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumHeight(220)
            canvas.setCursor(Qt.CursorShape.PointingHandCursor)
            canvas.setToolTip("Click to open full calibration view (zoom, pan, legend)")
            cal_layout.addWidget(canvas)
            self._mpl_figure = figure
            self._mpl_ax = ax
            self._mpl_canvas = canvas
            self._mpl_click_cid = canvas.mpl_connect(
                "button_press_event", self._on_embedded_plot_click
            )
            self._preview_stack.addWidget(cal_page)

            dxf_page = QWidget()
            dxf_layout = QVBoxLayout(dxf_page)
            dxf_layout.setContentsMargins(0, 0, 0, 0)
            dxf_fig = Figure(figsize=(4.5, 4.0))
            dxf_ax = dxf_fig.add_subplot(111)
            dxf_canvas = FigureCanvasQTAgg(dxf_fig)
            dxf_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            dxf_canvas.setMinimumHeight(220)
            dxf_canvas.setCursor(Qt.CursorShape.PointingHandCursor)
            dxf_canvas.setToolTip("Click to open DXF comparison (layers, preview && select)")
            dxf_canvas.mpl_connect("button_press_event", self._on_dxf_embedded_plot_click)
            dxf_layout.addWidget(dxf_canvas)
            self._dxf_preview_figure = dxf_fig
            self._dxf_preview_ax = dxf_ax
            self._dxf_preview_canvas = dxf_canvas
            self._preview_stack.addWidget(dxf_page)
        except Exception as exc:
            self._mpl_canvas = None
            self._mpl_figure = None
            self._mpl_ax = None
            self._mpl_click_cid = None
            self._dxf_preview_canvas = None
            self._dxf_preview_figure = None
            self._dxf_preview_ax = None
            msg = (
                f"Plot unavailable ({exc}).\nInstall matplotlib: pip install matplotlib"
            )
            err_cal = QLabel(msg)
            err_cal.setWordWrap(True)
            err_dxf = QLabel(msg)
            err_dxf.setWordWrap(True)
            self._preview_stack.addWidget(err_cal)
            self._preview_stack.addWidget(err_dxf)

        return frame

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
            ax.set_title("Calibration")
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

    def _on_preview_mode_changed(self, index: int) -> None:
        if self._preview_stack is None:
            return
        self._preview_stack.setCurrentIndex(index)
        if index == 0:
            self.update_shape_plot()
        else:
            self.update_dxf_preview_plot()

    def _on_dxf_embedded_plot_click(self, event: object) -> None:
        self.on_dxf_preview_clicked()

    def update_dxf_preview_plot(self) -> None:
        if (
            self._dxf_preview_ax is None
            or self._dxf_preview_canvas is None
            or self._dxf_preview_figure is None
        ):
            return
        ax = self._dxf_preview_ax
        ax.clear()
        ax.grid(True, alpha=0.35)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelbottom=False, labelleft=False)

        raw = self.dxf_input.text().strip()
        if not raw:
            ax.text(
                0.5,
                0.5,
                "Set Input DXF path\n(browse or type path)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            self._dxf_preview_figure.tight_layout()
            self._dxf_preview_canvas.draw_idle()
            return

        p = Path(raw).expanduser()
        if not p.exists():
            ax.text(
                0.5,
                0.5,
                "DXF file not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            self._dxf_preview_figure.tight_layout()
            self._dxf_preview_canvas.draw_idle()
            return
        if p.suffix.lower() != ".dxf":
            ax.text(
                0.5,
                0.5,
                "Not a .dxf file",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            self._dxf_preview_figure.tight_layout()
            self._dxf_preview_canvas.draw_idle()
            return

        try:
            entities = extract_dxf_entities_for_plot(p)
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                str(exc),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="gray",
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            self._dxf_preview_figure.tight_layout()
            self._dxf_preview_canvas.draw_idle()
            return

        if not entities:
            ax.text(
                0.5,
                0.5,
                "No plottable entities in DXF",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            self._dxf_preview_figure.tight_layout()
            self._dxf_preview_canvas.draw_idle()
            return

        sel = self._dxf_selected_handles
        drawn = False
        for ent in entities:
            path_ar = ent.path
            if path_ar.shape[0] < 2:
                continue
            picked = ent.handle in sel
            color = "#1f77b4" if picked else "#bbbbbb"
            lw = 1.35 if picked else 0.85
            ax.plot(
                path_ar[:, 0],
                path_ar[:, 1],
                color=color,
                linewidth=lw,
                solid_capstyle="round",
            )
            drawn = True

        if drawn:
            ax.relim()
            ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        self._dxf_preview_figure.tight_layout()
        self._dxf_preview_canvas.draw_idle()

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
        group = QGroupBox("DXF (inverse compensation / forward map)")
        layout = QGridLayout(group)

        self.dxf_input = QLineEdit()
        self.dxf_output = QLineEdit()
        output_path_row = QWidget()
        output_path_layout = QHBoxLayout(output_path_row)
        output_path_layout.setContentsMargins(0, 0, 0, 0)
        output_path_layout.setSpacing(6)
        output_path_layout.addWidget(self.dxf_output, stretch=1)
        self.btn_open_dxf_output_folder = QToolButton()
        self.btn_open_dxf_output_folder.setAutoRaise(True)
        self.btn_open_dxf_output_folder.setToolTip("Open folder containing this output file")
        self.btn_open_dxf_output_folder.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        )
        self.btn_open_dxf_output_folder.clicked.connect(self.on_open_dxf_output_folder)
        output_path_layout.addWidget(self.btn_open_dxf_output_folder)
        self.btn_browse_input = QPushButton("Browse Input")
        self.btn_browse_output = QPushButton("Browse Output")
        self.btn_inverse_dxf = QToolButton()
        self.btn_inverse_dxf.setText("Process/Inverse")
        self.btn_inverse_dxf.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_inverse_dxf.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._menu_dxf_mode = QMenu(self.btn_inverse_dxf)
        self._act_dxf_forward = self._menu_dxf_mode.addAction("Forward")
        self.btn_inverse_dxf.setMenu(self._menu_dxf_mode)
        self.btn_dxf_preview = QPushButton("Preview && Select…")

        inv_fwd_row = QWidget()
        inv_fwd_layout = QHBoxLayout(inv_fwd_row)
        inv_fwd_layout.setContentsMargins(0, 0, 0, 0)
        inv_fwd_layout.setSpacing(8)
        inv_fwd_layout.addWidget(self.btn_inverse_dxf)

        self.btn_browse_input.clicked.connect(self.on_browse_input_dxf)
        self.btn_browse_output.clicked.connect(self.on_browse_output_dxf)
        self.btn_inverse_dxf.clicked.connect(self.on_dxf_inverse)
        self._act_dxf_forward.triggered.connect(self.on_dxf_forward)
        self.btn_dxf_preview.clicked.connect(self.on_dxf_preview_clicked)
        self.dxf_input.editingFinished.connect(self._on_dxf_input_editing_finished)
        self.dxf_output.editingFinished.connect(self._on_dxf_output_editing_finished)

        layout.addWidget(QLabel("Input DXF:"), 0, 0)
        layout.addWidget(self.dxf_input, 0, 1)
        layout.addWidget(self.btn_browse_input, 0, 2)
        layout.addWidget(QLabel("Last output DXF:"), 1, 0)
        layout.addWidget(output_path_row, 1, 1)
        layout.addWidget(self.btn_browse_output, 1, 2)
        layout.addWidget(self.btn_dxf_preview, 2, 0)
        layout.addWidget(inv_fwd_row, 2, 2)
        return group

    def _build_button_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.btn_export = QPushButton("Export Report")
        self.btn_logs = QPushButton("Logs")
        self.btn_reset = QPushButton("Reset")
        self.btn_export.clicked.connect(self.on_export_report)
        self.btn_logs.clicked.connect(self.on_show_logs)
        self.btn_reset.clicked.connect(self.on_reset)
        self.lbl_latest_log = QLabel("-")
        self.lbl_latest_log.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_latest_log.setToolTip("Most recent log line")
        bar.addWidget(self.btn_export)
        bar.addWidget(self.btn_logs)
        bar.addWidget(self.btn_reset)
        bar.addStretch(1)
        bar.addWidget(self.lbl_latest_log, stretch=1)
        return bar

    def _set_actions_enabled(self, solved: bool) -> None:
        self.btn_rectify.setEnabled(solved)
        self.btn_rectify_fwd.setEnabled(solved)
        self.btn_add_verification.setEnabled(solved)
        self.btn_clear_verification.setEnabled(solved)
        self.btn_inverse_dxf.setEnabled(solved)
        self.btn_export.setEnabled(solved)
        self.btn_dxf_preview.setEnabled(True)

    def _log(self, text: str) -> None:
        lines = str(text).splitlines() or [""]
        self._logs.extend(lines)
        latest = lines[-1] if lines else ""
        self.lbl_latest_log.setText(latest if latest else "-")

    def on_show_logs(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Logs")
        dlg.resize(900, 500)
        layout = QVBoxLayout(dlg)
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setPlainText("\n".join(self._logs))
        layout.addWidget(box, stretch=1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn_close)
        layout.addLayout(row)
        dlg.exec()

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
        card = PairCardWidget(index=len(self.pair_cards) + 1, validator=self._validator)
        card.connect_plot_updates(self.update_shape_plot)
        self.pair_cards.append(card)
        self.pairs_layout.addWidget(card)
        self.update_shape_plot()

    def on_remove_pair(self) -> None:
        if not self.pair_cards:
            return
        card = self.pair_cards.pop()
        self.pairs_layout.removeWidget(card)
        card.deleteLater()
        for i, existing in enumerate(self.pair_cards, start=1):
            existing.set_index(i)
        self.update_shape_plot()

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

    def _reload_dxf_entity_catalog(self) -> None:
        try:
            raw = self.dxf_input.text().strip()
            if not raw:
                self._dxf_entity_catalog = []
                self._dxf_selected_handles.clear()
                return
            p = Path(raw)
            if not p.exists() or p.suffix.lower() != ".dxf":
                self._dxf_entity_catalog = []
                self._dxf_selected_handles.clear()
                return
            try:
                self._dxf_entity_catalog = extract_dxf_entities_for_plot(p)
            except Exception:
                self._dxf_entity_catalog = []
            self._dxf_selected_handles = {e.handle for e in self._dxf_entity_catalog}
        finally:
            self.update_dxf_preview_plot()

    def _apply_dxf_compare_data(self, *, show_dialog: bool = False) -> None:
        try:
            inp = Path(self.dxf_input.text().strip())
            if not inp.exists() or inp.suffix.lower() != ".dxf":
                return
            try:
                in_paths = extract_dxf_paths_for_plot(inp)
                entities = extract_dxf_entities_for_plot(inp)
            except Exception as exc:
                self._log(f"[DXF preview] {exc}")
                return
            self._dxf_entity_catalog = entities
            valid = {e.handle for e in entities}
            self._dxf_selected_handles.intersection_update(valid)
            if valid and not self._dxf_selected_handles:
                self._dxf_selected_handles = set(valid)
            out = Path(self.dxf_output.text().strip())
            out_paths: List[np.ndarray] = []
            if out.exists() and out.suffix.lower() == ".dxf":
                try:
                    out_paths = extract_dxf_paths_for_plot(out)
                except Exception as exc:
                    self._log(f"[DXF preview] output: {exc}")
            distorted: List[np.ndarray] = []
            eff = self.get_effective_forward_affine()
            if eff is not None and in_paths:
                distorted = _apply_affine_to_paths(in_paths, eff[0], eff[1])
            dlg = self._dxf_compare_dialog
            if dlg is None and show_dialog:
                dlg = DxfCompareDialog(self)
                self._dxf_compare_dialog = dlg
            if dlg is not None:
                dlg.set_data(in_paths, out_paths, distorted, input_entities=entities)
                if show_dialog:
                    dlg.refresh()
                    dlg.showMaximized()
                    dlg.raise_()
                    dlg.activateWindow()
                elif dlg.isVisible():
                    dlg.refresh()
        finally:
            self.update_dxf_preview_plot()

    def _on_dxf_input_editing_finished(self) -> None:
        self._reload_dxf_entity_catalog()
        self._apply_dxf_compare_data()

    def _on_dxf_output_editing_finished(self) -> None:
        self._apply_dxf_compare_data()

    def on_dxf_preview_clicked(self) -> None:
        self._apply_dxf_compare_data(show_dialog=True)

    def on_browse_input_dxf(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input DXF", "", "DXF Files (*.dxf)")
        if not file_path:
            return
        self.dxf_input.setText(file_path)
        if not self.dxf_output.text().strip():
            in_path = Path(file_path)
            self.dxf_output.setText(str(in_path.with_name(f"{in_path.stem}_inverted.dxf")))
        self._reload_dxf_entity_catalog()
        self._apply_dxf_compare_data()

    def on_browse_output_dxf(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output DXF", "", "DXF Files (*.dxf)")
        if file_path:
            self.dxf_output.setText(file_path)

    def on_open_dxf_output_folder(self) -> None:
        raw = self.dxf_output.text().strip()
        if not raw:
            QMessageBox.information(self, "Open folder", "Set an output DXF path first.")
            return
        path = Path(raw).expanduser()
        folder = path.parent if path.name else path
        try:
            folder = folder.resolve(strict=False)
        except OSError:
            QMessageBox.warning(self, "Open folder", f"Could not resolve folder for:\n{raw}")
            return
        if not folder.exists():
            QMessageBox.warning(
                self,
                "Open folder",
                f"This folder does not exist yet:\n{folder}",
            )
            return
        url = QUrl.fromLocalFile(str(folder))
        if not QDesktopServices.openUrl(url):
            self._error(f"Could not open folder:\n{folder}")

    def _run_dxf_export(self, *, use_inverse: bool) -> None:
        if self.result is None:
            self._error("Solve calibration first.")
            return
        try:
            input_path = Path(self.dxf_input.text().strip())
            if not input_path.exists():
                raise ValueError("Input DXF path does not exist.")
            if input_path.suffix.lower() != ".dxf":
                raise ValueError("Input must be a .dxf file.")

            suffix = "_inverted" if use_inverse else "_forwarded"
            output_path = input_path.with_name(f"{input_path.stem}{suffix}.dxf")

            eff = self.get_effective_forward_affine()
            assert eff is not None
            if use_inverse:
                comp_a, comp_t = build_compensation_transform(eff[0], eff[1])
            else:
                comp_a, comp_t = np.asarray(eff[0], dtype=float), np.asarray(eff[1], dtype=float)
            if not self._dxf_entity_catalog:
                raise ValueError("No selectable entities in this DXF. Open preview after choosing a valid input file.")
            if not self._dxf_selected_handles:
                raise ValueError(
                    "No entities selected. Open Preview && Select… and choose at least one entity."
                )
            only_handles = set(self._dxf_selected_handles)
            stats = transform_dxf_with_compensation(
                input_path=input_path,
                output_path=output_path,
                comp_a=comp_a,
                comp_t=comp_t,
                only_handles=only_handles,
            )
            self.dxf_output.setText(str(output_path))
            mode = "Inverse (pre-distort)" if use_inverse else "Forward (ideal→measured)"
            self.statusBar().showMessage(f"DXF {mode} complete")
            self._log(f"DXF {mode}: {output_path}")
            self._log(f"Transformed entities: {stats.transformed_entities}")
            self._log(f"Converted entities: {stats.converted_entities}")
            self._log(f"Skipped entities: {stats.skipped_entities}")
            if stats.untransformed_by_selection:
                self._log(f"Left unchanged (not selected): {stats.untransformed_by_selection}")
            for warning in stats.warnings[:20]:
                self._log(f"Warning: {warning}")
            if len(stats.warnings) > 20:
                self._log(f"... and {len(stats.warnings) - 20} additional warnings")
            QMessageBox.information(self, "DXF saved", f"{mode}\n{output_path}")
            self._apply_dxf_compare_data()
        except Exception as exc:
            self._error(str(exc))

    def on_dxf_inverse(self) -> None:
        self._run_dxf_export(use_inverse=True)

    def on_dxf_forward(self) -> None:
        self._run_dxf_export(use_inverse=False)

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
                "\n".join(self._logs),
            ]
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            self.statusBar().showMessage("Report exported")
            self._log(f"Report exported: {path}")
        except Exception as exc:
            self._error(str(exc))

    def on_reset(self) -> None:
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
        self._dxf_entity_catalog.clear()
        self._dxf_selected_handles.clear()
        self.lbl_matrix.setText("-")
        self.lbl_translation.setText("-")
        self.lbl_pred_center.setText("-")
        self.lbl_params.setText("-")
        self.lbl_rms.setText("-")
        self._logs.clear()
        self.lbl_latest_log.setText("-")
        self.result = None
        self._reset_user_adjustments()
        self._verification_measured_points.clear()
        self._set_actions_enabled(False)
        self.statusBar().showMessage("Reset complete")
        self.update_shape_plot()
        self.update_dxf_preview_plot()
        if self._plot_dialog is not None:
            self._plot_dialog.clear_dimension_state()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
