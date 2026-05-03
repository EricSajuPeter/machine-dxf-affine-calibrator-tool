from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class AffineResult:
    matrix: np.ndarray
    translation: np.ndarray
    rotation_deg: float
    scale_x: float
    scale_y: float
    shear_x: float
    shear_y: float
    residual_vectors: np.ndarray
    rms_error: float
    rank: int
    predicted_measured_center: np.ndarray
    provided_measured_center: np.ndarray | None
    center_delta: np.ndarray | None


@dataclass
class DxfProcessStats:
    transformed_entities: int = 0
    converted_entities: int = 0
    skipped_entities: int = 0
    """Entities skipped due to errors, unsupported geometry, or missing handle when filtering."""
    untransformed_by_selection: int = 0
    """When ``only_handles`` is set: entities left unchanged because their handle was not selected."""
    warnings: List[str] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


@dataclass
class DxfPlottedEntity:
    """One modelspace entity reduced to a 2D polyline for preview and selection."""

    handle: str
    dxftype: str
    path: np.ndarray
    bbox: Tuple[float, float, float, float]
    """(min_x, max_x, min_y, max_y) in drawing units for marquee hit tests."""


def _axis_aligned_bbox(path: np.ndarray) -> Tuple[float, float, float, float]:
    xs = path[:, 0]
    ys = path[:, 1]
    return (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))


def _entity_handle_str(entity: Any) -> str:
    try:
        if entity.dxf.hasattr("handle"):
            h = entity.dxf.get("handle")
            if h is not None and str(h).strip() != "":
                return str(h).strip().upper()
    except Exception:
        pass
    return ""


def points_to_array(points: Sequence[Point]) -> np.ndarray:
    return np.asarray(points, dtype=float)


def solve_affine_from_pairs(
    ideal_points: Sequence[Point],
    measured_points: Sequence[Point],
) -> Tuple[np.ndarray, np.ndarray, int]:
    src = points_to_array(ideal_points)
    dst = points_to_array(measured_points)
    if src.shape != dst.shape:
        raise ValueError("Ideal and measured point counts must match.")
    if src.shape[0] < 3:
        raise ValueError("At least 3 matched point pairs are required.")
    if np.linalg.matrix_rank(src - src.mean(axis=0)) < 2:
        raise ValueError("Ideal points are collinear/degenerate; need non-collinear points.")

    n = src.shape[0]
    design = np.zeros((2 * n, 6), dtype=float)
    target = np.zeros((2 * n,), dtype=float)
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        design[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0]
        design[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0]
        target[2 * i] = u
        target[2 * i + 1] = v

    coeffs, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    a_mat = np.array([[coeffs[0], coeffs[1]], [coeffs[3], coeffs[4]]], dtype=float)
    t_vec = np.array([coeffs[2], coeffs[5]], dtype=float)
    return a_mat, t_vec, int(rank)


def decompose_affine(a_mat: np.ndarray, t_vec: np.ndarray) -> Tuple[float, float, float, float, float]:
    del t_vec  # translation is reported separately; decomposition is for linear part
    col0 = a_mat[:, 0]
    col1 = a_mat[:, 1]
    scale_x = float(np.linalg.norm(col0))
    scale_y = float(np.linalg.norm(col1))
    if scale_x < 1e-12 or scale_y < 1e-12:
        raise ValueError("Degenerate affine matrix: one scale is near zero.")
    rotation_deg = math.degrees(math.atan2(col0[1], col0[0]))
    shear_x = float(np.dot(col0 / scale_x, col1 / scale_y))
    shear_y = float(np.dot(col1 / scale_y, col0 / scale_x))
    return rotation_deg, scale_x, scale_y, shear_x, shear_y


def recompose_affine_linear(
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
    shear_dot: float,
    handedness: int,
) -> np.ndarray:
    """
    Rebuild the 2x2 linear part of an affine map using the same conventions as
    :func:`decompose_affine`: first column along ``scale_x * u(θ)``, second column
    ``scale_y * v`` where ``u·v = shear_dot`` and ``u ⟂ v`` in the right-handed sense.
    """
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    u = np.array([c, s], dtype=float)
    perp = np.array([-s, c], dtype=float)
    sigma = float(np.clip(shear_dot, -1.0, 1.0))
    v_mag_sq = max(0.0, 1.0 - sigma * sigma)
    v_mag = math.sqrt(v_mag_sq)
    h = float(handedness) if handedness in (-1, 1) else 1.0
    if v_mag < 1e-15:
        v = sigma * u
    else:
        v = sigma * u + h * v_mag * perp
    sx = max(float(scale_x), 1e-15)
    sy = max(float(scale_y), 1e-15)
    col0 = sx * u
    col1 = sy * v
    return np.column_stack([col0, col1])


def compose_user_adjustment(
    a_mat: np.ndarray,
    t_vec: np.ndarray,
    *,
    dtx: float = 0.0,
    dty: float = 0.0,
    d_rot_deg: float = 0.0,
    d_scale_x: float = 0.0,
    d_scale_y: float = 0.0,
    d_shear: float = 0.0,
    min_scale: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply additive manual tweaks on top of a solved **forward** affine (ideal → measured).

    ``dtx``/``dty`` are added to translation. Rotation, scale, and shear deltas adjust
    the decomposed linear parameters, then the 2×2 matrix is rebuilt.
    """
    rot_deg, sx, sy, shx, shy = decompose_affine(a_mat, t_vec)
    shear_dot = 0.5 * (shx + shy)
    det = float(np.linalg.det(a_mat))
    if abs(det) < 1e-14:
        handedness = 1
    else:
        handedness = 1 if det >= 0.0 else -1
    new_rot = rot_deg + d_rot_deg
    new_sx = max(min_scale, sx + d_scale_x)
    new_sy = max(min_scale, sy + d_scale_y)
    new_shear = float(np.clip(shear_dot + d_shear, -1.0, 1.0))
    a_new = recompose_affine_linear(new_rot, new_sx, new_sy, new_shear, handedness)
    t_new = np.asarray(t_vec, dtype=float) + np.array([dtx, dty], dtype=float)
    return a_new, t_new


def apply_transform(point: Point, a_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float)
    return a_mat @ p + t_vec


def apply_inverse_transform(point: Point, a_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float)
    return np.linalg.solve(a_mat, p - t_vec)


def build_compensation_transform(a_mat: np.ndarray, t_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a_inv = np.linalg.inv(a_mat)
    t_comp = -a_inv @ t_vec
    return a_inv, t_comp


def compute_residuals(
    ideal_points: Sequence[Point],
    measured_points: Sequence[Point],
    a_mat: np.ndarray,
    t_vec: np.ndarray,
) -> Tuple[np.ndarray, float]:
    src = points_to_array(ideal_points)
    dst = points_to_array(measured_points)
    predicted = (a_mat @ src.T).T + t_vec
    residuals = predicted - dst
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return residuals, rms


def build_affine_result(
    ideal_center: Point,
    ideal_points: Sequence[Point],
    measured_points: Sequence[Point],
    provided_measured_center: Point | None = None,
) -> AffineResult:
    a_mat, t_vec, rank = solve_affine_from_pairs(ideal_points, measured_points)
    rotation_deg, sx, sy, shx, shy = decompose_affine(a_mat, t_vec)
    residuals, rms = compute_residuals(
        ideal_points=ideal_points,
        measured_points=measured_points,
        a_mat=a_mat,
        t_vec=t_vec,
    )
    predicted_measured_center = apply_transform(ideal_center, a_mat, t_vec)
    provided_center_np = None
    center_delta = None
    if provided_measured_center is not None:
        provided_center_np = np.asarray(provided_measured_center, dtype=float)
        center_delta = predicted_measured_center - provided_center_np
    return AffineResult(
        matrix=a_mat,
        translation=t_vec,
        rotation_deg=rotation_deg,
        scale_x=sx,
        scale_y=sy,
        shear_x=shx,
        shear_y=shy,
        residual_vectors=residuals,
        rms_error=rms,
        rank=rank,
        predicted_measured_center=predicted_measured_center,
        provided_measured_center=provided_center_np,
        center_delta=center_delta,
    )


def ensure_ezdxf_available() -> Any:
    try:
        import ezdxf
    except ImportError as exc:
        raise RuntimeError(
            "DXF processing requires 'ezdxf'. Install it with: pip install ezdxf"
        ) from exc
    return ezdxf


def tx_point_2d(point: Sequence[float], a_mat: np.ndarray, t_vec: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray([float(point[0]), float(point[1])], dtype=float)
    out = a_mat @ arr + t_vec
    return float(out[0]), float(out[1])


def polyline_from_vertices(entity: Any) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if entity.dxftype() == "LWPOLYLINE":
        for p in entity.get_points("xy"):
            points.append((float(p[0]), float(p[1])))
        return points
    if entity.dxftype() == "POLYLINE":
        for v in entity.vertices:
            points.append((float(v.dxf.location.x), float(v.dxf.location.y)))
        return points
    return points


def approximate_entity_to_polyline(entity: Any, segments: int = 128) -> List[Tuple[float, float]]:
    kind = entity.dxftype()
    points: List[Tuple[float, float]] = []
    if kind in {"LWPOLYLINE", "POLYLINE"}:
        return polyline_from_vertices(entity)
    if kind == "LINE":
        s = entity.dxf.start
        e = entity.dxf.end
        return [(float(s.x), float(s.y)), (float(e.x), float(e.y))]
    if kind == "POINT":
        p = entity.dxf.location
        return [(float(p.x), float(p.y))]
    if kind in {"ARC", "CIRCLE"}:
        center = entity.dxf.center
        cx, cy = float(center.x), float(center.y)
        r = float(entity.dxf.radius)
        if kind == "CIRCLE":
            a0, a1 = 0.0, 360.0
        else:
            a0, a1 = float(entity.dxf.start_angle), float(entity.dxf.end_angle)
            if a1 <= a0:
                a1 += 360.0
        for i in range(segments + 1):
            a = math.radians(a0 + (a1 - a0) * i / segments)
            points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return points
    if kind == "ELLIPSE":
        center = entity.dxf.center
        major = entity.dxf.major_axis
        ratio = float(entity.dxf.ratio)
        start = float(entity.dxf.start_param)
        end = float(entity.dxf.end_param)
        if end <= start:
            end += 2.0 * math.pi
        mx, my = float(major.x), float(major.y)
        nx, ny = -my * ratio, mx * ratio
        for i in range(segments + 1):
            t = start + (end - start) * i / segments
            x = float(center.x) + mx * math.cos(t) + nx * math.sin(t)
            y = float(center.y) + my * math.cos(t) + ny * math.sin(t)
            points.append((x, y))
        return points
    if kind == "SPLINE":
        try:
            fit = entity.flattening(distance=0.2)
            for p in fit:
                points.append((float(p.x), float(p.y)))
            return points
        except Exception:
            pass
        for p in entity.control_points:
            points.append((float(p.x), float(p.y)))
        return points
    return points


def copy_basic_graphics(src_entity: Any, dst_entity: Any) -> None:
    for key in ("layer", "color", "linetype", "lineweight", "ltscale"):
        if src_entity.dxf.hasattr(key):
            dst_entity.dxf.set(key, src_entity.dxf.get(key))


def transform_dxf_with_compensation(
    input_path: Path,
    output_path: Path,
    comp_a: np.ndarray,
    comp_t: np.ndarray,
    only_handles: Optional[Set[str]] = None,
) -> DxfProcessStats:
    ezdxf = ensure_ezdxf_available()
    doc = ezdxf.readfile(str(input_path))
    msp = doc.modelspace()
    stats = DxfProcessStats()
    to_delete: List[Any] = []
    filter_on = only_handles is not None
    filter_set = {h.strip().upper() for h in only_handles} if filter_on and only_handles else set()

    for entity in list(msp):
        kind = entity.dxftype()
        if filter_on:
            eh = _entity_handle_str(entity)
            if not eh or eh not in filter_set:
                stats.untransformed_by_selection += 1
                continue
        try:
            if kind == "LINE":
                sx, sy = tx_point_2d((entity.dxf.start.x, entity.dxf.start.y), comp_a, comp_t)
                ex, ey = tx_point_2d((entity.dxf.end.x, entity.dxf.end.y), comp_a, comp_t)
                entity.dxf.start = (sx, sy, entity.dxf.start.z if hasattr(entity.dxf.start, "z") else 0.0)
                entity.dxf.end = (ex, ey, entity.dxf.end.z if hasattr(entity.dxf.end, "z") else 0.0)
                stats.transformed_entities += 1
            elif kind == "POINT":
                x, y = tx_point_2d((entity.dxf.location.x, entity.dxf.location.y), comp_a, comp_t)
                entity.dxf.location = (x, y, entity.dxf.location.z if hasattr(entity.dxf.location, "z") else 0.0)
                stats.transformed_entities += 1
            elif kind == "LWPOLYLINE":
                new_pts = []
                for px, py in entity.get_points("xy"):
                    x, y = tx_point_2d((px, py), comp_a, comp_t)
                    new_pts.append((x, y))
                closed = bool(entity.closed)
                entity.clear()
                entity.append_points(new_pts, format="xy")
                entity.closed = closed
                stats.transformed_entities += 1
            elif kind == "POLYLINE":
                for v in entity.vertices:
                    x, y = tx_point_2d((v.dxf.location.x, v.dxf.location.y), comp_a, comp_t)
                    v.dxf.location = (x, y, v.dxf.location.z if hasattr(v.dxf.location, "z") else 0.0)
                stats.transformed_entities += 1
            elif kind == "SPLINE":
                cps = []
                for p in entity.control_points:
                    x, y = tx_point_2d((p.x, p.y), comp_a, comp_t)
                    cps.append((x, y, p.z if hasattr(p, "z") else 0.0))
                if cps:
                    entity.control_points = cps
                    stats.transformed_entities += 1
                else:
                    raise ValueError("SPLINE has no control points.")
            else:
                points = approximate_entity_to_polyline(entity)
                if len(points) >= 2:
                    transformed = [tx_point_2d(p, comp_a, comp_t) for p in points]
                    new_e = msp.add_lwpolyline(transformed)
                    copy_basic_graphics(entity, new_e)
                    to_delete.append(entity)
                    stats.converted_entities += 1
                    stats.warnings.append(f"Converted {kind} to LWPOLYLINE approximation.")
                else:
                    stats.skipped_entities += 1
                    stats.warnings.append(f"Skipped unsupported entity: {kind}")
        except Exception as exc:
            points = approximate_entity_to_polyline(entity)
            if len(points) >= 2:
                transformed = [tx_point_2d(p, comp_a, comp_t) for p in points]
                new_e = msp.add_lwpolyline(transformed)
                copy_basic_graphics(entity, new_e)
                to_delete.append(entity)
                stats.converted_entities += 1
                stats.warnings.append(f"Converted {kind} after transform error: {exc}")
            else:
                stats.skipped_entities += 1
                stats.warnings.append(f"Skipped {kind} due to error: {exc}")

    for entity in to_delete:
        try:
            msp.delete_entity(entity)
        except Exception:
            pass

    doc.saveas(str(output_path))
    return stats


def extract_dxf_entities_for_plot(path: Path) -> List[DxfPlottedEntity]:
    """Modelspace entities as 2D polylines with stable handles for GUI selection."""
    ezdxf = ensure_ezdxf_available()
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    out: List[DxfPlottedEntity] = []
    warned_no_handle = False
    for entity in list(msp):
        pts = approximate_entity_to_polyline(entity)
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        handle = _entity_handle_str(entity)
        if not handle:
            if not warned_no_handle:
                # Rare for normal DXF; selection cannot target these until handles exist.
                warned_no_handle = True
            continue
        bbox = _axis_aligned_bbox(arr)
        out.append(
            DxfPlottedEntity(
                handle=handle,
                dxftype=str(entity.dxftype()),
                path=arr,
                bbox=bbox,
            )
        )
    return out


def extract_dxf_paths_for_plot(path: Path) -> List[np.ndarray]:
    """All modelspace paths suitable for plotting (includes entities without DXF handles)."""
    ezdxf = ensure_ezdxf_available()
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    paths: List[np.ndarray] = []
    for entity in list(msp):
        pts = approximate_entity_to_polyline(entity)
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        paths.append(arr)
    return paths
