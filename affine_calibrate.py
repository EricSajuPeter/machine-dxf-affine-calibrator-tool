"""CLI runner for affine calibration and DXF compensation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from affine_core import (
    AffineResult,
    apply_inverse_transform,
    build_affine_result,
    build_compensation_transform,
    transform_dxf_with_compensation,
)

Point = Tuple[float, float]


def read_point(prompt_label: str) -> Point:
    """Read one point as two floats from stdin."""
    while True:
        raw = input(f"{prompt_label} (format: x y): ").strip().replace(",", " ")
        parts = [p for p in raw.split() if p]
        if len(parts) != 2:
            print("Invalid input. Please enter exactly two numbers, e.g. 12.5 -3.2")
            continue
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
            return (x_val, y_val)
        except ValueError:
            print("Invalid numbers. Try again.")


def read_optional_point(prompt_label: str) -> Point | None:
    while True:
        raw = input(f"{prompt_label} (format: x y, or blank to skip): ").strip().replace(",", " ")
        if raw == "":
            return None
        parts = [p for p in raw.split() if p]
        if len(parts) != 2:
            print("Invalid input. Please enter exactly two numbers, e.g. 12.5 -3.2")
            continue
        try:
            return (float(parts[0]), float(parts[1]))
        except ValueError:
            print("Invalid numbers. Try again.")


def read_target_or_command(prompt_label: str) -> Point | str:
    """Read a target point, or return command string ('q' or 'p')."""
    while True:
        raw = input(
            f"{prompt_label} (format: x y, or 'p' to process DXF, or 'q' to quit): "
        ).strip().replace(",", " ")
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            return "q"
        if lowered in {"p", "proceed"}:
            return "p"
        parts = [p for p in raw.split() if p]
        if len(parts) != 2:
            print("Invalid input. Please enter exactly two numbers, e.g. 12.5 -3.2")
            continue
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
            return (x_val, y_val)
        except ValueError:
            print("Invalid numbers. Try again.")


def read_calibration_pairs() -> Tuple[Point, Point | None, List[Point], List[Point]]:
    print("\nEnter center information first.")
    print("Ideal center defaults to (0,0). Press Enter to keep default.")
    ideal_center_raw = input("Ideal center (format: x y, or blank for 0 0): ").strip().replace(",", " ")
    if ideal_center_raw:
        parts = [p for p in ideal_center_raw.split() if p]
        if len(parts) != 2:
            raise ValueError("Invalid ideal center input.")
        ideal_center = (float(parts[0]), float(parts[1]))
    else:
        ideal_center = (0.0, 0.0)

    measured_center = read_optional_point("Measured center")
    print("\nEnter matched point pairs (minimum 3).")
    print("Type 'done' for ideal point when finished.")
    ideal_pairs: List[Point] = []
    measured_pairs: List[Point] = []
    idx = 1
    while True:
        raw = input(f"Ideal point p{idx} (x y, or 'done'): ").strip().replace(",", " ")
        if raw.lower() == "done":
            if len(ideal_pairs) < 3:
                print("Need at least 3 pairs before finishing.")
                continue
            break
        parts = [p for p in raw.split() if p]
        if len(parts) != 2:
            print("Invalid input. Please enter x y.")
            continue
        try:
            ideal_pt = (float(parts[0]), float(parts[1]))
        except ValueError:
            print("Invalid numbers. Try again.")
            continue
        measured_pt = read_point(f"Measured point p{idx}")
        ideal_pairs.append(ideal_pt)
        measured_pairs.append(measured_pt)
        idx += 1
    return ideal_center, measured_center, ideal_pairs, measured_pairs


def print_affine_result(result: AffineResult) -> None:
    print("\n=== Solved Affine Calibration ===")
    print("A (2x2 linear matrix):")
    print(np.array2string(result.matrix, precision=6, suppress_small=False))
    print(f"t (translation): tx={result.translation[0]:.6f}, ty={result.translation[1]:.6f}")
    print("\nDecomposed parameters:")
    print(f"  Rotation (deg): {result.rotation_deg:.6f}")
    print(f"  Scale X:        {result.scale_x:.6f}")
    print(f"  Scale Y:        {result.scale_y:.6f}")
    print(f"  Shear X:        {result.shear_x:.6f}")
    print(f"  Shear Y:        {result.shear_y:.6f}")
    print(f"  Linear solve rank: {result.rank}")
    if result.rank < 6:
        print("WARNING: Corner points are near-degenerate/collinear. Fit may be unstable.")
    print(
        f"Predicted measured center from transform: "
        f"x={result.predicted_measured_center[0]:.6f}, y={result.predicted_measured_center[1]:.6f}"
    )
    if result.provided_measured_center is not None and result.center_delta is not None:
        print(
            f"Provided measured center: x={result.provided_measured_center[0]:.6f}, "
            f"y={result.provided_measured_center[1]:.6f}"
        )
        print(
            f"Center delta (predicted - provided): "
            f"dx={result.center_delta[0]:.6f}, dy={result.center_delta[1]:.6f}"
        )

    print("\nResiduals (predicted - observed):")
    labels = [f"p{i+1}" for i in range(len(result.residual_vectors))]
    for i, (dx, dy) in enumerate(result.residual_vectors):
        print(f"  {labels[i]}: dx={dx:.6f}, dy={dy:.6f}")
    print(f"RMS error: {result.rms_error:.6f}")


def prompt_dxf_paths() -> Tuple[Path, Path]:
    while True:
        input_raw = input("Input ideal DXF path: ").strip().strip('"')
        input_path = Path(input_raw)
        if input_path.exists() and input_path.is_file():
            break
        print("Invalid file path. Please enter an existing .dxf file path.")

    suggested = input_path.with_name(f"{input_path.stem}_compensated.dxf")
    output_raw = input(
        f"Output DXF path [default: {suggested}]: "
    ).strip().strip('"')
    output_path = Path(output_raw) if output_raw else suggested
    return input_path, output_path


def process_dxf_flow(a_mat: np.ndarray, t_vec: np.ndarray) -> None:
    input_path, output_path = prompt_dxf_paths()
    comp_a, comp_t = build_compensation_transform(a_mat, t_vec)
    print("\nDXF compensation transform (ideal -> compensated):")
    print("A_comp (2x2):")
    print(np.array2string(comp_a, precision=6, suppress_small=False))
    print(f"t_comp: tx={comp_t[0]:.6f}, ty={comp_t[1]:.6f}")

    stats = transform_dxf_with_compensation(
        input_path=input_path,
        output_path=output_path,
        comp_a=comp_a,
        comp_t=comp_t,
    )

    print("\nDXF processing complete.")
    print(f"Output file: {output_path}")
    print(f"Transformed entities: {stats.transformed_entities}")
    print(f"Converted entities:   {stats.converted_entities}")
    print(f"Skipped entities:     {stats.skipped_entities}")
    if stats.warnings:
        print("\nWarnings:")
        for warning in stats.warnings[:20]:
            print(f"  - {warning}")
        if len(stats.warnings) > 20:
            print(f"  - ... and {len(stats.warnings) - 20} more warnings")
        print("Note: Converted curves/entities are approximated as polylines.")


def main() -> None:
    print("Affine calibration tool (ideal ↔ measured coordinates)")
    print("Point pairing order must match between ideal and measured points.")

    ideal_center, measured_center, ideal_pairs, measured_pairs = read_calibration_pairs()
    result = build_affine_result(
        ideal_center=ideal_center,
        ideal_points=ideal_pairs,
        measured_points=measured_pairs,
        provided_measured_center=measured_center,
    )
    print_affine_result(result)

    print("\nEnter measured coordinates to rectify back to ideal/computer.")
    print("Type 'p' to proceed with DXF reverse-distortion, or 'q' to stop.")
    while True:
        target_or_cmd = read_target_or_command("InputTargetCo-ordinate (measured space)")
        if isinstance(target_or_cmd, str) and target_or_cmd == "q":
            print("Exiting calibration tool.")
            break
        if isinstance(target_or_cmd, str) and target_or_cmd == "p":
            try:
                process_dxf_flow(result.matrix, result.translation)
            except RuntimeError as exc:
                print(f"DXF flow error: {exc}")
            except Exception as exc:
                print(f"DXF processing failed: {exc}")
            continue

        rectified = apply_inverse_transform(target_or_cmd, result.matrix, result.translation)
        print(
            f"Rectified coordinate (ideal/computer): x={rectified[0]:.2f}, y={rectified[1]:.2f}"
        )
        print(f"CSV: {rectified[0]:.2f},{rectified[1]:.2f}")


if __name__ == "__main__":
    main()
