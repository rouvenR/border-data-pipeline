#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from __helpers.metrics_common import (
    parse_bool_flag,
    parse_cpu,
    parse_number,
    parse_ram_to_gib,
    split_run_tag,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_DIR = BASE_DIR / "inputs" / "training_data"
DEFAULT_PLOTS_DIR = BASE_DIR / "outputs" /  "plots" / "regressions"

STATIC_THROUGHPUT_CUTOFF = 0.0

COLUMN_UNITS: Dict[str, str] = {
    "incoming_throughput": "messages/sec",
    "outgoing_throughput": "messages/sec",
    "received_throughput": "messages/sec",
    "sent_throughput": "messages/sec",
    "ram_mean_consumption": "%",
    "ram_max_consumption": "%",
    "cpu_mean_consumption": "%",
    "cpu_max_consumption": "%",
    "message_size": "bytes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an in-memory linear regression model from one metrics CSV "
            "identified by timestamp."
        )
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Timestamp used to locate training_data/<TIMESTAMP>_metrics.csv.",
    )
    parser.add_argument(
        "--variable-column",
        required=True,
        help=(
            "Column name used as the regressed variable feature, e.g. "
            "message_size, number_of_messages, quality_of_service."
        ),
    )
    parser.add_argument(
        "--target",
        default="received_throughput",
        help=(
            "Target metric column. 'received_throughput' maps to "
            "received_throughput_mean and 'sent_throughput' maps to "
            "sent_throughput_mean."
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        default=str(DEFAULT_METRICS_DIR),
        help="Directory containing *_metrics.csv files (default: training_data).",
    )
    parser.add_argument(
        "--base-load",
        type=float,
        default=0.0,
        help=(
            "Constant base-load to subtract from regression predictions before "
            "applying the saturation cap. Result is clipped to 0. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--no-regression",
        action="store_true",
        help="Do not draw the regression line in the output visualization.",
    )
    return parser.parse_args()


def resolve_target_column(target_name: str, row: Dict[str, str]) -> Optional[str]:
    aliases = {
        "received_throughput": "received_throughput_mean",
        "sent_throughput": "sent_throughput_mean",
    }
    direct = target_name
    aliased = aliases.get(target_name, target_name)

    if direct in row:
        return direct
    if aliased in row:
        return aliased
    return None


def load_rows(metrics_path: Path) -> List[Tuple[Path, Dict[str, str]]]:
    results: List[Tuple[Path, Dict[str, str]]] = []

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            results.append((metrics_path, row))

    return results


def snake_to_title_case(s: str) -> str:
    """Convert snake_case to Title Case (e.g., 'message_size' -> 'Message Size')."""
    special_cases = {
        "ram": "RAM",
        "cpu": "CPU",
        "consumption": "Utilisation",
        "qos0": "QoS 0",
        "qos1": "QoS 1",
        "qos2": "QoS 2",
    }
    words = []
    for word in s.split("_"):
        lower_word = word.lower()
        if lower_word in special_cases:
            words.append(special_cases[lower_word])
        else:
            words.append(word.capitalize())
    return " ".join(words)


def get_label_with_units(col_name: str) -> str:
    """Get formatted label with units in brackets if available."""
    title = snake_to_title_case(col_name)
    # Try to find matching unit by checking column name prefixes
    for unit_key, unit_value in COLUMN_UNITS.items():
        if unit_key in col_name.lower():
            return f"{title} [{unit_value}]"
    return title


def save_regression_plot(
    output_dir: Path,
    timestamp: str,
    variable_col: str,
    x_rows: List[List[float]],
    y_values: List[float],
    intercept: float,
    coefs: List[float],
    target_col: str,
    excluded_variable_values: List[float],
    excluded_y_values: List[float],
    saturation_threshold: Optional[float],
    saturation_value: float,
    saturation_side: Optional[str],
    static_cutoff: float,
    static_value: float,
    base_load: float = 0.0,
    show_regression: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    variable_values = [row[2] for row in x_rows]
    cpu_values = [row[0] for row in x_rows]
    ram_values = [row[1] for row in x_rows]

    cpu_ref = sum(cpu_values) / len(cpu_values)
    ram_ref = sum(ram_values) / len(ram_values)

    # Extend x-axis range to cover both training and excluded (breached) points
    all_x = variable_values + (excluded_variable_values if excluded_variable_values else [])
    x_min = min(all_x)
    x_max = max(all_x)
    line_x_min = max(x_min, static_cutoff)
    line_x_max = x_max
    if saturation_threshold is not None:
        if saturation_side == "right":
            line_x_max = min(x_max, saturation_threshold)
        elif saturation_side == "left":
            line_x_min = max(x_min, static_cutoff, saturation_threshold)

    if line_x_max > line_x_min:
        x_line = [line_x_min + (line_x_max - line_x_min) * i / 99.0 for i in range(100)]
    else:
        x_line = [line_x_min]

    # Compute regression line (shown only on the non-breach side of threshold).
    y_line: List[float] = []
    if show_regression:
        for x in x_line:
            y_line.append(max(0.0, intercept + coefs[0] * cpu_ref + coefs[1] * ram_ref + coefs[2] * x - base_load))

    static_line_x_min = 0.0
    static_line_x_max = min(static_cutoff, x_max)
    static_line_y = max(0.0, static_value - base_load)

    y_pred = [intercept + coefs[0] * row[0] + coefs[1] * row[1] + coefs[2] * row[2] for row in x_rows]

    # Add mean points for duplicated x-axis values across experiments.
    grouped_y_by_x: Dict[float, List[float]] = {}
    for x_val, y_val in zip(variable_values, y_values):
        grouped_y_by_x.setdefault(x_val, []).append(y_val)

    mean_x_values = [x_val for x_val, ys in grouped_y_by_x.items() if len(ys) > 1]
    mean_y_values = [sum(grouped_y_by_x[x_val]) / len(grouped_y_by_x[x_val]) for x_val in mean_x_values]
    print(mean_y_values)
    plt.figure(figsize=(10, 6))

    plt.scatter(variable_values, y_values, alpha=0.8, label="Experiment", s=30, color="#0173B2")
    if mean_x_values and mean_y_values:
        plt.scatter(
            mean_x_values,
            mean_y_values,
            alpha=0.95,
            label="Mean",
            s=65,
            color="#8A2BE2",
            edgecolors="#4B0082",
            linewidths=0.8,
            zorder=4,
        )
    if excluded_variable_values and excluded_y_values:
        plt.scatter(
            excluded_variable_values,
            excluded_y_values,
            alpha=0.8,
            label="Experiment (resource-breach)",
            s=30,
            color="#CCCCCC",
        )
    # plt.scatter(variable_values, y_pred, alpha=0.6, label="predicted (sample)", s=25, marker="x")
    if show_regression:
        plt.plot(
            x_line,
            y_line,
            color="#DE8F05",
            linewidth=2.0,
            label=f"Regressed function ({int(static_cutoff)}..threshold)",
        )
    if static_line_x_max >= static_line_x_min:
        plt.hlines(
            static_line_y,
            static_line_x_min,
            static_line_x_max,
            color="#029E73",
            linewidth=2.0,
            label=f"Static value (0..{int(static_cutoff)})",
        )
    if saturation_threshold is not None:
        plt.axvline(
            saturation_threshold,
            color="#CC78BC",
            linestyle="--",
            linewidth=1.5,
            label=f"Resource breach threshold",
        )

    if excluded_variable_values:
        breach_start = min(excluded_variable_values)
        ax = plt.gca()
        x_left_limit, x_right_limit = ax.get_xlim()
        ax.axvspan(
            breach_start,
            x_right_limit,
            color="#F9D6D5",
            alpha=0.35,
            label="Resource breach area",
            zorder=0,
        )
        # Keep the original autoscaled limits so shading does not introduce extra margin.
        ax.set_xlim(x_left_limit, x_right_limit)

    plt.xlabel(get_label_with_units(variable_col))
    plt.ylabel(get_label_with_units(target_col))
    # plt.title(f"Linear Regression: {get_label_with_units(target_col)} vs {get_label_with_units(variable_col)}")
    
    if ("ram" in target_col.lower() or "cpu" in target_col.lower()) and not target_col.lower() == "cpu_factor_compared_to_min" and not target_col.lower() == "ram_factor_compared_to_min":
        plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    safe_variable = re.sub(r"[^A-Za-z0-9_.-]", "_", variable_col)
    safe_target = re.sub(r"[^A-Za-z0-9_.-]", "_", target_col)
    output_path = output_dir / f"{timestamp}__{safe_variable}__{safe_target}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_regression_factors_json(
    output_dir: Path,
    timestamp: str,
    variable_col: str,
    target_col: str,
    intercept: float,
    coefs: List[float],
    r2: float,
    backend: str,
    saturation_threshold: Optional[float],
    saturation_value: float,
    saturation_side: Optional[str],
    static_cutoff: float,
    static_value: float,
    linear_fit_rows: int,
    base_load: float = 0.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_variable = re.sub(r"[^A-Za-z0-9_.-]", "_", variable_col)
    safe_target = re.sub(r"[^A-Za-z0-9_.-]", "_", target_col)
    output_path = output_dir / f"{timestamp}__{safe_variable}__{safe_target}.json"

    payload = {
        "timestamp": timestamp,
        "target_column": target_col,
        "variable_column": variable_col,
        "backend": backend,
        "intercept": intercept,
        "coef_cpu": coefs[0],
        "coef_ram_gib": coefs[1],
        "coef_variable": coefs[2],
        "r2": r2,
        "saturation_threshold": saturation_threshold,
        "saturation_value": saturation_value,
        "saturation_side": saturation_side,
        "static_cutoff": static_cutoff,
        "static_value": static_value,
        "linear_fit_rows": linear_fit_rows,
        "base_load": base_load,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_path


def fit_with_sklearn(x_rows: List[List[float]], y_values: List[float]) -> Optional[Tuple[float, List[float], float]]:
    try:
        from sklearn.linear_model import LinearRegression  # type: ignore
    except Exception:
        return None

    model = LinearRegression()
    model.fit(x_rows, y_values)
    r2 = model.score(x_rows, y_values)
    return float(model.intercept_), [float(v) for v in model.coef_.tolist()], float(r2)


def fit_with_numpy(x_rows: List[List[float]], y_values: List[float]) -> Tuple[float, List[float], float]:
    import numpy as np

    x = np.array(x_rows, dtype=float)
    y = np.array(y_values, dtype=float)
    x_aug = np.column_stack([np.ones(len(x)), x])

    coeffs, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
    y_hat = x_aug @ coeffs

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    intercept = float(coeffs[0])
    weights = [float(v) for v in coeffs[1:]]
    return intercept, weights, r2


def compute_r2(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 1.0

    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y_true, y_pred))
    ss_tot = sum((actual - mean_y) ** 2 for actual in y_true)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.exists():
        raise SystemExit(f"Metrics dir does not exist: {metrics_dir}")

    metrics_path = metrics_dir / f"{args.timestamp}_metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file does not exist: {metrics_path}")

    rows = load_rows(metrics_path)

    x_rows: List[List[float]] = []
    y_values: List[float] = []
    excluded_variable_values: List[float] = []
    excluded_y_values: List[float] = []
    excluded_rows_count = 0
    capacity_limited_target_values: List[float] = []
    resolved_target_col: Optional[str] = None
    variable_col = args.variable_column
    breached_variable_values: List[float] = []

    if args.variable_column in ["incoming_throughput_qos0", "incoming_throughput_qos1", "incoming_throughput_qos2"]:
        global STATIC_THROUGHPUT_CUTOFF
        STATIC_THROUGHPUT_CUTOFF = 499.0

    for path, row in rows:
        run_tag = row.get("run_tag", "")
        _, config = split_run_tag(run_tag)

        variable_num = parse_number(row.get(variable_col, ""))
        cpu_num = parse_cpu(config, row.get("cpu", ""))
        ram_gib = parse_ram_to_gib(config, row.get("ram", ""))

        target_col = resolve_target_column(args.target, row)
        if target_col is None:
            raise SystemExit(
                f"Target column '{args.target}' not found in row columns for file: {path}"
            )
        if resolved_target_col is None:
            resolved_target_col = target_col
        y_val = parse_number(row.get(target_col, ""))

        is_capacity_limited = (
            parse_bool_flag(row.get("cpu_reached_maximum_capacity", ""))
            or parse_bool_flag(row.get("ram_reached_maximum_capacity", ""))
            or parse_bool_flag(row.get("throughput_reached_maximum_capacity", ""))
            or parse_bool_flag(row.get("message_loss_threshold_exceeded", ""))
        )

        if is_capacity_limited:
            excluded_rows_count += 1
            if y_val is not None:
                capacity_limited_target_values.append(y_val)
            if variable_num is not None and y_val is not None:
                excluded_variable_values.append(variable_num)
                excluded_y_values.append(y_val)
                breached_variable_values.append(variable_num)  
            print(f"Excluding row due to capacity breach: {row['run_tag']}")  
            continue

        if variable_num is None or cpu_num is None or ram_gib is None or y_val is None:
            print(f"Skipping row with missing/invalid data: {row}")
            continue

        x_rows.append([cpu_num, ram_gib, variable_num])
        y_values.append(y_val)

    if rows:
        sample_row = rows[0][1]
        if variable_col not in sample_row:
            raise SystemExit(
                f"Variable column '{variable_col}' not found in CSV columns. "
                f"Available columns: {', '.join(sample_row.keys())}"
            )

    if len(x_rows) < 2:
        raise SystemExit(
            "Not enough matched rows to fit regression (need at least 2 valid rows). "
            "Check template, CPU/RAM fields, and target availability."
        )

    # Auto-detect saturation side by comparing mean of breached vs valid variable values.
    # Breached mean > valid mean  → right-side saturation (high values breach).
    # Breached mean < valid mean  → left-side saturation (low values breach).
    saturation_threshold: Optional[float] = None
    saturation_side: Optional[str] = None
    saturation_value = (
        sum(capacity_limited_target_values) / len(capacity_limited_target_values)
        if capacity_limited_target_values
        else 100.0
    )
    if breached_variable_values:
        valid_variable_values = [row[2] for row in x_rows]
        mean_breached = sum(breached_variable_values) / len(breached_variable_values)
        mean_valid = sum(valid_variable_values) / len(valid_variable_values) if valid_variable_values else mean_breached
        if mean_breached >= mean_valid:
            saturation_side = "right"
            saturation_threshold = min(breached_variable_values)
        else:
            saturation_side = "left"
            saturation_threshold = max(breached_variable_values)

    linear_x_rows: List[List[float]] = []

    linear_y_values: List[float] = []
    for row, y_val in zip(x_rows, y_values):
        variable_value = row[2]
        if variable_value < STATIC_THROUGHPUT_CUTOFF:
            print(f"Excluding row with variable value {variable_value} below static cutoff {STATIC_THROUGHPUT_CUTOFF}")
            continue
        if saturation_threshold is not None:
            if saturation_side == "right" and variable_value > saturation_threshold:
                print(f"Excluding row with variable value {variable_value} above right-side saturation threshold {saturation_threshold}")
                continue
            if saturation_side == "left" and variable_value < saturation_threshold:
                print(f"Excluding row with variable value {variable_value} below left-side saturation threshold {saturation_threshold}")
                continue
        linear_x_rows.append(row)
        linear_y_values.append(y_val)

    if len(linear_x_rows) < 2:
        raise SystemExit(
            "Not enough matched rows to fit linear segment (need at least 2 valid rows at or above "
            f"{STATIC_THROUGHPUT_CUTOFF})."
        )

    low_band_targets = [y_val for row, y_val in zip(x_rows, y_values) if row[2] <= STATIC_THROUGHPUT_CUTOFF]
    static_value = (
        sum(low_band_targets) / len(low_band_targets)
        if low_band_targets
        else sum(linear_y_values) / len(linear_y_values)
    )

    fit = fit_with_sklearn(linear_x_rows, linear_y_values)
    if fit is None:
        intercept, coefs, _ = fit_with_numpy(linear_x_rows, linear_y_values)
        backend = "numpy"
    else:
        intercept, coefs, _ = fit
        backend = "sklearn"

    all_piecewise_predictions: List[float] = []
    for row in x_rows:
        variable_value = row[2]
        if variable_value <= STATIC_THROUGHPUT_CUTOFF:
            all_piecewise_predictions.append(static_value)
        else:
            all_piecewise_predictions.append(
                intercept + coefs[0] * row[0] + coefs[1] * row[1] + coefs[2] * variable_value
            )

    r2 = compute_r2(y_values, all_piecewise_predictions)

    plot_path = save_regression_plot(
        output_dir=DEFAULT_PLOTS_DIR,
        timestamp=args.timestamp,
        variable_col=variable_col,
        x_rows=x_rows,
        y_values=y_values,
        intercept=intercept,
        coefs=coefs,
        target_col=resolved_target_col or args.target,
        excluded_variable_values=excluded_variable_values,
        excluded_y_values=excluded_y_values,
        saturation_threshold=saturation_threshold,
        saturation_value=saturation_value,
        saturation_side=saturation_side,
        static_cutoff=STATIC_THROUGHPUT_CUTOFF,
        static_value=static_value,
        base_load=args.base_load,
        show_regression=not args.no_regression,
    )

    factors_path = save_regression_factors_json(
        output_dir=DEFAULT_PLOTS_DIR,
        timestamp=args.timestamp,
        variable_col=variable_col,
        target_col=resolved_target_col or args.target,
        intercept=intercept,
        coefs=coefs,
        r2=r2,
        backend=backend,
        saturation_threshold=saturation_threshold,
        saturation_value=saturation_value,
        saturation_side=saturation_side,
        static_cutoff=STATIC_THROUGHPUT_CUTOFF,
        static_value=static_value,
        linear_fit_rows=len(linear_x_rows),
        base_load=args.base_load,
    )
    print(f"Backend: {backend}")
    print(f"Metrics file: {metrics_path}")
    print(f"Variable column: {variable_col}")
    print(f"Matched rows: {len(x_rows)}")
    print(f"Excluded capacity-limited rows: {excluded_rows_count}")
    print(f"Static range rows (<= {int(STATIC_THROUGHPUT_CUTOFF)}): {len(low_band_targets)}")
    print(f"Linear fit rows (>={int(STATIC_THROUGHPUT_CUTOFF)}): {len(linear_x_rows)}")
    print(f"Target column: {resolved_target_col or args.target}")
    print("Features: [cpu, ram_gib, variable]")
    print(f"Static cutoff: {STATIC_THROUGHPUT_CUTOFF}")
    print(f"Static value: {static_value}")
    print(f"Intercept: {intercept}")
    print(f"coef_cpu: {coefs[0]}")
    print(f"coef_ram_gib: {coefs[1]}")
    print(f"coef_{variable_col.lower()}: {coefs[2]}")
    print(f"R^2 (piecewise): {r2}")
    print(f"Base load: {args.base_load}")
    print(f"Regression line enabled: {not args.no_regression}")
    if saturation_threshold is not None:
        print(f"Saturation side: {saturation_side}")
        print(f"Saturation threshold ({variable_col}): {saturation_threshold}")
        print(f"Saturation value: {saturation_value}")
    print(f"Regression plot: {plot_path}")
    print(f"Regression factors JSON: {factors_path}")


if __name__ == "__main__":
    main()
