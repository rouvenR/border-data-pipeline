#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# High Load
CPU_TRANSLATION = {
    "2": 1.0,
    "4": 0.93,
    "8": 0.7,
    "16": 0.58,
}

# Low Load
# CPU_TRANSLATION = {
#     "2": 1.0,
#     "4": 1.31,
#     "8": 0.95,
#     "16": 0.5,
# }

RAM_TRANSLATION = {
    "0.5": 1.5,
    "1": 1.0,
    "2": 0.64,
    "4": 0.48,
    "8": 0.22,
}

@dataclass
class RegressionComponent:
    model_path: Path
    target_column: str
    variable_column: str
    intercept: float
    coef_cpu: float
    coef_ram_gib: float
    coef_variable: float
    saturation_threshold: Optional[float]
    saturation_value: float
    saturation_side: Optional[str]
    static_cutoff: Optional[float]
    static_value: Optional[float]


def _is_message_size_factor_component(component: RegressionComponent) -> bool:
    return (
        "message_size" in component.variable_column.lower()
        and "factor_compared_to_min" in component.target_column.lower()
    )


def _evaluate_component_raw(
    component: RegressionComponent,
    cpu: float,
    ram_gib: float,
    variable_value: float,
    baseline: float,
) -> float:
    is_saturated = (
        component.saturation_threshold is not None
        and (
            (component.saturation_side == "right" and variable_value >= component.saturation_threshold)
            or (component.saturation_side == "left" and variable_value <= component.saturation_threshold)
        )
    )
    if is_saturated:
        return component.saturation_value

    uses_static_segment = (
        component.static_cutoff is not None
        and component.static_value is not None
        and variable_value <= component.static_cutoff
    )
    if uses_static_segment:
        return component.static_value - baseline

    # TODO remove coef_cpu and coef_ram_gib (seem useless)
    regressed_value = (
        component.intercept
        + component.coef_cpu * cpu
        + component.coef_ram_gib * ram_gib
        + component.coef_variable * variable_value
        - baseline
    )
    print(f"Regressed {variable_value} to {regressed_value:.4f}")
    return regressed_value



def _evaluate_factor_multiplier(component: RegressionComponent, message_size_value: float) -> float:
    is_saturated = (
        component.saturation_threshold is not None
        and (
            (component.saturation_side == "right" and message_size_value >= component.saturation_threshold)
            or (component.saturation_side == "left" and message_size_value <= component.saturation_threshold)
        )
    )
    if is_saturated:
        return component.saturation_value

    uses_static_segment = (
        component.static_cutoff is not None
        and component.static_value is not None
        and message_size_value <= component.static_cutoff
    )
    if uses_static_segment:
        return component.static_value

    return component.intercept + component.coef_variable * message_size_value


def _message_size_column_for_incoming_throughput(variable_column: str) -> Optional[str]:
    # incoming_throughput_qos1 -> message_size_qos1
    match = re.search(r"(qos[0-9]+)$", variable_column.lower())
    if not match:
        return None
    return f"message_size_{match.group(1)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load linear regression JSON components and validate predictions "
            "against a metrics CSV, reporting MAE, RMSE, and MaxError."
        )
    )
    parser.add_argument(
        "--model-dir",
        default="regressions/*.json",
        help=(
            "Glob pattern for component model JSON files "
            "(default: regressions/*.json)."
        ),
    )
    parser.add_argument(
        "--validate-against",
        required=True,
        metavar="METRICS_CSV",
        help=(
            "Path to a *_metrics.csv file. For each row the additive prediction "
            "is computed and compared to the target column, reporting MAE, RMSE, "
            "and MaxError."
        ),
    )
    parser.add_argument(
        "--include-capacity-limited",
        action="store_true",
        help="Include rows where CPU or RAM reached maximum capacity.",
    )
    parser.add_argument(
        "--base-load",
        type=float,
        default=0.0,
        help=(
            "Baseline value subtracted from each component output before clipping "
            "to 0 (default: 0.0)."
        ),
    )
    return parser.parse_args()


def load_component(path: Path) -> RegressionComponent:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    required_keys = [
        "target_column",
        "variable_column",
        "intercept",
        "coef_cpu",
        "coef_ram_gib",
        "coef_variable",
    ]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise SystemExit(f"Model JSON missing keys {missing}: {path}")

    return RegressionComponent(
        model_path=path,
        target_column=str(data["target_column"]),
        variable_column=str(data["variable_column"]),
        intercept=float(data["intercept"]),
        coef_cpu=float(data["coef_cpu"]),
        coef_ram_gib=float(data["coef_ram_gib"]),
        coef_variable=float(data["coef_variable"]),
        saturation_threshold=float(data["saturation_threshold"]) if data.get("saturation_threshold") is not None else None,
        saturation_value=float(data.get("saturation_value", 100.0)),
        saturation_side=str(data["saturation_side"]) if data.get("saturation_side") is not None else None,
        static_cutoff=float(data["static_cutoff"]) if data.get("static_cutoff") is not None else None,
        static_value=float(data["static_value"]) if data.get("static_value") is not None else None,
    )


def discover_components(model_glob: str) -> List[RegressionComponent]:
    matched_paths = sorted(Path(p) for p in glob.glob(model_glob))
    if not matched_paths:
        raise SystemExit(f"No model JSON files matched pattern: {model_glob}")

    components = [load_component(path) for path in matched_paths]

    return components


# ---------------------------------------------------------------------------
# CSV parsing helpers (mirrors train_throughput_regression.py)
# ---------------------------------------------------------------------------

def _parse_number(value: str) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _parse_ram_to_gib(csv_value: str) -> Optional[float]:
    token = csv_value.strip() if csv_value else ""
    if not token:
        return None
    token = token.strip().lower()
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)([a-z]*)$", token)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    factors: Dict[str, float] = {
        "": 1.0, "g": 1.0, "gb": 1.0, "gib": 1.0, "gi": 1.0,
        "m": 1.0 / 1024.0, "mb": 1.0 / 1024.0, "mib": 1.0 / 1024.0, "mi": 1.0 / 1024.0,
        "k": 1.0 / (1024.0 ** 2), "kb": 1.0 / (1024.0 ** 2), "kib": 1.0 / (1024.0 ** 2), "ki": 1.0 / (1024.0 ** 2),
        "b": 1.0 / (1024.0 ** 3),
    }
    factor = factors.get(unit)
    if factor is None:
        return None
    return value * factor


def _predict_total(
    components: List[RegressionComponent],
    cpu: float,
    ram_gib: float,
    variable_values: Dict[str, float],
    baseline: float,
) -> float:
    factor_components = [c for c in components if _is_message_size_factor_component(c)]
    active_components = [c for c in components if not _is_message_size_factor_component(c)]

    if len(factor_components) > 1:
        names = ", ".join(c.model_path.name for c in factor_components)
        raise SystemExit(f"Expected at most one message_size factor component, found: {names}")

    factor_component: Optional[RegressionComponent] = None
    if factor_components:
        factor_component = factor_components[0]

    total = 0.0
    target_col = components[0].target_column
    for component in active_components:
        print("\n")
        variable_value = variable_values[component.variable_column]

        raw = _evaluate_component_raw(component, cpu, ram_gib, variable_value, baseline)
        is_saturated = (
            component.saturation_threshold is not None
            and (
                (component.saturation_side == "right" and variable_value >= component.saturation_threshold)
                or (component.saturation_side == "left" and variable_value <= component.saturation_threshold)
            )
        )
        if is_saturated:
            print(f"Component {component.model_path.name}: raw={raw:.4f} (saturated)")
        elif (
            component.static_cutoff is not None
            and component.static_value is not None
            and variable_value <= component.static_cutoff
        ):
            print(
                f"Component {component.model_path.name}: raw={raw:.4f} "
                f"(static <= {component.static_cutoff})"
            )
        else:
            print(f"Component {component.model_path.name}: raw={raw:.4f}")

        if "incoming_throughput" in component.variable_column.lower():
            factor_multiplier = 1.0
            if factor_component is not None:
                factor_input_col = _message_size_column_for_incoming_throughput(component.variable_column)
                if factor_input_col and factor_input_col in variable_values:
                    factor_input_value = variable_values[factor_input_col]
                else:
                    # Fallback to the factor component's own variable if no qos-specific message size was found.
                    factor_input_value = variable_values[factor_component.variable_column]

                factor_multiplier = _evaluate_factor_multiplier(factor_component, factor_input_value)
                print(
                    f"Component {factor_component.model_path.name}: raw={factor_multiplier:.4f} "
                    f"(message_size factor for {component.variable_column})"
                )
            raw_previous = raw
            raw *= factor_multiplier
            print(
                f"Applied message_size factor multiplier: {raw_previous:.4f} * {factor_multiplier:.4f} = {raw:.4f}"
            )

        if "cpu" in target_col.lower():
            raw_previous = raw
            raw *= CPU_TRANSLATION.get(str(int(cpu)), 1.0)
            print(f"Applied CPU translation factor for {int(cpu)} vCPU: {raw_previous:.4f} * {CPU_TRANSLATION.get(str(int(cpu)), 1.0):.4f} = {raw:.4f}")

        if "ram" in target_col.lower():
            raw_previous = raw
            raw *= RAM_TRANSLATION.get(f"{int(ram_gib)}", 1.0)
            print(f"Applied RAM translation factor for {int(ram_gib)} GiB: {raw_previous:.4f} * {RAM_TRANSLATION.get(f'{int(ram_gib)}', 1.0):.4f} = {raw:.4f}")

        clipped = max(0.0, raw)
        total += clipped
    return min(total+baseline, 100.0)


def validate_against_csv(
    csv_path: Path,
    components: List[RegressionComponent],
    include_capacity_limited: bool,
    baseline: float,
    target_col: Optional[str] = None,
) -> None:
    if not csv_path.exists():
        raise SystemExit(f"Validation CSV does not exist: {csv_path}")

    target_col = components[0].target_column
    variable_cols = sorted({c.variable_column for c in components})
    for qos in (0, 1, 2):
        variable_cols.append(f"message_size_qos{qos}")
    variable_cols = sorted(set(variable_cols))

    errors: List[float] = []
    skipped = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            print(f"\nEvaluating row {row.get('run_tag', 'N/A')} with {target_col}={row.get(target_col, 'N/A')}")
            if not include_capacity_limited and (
                row.get("cpu_reached_maximum_capacity", "").strip().lower() == "true"
                or row.get("ram_reached_maximum_capacity", "").strip().lower() == "true"
                or row.get("throughput_reached_maximum_capacity", "").strip().lower() == "true"
                or row.get("message_loss_threshold_exceeded", "") == "true"
            ):
                skipped += 1
                print("Skipping row due to capacity limits.")
                continue

            cpu_val = _parse_number(row.get("cpu", ""))
            ram_val = _parse_ram_to_gib(row.get("ram", ""))
            actual = _parse_number(row.get(target_col, ""))

            if cpu_val is None or ram_val is None or actual is None:
                skipped += 1
                continue

            variable_values: Dict[str, float] = {}
            valid = True
            for col in variable_cols:
                val = _parse_number(row.get(col, ""))
                if val is None:
                    valid = False
                    break
                variable_values[col] = val

            if not valid:
                skipped += 1
                continue

            predicted = _predict_total(components, cpu_val, ram_val, variable_values, baseline)

            if predicted == 100.0:
                print("\n\n== Predicted Saturation ==\n\n")
            print(f"===== Error: {predicted - actual} ({target_col}={actual:.2f}, predicted={predicted:.2f}) =====")
            
            errors.append(abs(predicted - actual))

    if not errors:
        raise SystemExit("No valid rows found in validation CSV for computing metrics.")

    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    max_err = max(errors)

    print(json.dumps({
        "validation_csv": str(csv_path),
        "target_column": target_col,
        "component_count": len(components),
        "rows_evaluated": len(errors),
        "rows_skipped": skipped,
        "baseline": baseline,
        "mae": mae,
        "rmse": rmse,
        "max_error": max_err,
    }, indent=2))


def main() -> None:
    args = parse_args()
    components = discover_components(args.model_dir)
    validate_against_csv(
        Path(args.validate_against),
        components,
        args.include_capacity_limited,
        args.base_load
    )


if __name__ == "__main__":
    main()
