#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_DIR = BASE_DIR / "inputs/training_data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs/plots"

INPUT_COLUMNS = [
    "cpu",
    "ram",
    "message_size_qos0",
    "message_size_qos1",
    "message_size_qos2",
    "number_of_clients_qos0",
    "number_of_clients_qos1",
    "number_of_clients_qos2",
    "delay_qos0",
    "delay_qos1",
    "delay_qos2",
    "incoming_throughput_qos0",
    "incoming_throughput_qos1",
    "incoming_throughput_qos2",
]

OUTPUT_COLUMNS = [
    "received_throughput_mean",
    "cpu_mean_consumption",
    "ram_mean_consumption",
    "message_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot input-to-output correlation heatmap from metrics CSV."
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Timestamp used to locate training_data/<TIMESTAMP>_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save heatmap (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = DEFAULT_METRICS_DIR / f"{args.timestamp}_metrics.csv"

    if not metrics_path.exists():
        raise SystemExit(f"Metrics file does not exist: {metrics_path}")

    # Load CSV
    df = pd.read_csv(metrics_path)

    # Check for required columns
    missing_inputs = [col for col in INPUT_COLUMNS if col not in df.columns]
    missing_outputs = [col for col in OUTPUT_COLUMNS if col not in df.columns]

    if missing_inputs:
        raise SystemExit(f"Missing input columns: {', '.join(missing_inputs)}")
    if missing_outputs:
        raise SystemExit(f"Missing output columns: {', '.join(missing_outputs)}")

    # Convert selected columns to numeric so correlation works even if values are strings.
    for column in INPUT_COLUMNS + OUTPUT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Compute pairwise correlation matrix (inputs x outputs).
    corr_df = pd.DataFrame(index=INPUT_COLUMNS, columns=OUTPUT_COLUMNS, dtype=float)
    for input_col in INPUT_COLUMNS:
        for output_col in OUTPUT_COLUMNS:
            valid = df[[input_col, output_col]].dropna()
            if len(valid) < 2:
                corr_df.loc[input_col, output_col] = np.nan
                continue

            # Correlation is undefined for constant columns.
            if valid[input_col].nunique() < 2 or valid[output_col].nunique() < 2:
                corr_df.loc[input_col, output_col] = np.nan
                continue

            corr_df.loc[input_col, output_col] = valid[input_col].corr(valid[output_col])

    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation Coefficient"},
        linewidths=0.5,
    )
    plt.title(f"Input-to-Output Correlations ({args.timestamp})")
    plt.xlabel("Output Metrics")
    plt.ylabel("Input Features")
    plt.tight_layout()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save figure
    output_path = args.output_dir / f"{args.timestamp}_correlation_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Correlation heatmap saved to: {output_path}")

    # Print correlation summary
    print("\nCorrelation Summary:")
    print(corr_df.to_string())


if __name__ == "__main__":
    main()
