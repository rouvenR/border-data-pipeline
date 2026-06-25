#!/usr/bin/env python3

"""Categorical Latin hypercube sampling for exactly 11 input dimensions.

This script generates N samples where each dimension is stratified through a
Latin hypercube permutation and then mapped to user-provided categorical values.
For categorical dimensions, this produces balanced (or near-balanced) usage of
categories across samples. Throughput is provided per QoS level and replaced by
derived per-QoS delay/messages columns in the output.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import List, Sequence


def parse_categories(raw: str) -> List[str]:
	values = [v.strip() for v in raw.split(",") if v.strip()]
	if not values:
		raise argparse.ArgumentTypeError("Each dimension must contain at least one category")
	return values


DIMENSION_NAMES = [
	"clients_qos0", "throughput_qos0", "size_qos0",
	"clients_qos1", "throughput_qos1", "size_qos1",
	"clients_qos2", "throughput_qos2", "size_qos2",
	"cpu",
]
OUTPUT_COLUMNS = [
	"clients_qos0", "delay_qos0", "messages_qos0", "size_qos0",
	"clients_qos1", "delay_qos1", "messages_qos1", "size_qos1",
	"clients_qos2", "delay_qos2", "messages_qos2", "size_qos2",
	"cpu", "ram_limit",
]

CPU_TO_RAM_LIMIT = {
	"2": "1g",
	"4": "2g",
	"8": "4g",
	"16": "8g",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate categorical Latin hypercube samples with per-QoS load dimensions"
	)
	parser.add_argument(
		"--samples",
		type=int,
		required=True,
		help="Number of samples to generate",
	)
	for name in DIMENSION_NAMES:
		parser.add_argument(
			f"--{name}",
			type=parse_categories,
			required=True,
			help=f"Comma-separated categories for {name}",
		)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional random seed for reproducibility",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional output CSV path",
	)
	return parser.parse_args()


def stratum_to_category(stratum_index: int, n_samples: int, categories: Sequence[str]) -> str:
	# Maps stratum index [0, n_samples-1] to category index [0, k-1] proportionally.
	k = len(categories)
	cat_index = min(k - 1, math.floor((stratum_index * k) / n_samples))
	return categories[cat_index]


def categorical_lhs(
	n_samples: int, dims: Sequence[Sequence[str]], rng: random.Random
) -> List[List[str]]:
	if n_samples <= 0:
		raise ValueError("samples must be > 0")
	if len(dims) != len(DIMENSION_NAMES):
		raise ValueError(f"Exactly {len(DIMENSION_NAMES)} dimensions are required")

	permutations: List[List[int]] = []
	base = list(range(n_samples))

	# For each dimension, generate shuffled list of indices like (1, 0, 4, 2, 3) for n_samples=5, for example. 
	for _ in range(len(DIMENSION_NAMES)):
		p = base[:]
		rng.shuffle(p)
		permutations.append(p)

	rows: List[List[str]] = []
	for i in range(n_samples):
		row = []
		for d in range(len(DIMENSION_NAMES)):
			# Get category by essential modulo operator of number of dimensions k
			# E.g. k = 2 (1, 0, 4, 2, 3) -> (1, 0, 0, 0, 1)
			# -> ensures wide spread of categories throughout the samples
			row.append(stratum_to_category(permutations[d][i], n_samples, dims[d]))
		rows.append(row)
	return rows


def print_rows(rows: Sequence[Sequence[str]]) -> None:
	print(",".join(OUTPUT_COLUMNS))
	for row in rows:
		print("\"" + " ".join(row) + "\"")


def write_csv(output: Path, rows: Sequence[Sequence[str]]) -> None:
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(OUTPUT_COLUMNS)
		writer.writerows(rows)


def format_number(value: float) -> str:
	return str(math.floor(value))


def cpu_to_ram_limit(cpu_value: str) -> str:
	ram_limit = CPU_TO_RAM_LIMIT.get(cpu_value)
	return ram_limit


def postprocess_throughput(rows: Sequence[Sequence[str]]) -> List[List[str]]:
	processed_rows: List[List[str]] = []

	# TODO this depends on the number of subscribers per publisher, configured in start_clients.sh
	N_SUBSCRIBER_MULTIPLIER = 5
	
	# For experiment configuration, throughput has to be translated to delay and messages per QoS level
	for row in rows:
		clients_qos0 = float(row[0])
		throughput_qos0 = float(row[1])
		clients_qos1 = float(row[3])
		throughput_qos1 = float(row[4])
		clients_qos2 = float(row[6])
		throughput_qos2 = float(row[7])

		if clients_qos0 <= 0 or clients_qos1 <= 0 or clients_qos2 <= 0:
			raise ValueError("clients_qos0/1/2 must be > 0 to compute delay/messages")
		if throughput_qos0 <= 0 or throughput_qos1 <= 0 or throughput_qos2 <= 0:
			raise ValueError("throughput_qos0/1/2 must be > 0 to compute delay/messages")

		print("Desired throughput per QoS level:", throughput_qos0, throughput_qos1, throughput_qos2)
		delay_qos0 = int(((60 *1000.0 / throughput_qos0) * clients_qos0 * N_SUBSCRIBER_MULTIPLIER)/60)
		delay_qos1 = int(((60 *1000.0 / throughput_qos1) * clients_qos1 * N_SUBSCRIBER_MULTIPLIER)/60)
		delay_qos2 = int(((60 *1000.0 / throughput_qos2) * clients_qos2 * N_SUBSCRIBER_MULTIPLIER)/60)

		messages_qos0 = (throughput_qos0 / (clients_qos0 * N_SUBSCRIBER_MULTIPLIER)) * 300.0
		messages_qos1 = (throughput_qos1 / (clients_qos1 * N_SUBSCRIBER_MULTIPLIER)) * 300.0
		messages_qos2 = (throughput_qos2 / (clients_qos2 * N_SUBSCRIBER_MULTIPLIER)) * 300.0

		processed_rows.append(
			[
				row[0],
				format_number(delay_qos0),
				format_number(messages_qos0),
				row[2],
				row[3],
				format_number(delay_qos1),
				format_number(messages_qos1),
				row[5],
				row[6],
				format_number(delay_qos2),
				format_number(messages_qos2),
				row[8],
				row[9],
				cpu_to_ram_limit(row[9]),
				"0",
				"1"
			]
		)
	return processed_rows

def main() -> None:
	args = parse_args()
	rng = random.Random(args.seed)

	dims = [getattr(args, name) for name in DIMENSION_NAMES]
	rows = categorical_lhs(args.samples, dims, rng)
	rows = postprocess_throughput(rows)

	if args.output is not None:
		write_csv(args.output, rows)
		print(f"Wrote {len(rows)} samples to {args.output}")
	else:
		print_rows(rows)


if __name__ == "__main__":
	main()
