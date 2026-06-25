#!/usr/bin/env python3

"""Combine split subscriber e2e/conn files into single files per run tag.

Expected split filename format:
  e2e_b0_<timestamp>__SUB_0_qos<0|1|2>_client-<n>.txt
  conn_b0_<timestamp>__SUB_0_qos<0|1|2>_client-<n>.txt

Produces:
    e2e_b0_<timestamp>_<i>_SUB_0.txt
    conn_b0_<timestamp>_<i>_SUB_0.txt
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


FILE_PATTERN = re.compile(
    r"^(?P<kind>e2e|conn)_"
    r"b(?P<broker>\d+)_"
    r"(?P<tag>.+?)_+SUB_(?P<sub>\d+)_"
    r"qos(?P<qos>[0-2])_"
    r"client-(?P<client>\d+)\.txt$"
)


@dataclass(frozen=True)
class ParsedName:
    kind: str
    broker: int
    tag: str
    sub: int
    qos: int
    client: int
    path: Path


def extract_iteration(tag: str, timestamp: str) -> int | None:
    """Extract iterator from tags like '<timestamp>_<i>_...'."""
    match = re.match(rf"^{re.escape(timestamp)}_(\d+)(?:_|$)", tag)
    if not match:
        return None
    return int(match.group(1))


def parse_filename(path: Path) -> ParsedName | None:
    match = FILE_PATTERN.match(path.name)
    if not match:
        return None
    return ParsedName(
        kind=match.group("kind"),
        broker=int(match.group("broker")),
        tag=match.group("tag"),
        sub=int(match.group("sub")),
        qos=int(match.group("qos")),
        client=int(match.group("client")),
        path=path,
    )


def collect_records(input_dir: Path) -> list[ParsedName]:
    records: list[ParsedName] = []
    for path in input_dir.glob("*.txt"):
        parsed = parse_filename(path)
        if parsed is not None:
            records.append(parsed)
    return records


def merge_group(output_path: Path, files: list[ParsedName]) -> tuple[int, int]:
    """Merge records from files into output_path.

    Returns (file_count, data_line_count).
    """
    sorted_files = sorted(files, key=lambda r: (r.qos, r.client, r.path.name))

    header: str | None = None
    data_lines: list[str] = []

    for record in sorted_files:
        lines = record.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue

        file_header = lines[0].strip()
        if not file_header:
            continue

        if header is None:
            header = file_header

        for line in lines[1:]:
            stripped = line.strip()
            if stripped:
                data_lines.append(stripped)

    if header is None:
        return (len(sorted_files), 0)

    output_lines = [header] + data_lines
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return (len(sorted_files), len(data_lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine split subscriber e2e/conn files into single files."
    )

    BASE_DIR = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BASE_DIR / "inputs/result_data/experiments",
        help=f"Directory containing split files (default: {BASE_DIR / 'inputs/result_data/experiments'}).",
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help=(
            "Base timestamp used to select files, e.g. 20260512145503. "
            "Expected file tags start with '<timestamp>_<i>...'."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_dir: Path = args.input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}")
        return 2

    records = collect_records(input_dir)
    if not records:
        print(f"No matching split files found in: {input_dir}")
        return 1

    timestamp_escaped = re.escape(args.timestamp)
    # Select only split files that begin with the provided timestamp prefix.
    name_filter = re.compile(
        rf"^(e2e|conn)_b\d+_{timestamp_escaped}.*_+SUB_\d+_qos[0-2]_client-\d+\.txt$"
    )

    filtered = [
        r
        for r in records
        if name_filter.match(r.path.name)
    ]

    if not filtered:
        print("No split files matched the selected filters.")
        return 1

    # Group by exact parsed tag so combined output keeps parameter suffixes.
    groups: dict[tuple[str, int, str, int], list[ParsedName]] = {}
    skipped_without_iteration = 0
    for record in filtered:
        iteration = extract_iteration(record.tag, args.timestamp)
        if iteration is None:
            skipped_without_iteration += 1
            continue
        key = (record.kind, record.broker, record.tag, record.sub)
        groups.setdefault(key, []).append(record)

    if not groups:
        print(
            "No split files contained an iteration suffix '<timestamp>_<i>' "
            "for the selected timestamp."
        )
        return 1

    for (kind, broker, tag, sub), files in sorted(groups.items()):
        output_name = f"{kind}_b{broker}_{tag}_SUB_{sub}.txt"
        output_path = input_dir / output_name
        file_count, data_count = merge_group(output_path, files)
        print(
            f"Wrote {output_name} from {file_count} files with {data_count} data lines."
        )

    if skipped_without_iteration:
        print(
            f"Skipped {skipped_without_iteration} files without '<timestamp>_<i>' in tag."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
