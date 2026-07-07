#!/usr/bin/env python3
"""Move single broker result files into inputs/result_data.

Source: ../results/single_broker_results/
Target: ./inputs/result_data/

The script moves files recursively and preserves each file's path relative
 to single_broker_results (for example experiments/* remains under
 inputs/result_data/experiments/*).
"""

from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    source_root = (script_dir / ".." / "results" / "single_broker_results").resolve()
    target_root = (script_dir / "inputs" / "result_data").resolve()

    if not source_root.exists():
        print(f"Source directory does not exist: {source_root}. When running locally, this may be expected.")
        return

    target_root.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    for src in source_root.rglob("*"):
        if not src.is_file():
            continue

        rel_path = src.relative_to(source_root)
        dst = target_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src), str(dst))
        moved_count += 1

    # Remove empty directories left behind in source_root.
    for directory in sorted(source_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                # Ignore non-empty directories.
                pass

    print(f"Moved {moved_count} file(s) from {source_root} to {target_root}")


if __name__ == "__main__":
    main()
