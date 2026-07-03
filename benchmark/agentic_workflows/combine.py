from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import DEFAULT_RESULTS_ROOT


def find_summary_files(results_root: Path) -> list[Path]:
    return sorted(Path(results_root).glob("**/agentic_workflow_summary.csv"))


def combine_summary_files(summary_files: list[Path], output_csv: Path) -> int:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in summary_files:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None and reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                row["summary_csv"] = str(path)
                rows.append(row)
    if not rows:
        return 0
    output_fields = list(fieldnames or rows[0].keys())
    if "summary_csv" not in output_fields:
        output_fields.append("summary_csv")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine agentic workflow online summary CSVs."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "agentic_workflow_online_summary.csv",
    )
    args = parser.parse_args(argv)

    count = combine_summary_files(find_summary_files(args.results_root), args.output_csv)
    print(args.output_csv if count else "no summary files found")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
