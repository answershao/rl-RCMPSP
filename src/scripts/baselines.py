"""Evaluate scheduling baselines on all two-project, j30 RCMPSP instances."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.core.exact import solve_exact
from src.core.rcmpsp import parse_rcmp
from src.training.ppo import baseline_makespans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=Path("MPSPLIB/RCMP"),
        help="directory containing MPSPLIB .rcmp instances",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/baselines_j30_a2/makespan_summary.csv"),
        help="path for the instance-by-method makespan matrix",
    )
    parser.add_argument("--seed", type=int, default=7, help="seed for the random-priority baseline")
    parser.add_argument(
        "--exact-time-limit",
        type=float,
        default=60.0,
        help="CP-SAT seconds per instance; use 0 to omit the CP-SAT column.",
    )
    parser.add_argument("--exact-workers", type=int, default=1, help="CP-SAT workers; 1 is reproducible.")
    return parser.parse_args()


def find_instances(root: Path) -> list[Path]:
    """Return the complete two-project, 30-activity benchmark set."""
    paths = sorted(root.glob("mp_j30_a2_nr*.rcmp"))
    if len(paths) != 5:
        raise ValueError(f"expected five mp_j30_a2 instances under {root}, found {len(paths)}")
    return paths


def evaluate_instances(
    paths: list[Path], *, seed: int, exact_time_limit: float, exact_workers: int
) -> list[dict[str, int | str]]:
    """Return makespans with one row per instance and one column per method."""
    rows = []
    for path in paths:
        instance = parse_rcmp(path)
        baselines = baseline_makespans(instance, seed)
        row: dict[str, int | str] = {
            "instance": path.name,
            "FIFO": baselines["fifo"],
            "Shortest": baselines["shortest"],
            "Random": baselines["random"],
        }
        if exact_time_limit:
            exact = solve_exact(
                instance, time_limit=exact_time_limit, workers=exact_workers
            )
            row["CP-SAT"] = exact.schedule.makespan
            exact_details = (
                f" CP-SAT={row['CP-SAT']} ({exact.status}; "
                f"bound={exact.best_bound:.0f}; {exact.wall_time:.2f}s)"
            )
        else:
            exact_details = ""
        rows.append(row)
        print(
            f"{row['instance']}: FIFO={row['FIFO']} Shortest={row['Shortest']} "
            f"Random={row['Random']}" + exact_details
        )
    return rows


def write_summary(rows: list[dict[str, int | str]], output_csv: Path, *, include_exact: bool) -> Path:
    """Write the instance-by-method makespan matrix."""
    methods = ["FIFO", "Shortest", "Random"]
    if include_exact:
        methods.append("CP-SAT")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="ascii") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["instance", *methods])
        writer.writeheader()
        writer.writerows(rows)
    print(f"makespan summary: {output_csv}")
    print(
        "mean: " + " ".join(
            f"{method}={np.mean([row[method] for row in rows]):.2f}"
            for method in methods
        )
    )
    return output_csv


def main() -> None:
    args = parse_args()
    if args.exact_time_limit < 0:
        raise ValueError("--exact-time-limit must be non-negative")
    if args.exact_workers < 1:
        raise ValueError("--exact-workers must be positive")
    paths = find_instances(args.instances_root)
    rows = evaluate_instances(
        paths,
        seed=args.seed,
        exact_time_limit=args.exact_time_limit,
        exact_workers=args.exact_workers,
    )
    write_summary(rows, args.output_csv, include_exact=bool(args.exact_time_limit))


if __name__ == "__main__":
    main()
