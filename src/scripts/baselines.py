"""Run deterministic RCMPSP scheduling baselines on an MPSPLIB instance."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.exact import solve_exact
from src.core.rcmpsp import generate_schedule, parse_rcmp, priority_fifo, priority_shortest_duration, random_priorities
from src.visualization.aon import plot_aon
from src.visualization.gantt import plot_gantt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate valid integer-resource RCMPSP schedules.")
    parser.add_argument(
        "instance",
        nargs="?",
        type=Path,
        default=Path("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp"),
        help="path to an MPSPLIB .rcmp instance",
    )
    parser.add_argument("--seed", type=int, default=7, help="seed for the random-priority baseline")
    parser.add_argument(
        "--exact-time-limit",
        type=float,
        default=60.0,
        help="CP-SAT seconds per instance; use 0 to skip exact solving.",
    )
    parser.add_argument("--exact-workers", type=int, default=1, help="CP-SAT workers; 1 is reproducible.")
    parser.add_argument(
        "--baseline",
        choices=("fifo", "shortest", "random"),
        default="random",
        help="baseline used for the optional Gantt chart",
    )
    parser.add_argument(
        "--gantt",
        type=Path,
        metavar="PNG",
        help="save a Gantt chart to this PNG path",
    )
    parser.add_argument(
        "--aon",
        type=Path,
        metavar="PNG",
        help="save an Activity-on-Node precedence network to this PNG path",
    )
    args = parser.parse_args()

    instance = parse_rcmp(args.instance)
    baselines = {
        "FIFO": priority_fifo,
        "shortest-duration": priority_shortest_duration,
        f"random-priority(seed={args.seed})": random_priorities(instance, args.seed),
    }
    print(f"instance: {instance.name}; activities: {len(instance.activities)}; resources: {instance.resource_count}")
    for name, priority in baselines.items():
        schedule = generate_schedule(instance, priority)
        print(f"{name}: makespan={schedule.makespan}")
    if args.exact_time_limit < 0:
        parser.error("--exact-time-limit must be non-negative")
    if args.exact_time_limit:
        exact = solve_exact(
            instance, time_limit=args.exact_time_limit, workers=args.exact_workers
        )
        print(
            f"CP-SAT: makespan={exact.schedule.makespan}; status={exact.status}; "
            f"best_bound={exact.best_bound:.0f}; wall_time={exact.wall_time:.2f}s"
        )

    if args.gantt:
        priority = {
            "fifo": priority_fifo,
            "shortest": priority_shortest_duration,
            "random": random_priorities(instance, args.seed),
        }[args.baseline]
        schedule = generate_schedule(instance, priority)
        output = plot_gantt(
            instance,
            schedule,
            args.gantt,
            title=f"{instance.name} - {args.baseline} priority (Cmax={schedule.makespan})",
        )
        print(f"gantt: {output}")

    if args.aon:
        priority = {
            "fifo": priority_fifo,
            "shortest": priority_shortest_duration,
            "random": random_priorities(instance, args.seed),
        }[args.baseline]
        schedule = generate_schedule(instance, priority)
        output = plot_aon(
            instance,
            args.aon,
            schedule=schedule,
            title=f"{instance.name} - AON network ({args.baseline} schedule)",
        )
        print(f"aon: {output}")


if __name__ == "__main__":
    main()
