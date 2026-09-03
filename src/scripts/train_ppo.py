"""Train the graph-aware PPO policy on the configured RCMPSP training split."""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from src.core.exact import solve_exact
from src.core.rcmpsp import parse_rcmp
from src.environments.multi_instance import make_splits, write_splits
from src.environments.observation import observation_size
from src.training.callbacks import RCMPSPMetricsCallback
from src.training.environments import make_multi_env, make_vector_env
from src.training.ppo import baseline_makespans, create_ppo, evaluate_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=6_400_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument(
        "--graph-layers",
        type=int,
        default=0,
        help="Precedence message-passing layers; 0 uses the faster structured MLP.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppo_gnn"))
    parser.add_argument("--splits", type=Path, default=Path("splits.json"))
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and evaluate output-dir/final_model.zip.",
    )
    parser.add_argument(
        "--exact-time-limit",
        type=float,
        default=60.0,
        help="CP-SAT seconds per instance during final evaluation; use 0 to skip.",
    )
    parser.add_argument("--exact-workers", type=int, default=1, help="CP-SAT workers; 1 is reproducible.")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA {requested!r} unavailable; falling back to CPU")
        return "cpu"
    if requested in {"cpu", "cuda"} or requested.startswith("cuda:"):
        return requested
    raise ValueError("--device must be one of: auto, cpu, cuda, cuda:N")


def evaluate_and_save_results(model: PPO, splits: dict[str, list[str]], seed: int,
                              reference_env, output_dir: Path, exact_time_limit: float,
                              exact_workers: int) -> Path:
    """Compare one deterministic PPO rollout per instance with schedule baselines."""
    rows = []
    for split, paths in splits.items():
        ppo_by_name = dict(evaluate_paths(model, paths, seed, reference_env))
        for path in paths:
            instance = parse_rcmp(path)
            baselines = baseline_makespans(instance, seed)
            ppo_makespan = ppo_by_name[Path(path).name]
            row = {
                "split": split,
                "instance": Path(path).name,
                "ppo_makespan": ppo_makespan,
            }
            for name, makespan in baselines.items():
                row[f"{name}_makespan"] = makespan
                row[f"ppo_minus_{name}"] = ppo_makespan - makespan
                row[f"ppo_vs_{name}_pct"] = 100.0 * (ppo_makespan - makespan) / makespan
            if exact_time_limit:
                exact = solve_exact(
                    instance, time_limit=exact_time_limit, workers=exact_workers
                )
                exact_makespan = exact.schedule.makespan
                row.update(
                    {
                        "cp_sat_makespan": exact_makespan,
                        "cp_sat_status": exact.status,
                        "cp_sat_best_bound": exact.best_bound,
                        "cp_sat_wall_time_s": exact.wall_time,
                        "ppo_minus_cp_sat": ppo_makespan - exact_makespan,
                        "ppo_vs_cp_sat_pct": 100.0 * (ppo_makespan - exact_makespan) / exact_makespan,
                    }
                )
            rows.append(row)
            print(
                f"{split} {row['instance']}: ppo={ppo_makespan:.0f} "
                f"fifo={baselines['fifo']} ({row['ppo_vs_fifo_pct']:+.2f}%) "
                f"shortest={baselines['shortest']} ({row['ppo_vs_shortest_pct']:+.2f}%) "
                f"random={baselines['random']} ({row['ppo_vs_random_pct']:+.2f}%)"
                + (
                    f" cp-sat={row['cp_sat_makespan']:.0f} ({row['cp_sat_status']}; "
                    f"{row['ppo_vs_cp_sat_pct']:+.2f}%)"
                    if exact_time_limit
                    else ""
                )
            )

    result_path = output_dir / "per_instance_results.csv"
    if not rows:
        print("no instances configured; skipping per-instance evaluation")
        return result_path
    with result_path.open("w", newline="", encoding="ascii") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-instance results: {result_path}")
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        if split_rows:
            summary = (
                f"{split}: ppo_mean={np.mean([row['ppo_makespan'] for row in split_rows]):.2f} "
                f"fifo_mean={np.mean([row['fifo_makespan'] for row in split_rows]):.2f} "
                f"shortest_mean={np.mean([row['shortest_makespan'] for row in split_rows]):.2f} "
                f"random_mean={np.mean([row['random_makespan'] for row in split_rows]):.2f}"
            )
            if exact_time_limit:
                summary += (
                    f" cp_sat_incumbent_mean="
                    f"{np.mean([row['cp_sat_makespan'] for row in split_rows]):.2f} "
                    f"cp_sat_bound_mean="
                    f"{np.mean([row['cp_sat_best_bound'] for row in split_rows]):.2f}"
                )
            print(summary)
    return result_path


def main() -> None:
    args = parse_args()
    if (
        args.total_timesteps < 1
        or args.n_envs < 1
        or args.n_steps < 1
        or args.batch_size < 1
        or args.n_epochs < 1
        or args.graph_layers < 0
    ):
        raise ValueError(
            "timesteps, n-envs, n-steps, and batch-size must be positive; "
            "n-epochs must be positive and graph-layers non-negative"
        )
    if args.exact_time_limit < 0:
        raise ValueError("--exact-time-limit must be non-negative")
    if args.exact_workers < 1:
        raise ValueError("--exact-workers must be positive")
    args.device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    set_random_seed(args.seed)

    splits = make_splits()
    write_splits(args.splits)
    train_paths = splits["train"]
    instances = [parse_rcmp(path) for path in train_paths]
    max_activities = max(len(instance.activities) for instance in instances)
    max_resources = max(instance.resource_count for instance in instances)
    max_horizon = max(sum(activity.duration for activity in instance.activities.values()) for instance in instances)
    reference_env = SimpleNamespace(
        max_activities=max_activities,
        max_resources=max_resources,
        max_horizon=max_horizon,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}; envs={args.n_envs}; obs_dim={observation_size(max_activities, max_resources)}")
    if args.evaluate_only:
        model_path = args.output_dir / "final_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"PPO model not found: {model_path}")
        model = PPO.load(str(model_path), device=args.device)
        evaluate_and_save_results(
            model, splits, args.seed, reference_env, args.output_dir,
            args.exact_time_limit, args.exact_workers,
        )
        return

    env_fns = [partial(make_multi_env, train_paths, args.seed + rank) for rank in range(args.n_envs)]
    env = make_vector_env(env_fns, parallel=args.n_envs > 1)
    model = create_ppo(
        env,
        seed=args.seed,
        device=args.device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        graph_layers=args.graph_layers,
        tensorboard_log=str(args.output_dir / "tensorboard"),
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=RCMPSPMetricsCallback(),
            progress_bar=False,
        )
        model.save(str(args.output_dir / "final_model"))
        evaluate_and_save_results(
            model, splits, args.seed, reference_env, args.output_dir,
            args.exact_time_limit, args.exact_workers,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
