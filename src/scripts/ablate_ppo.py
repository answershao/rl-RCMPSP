"""Compare GIN depth and PPO update epochs using wall-clock scheduling quality."""

from __future__ import annotations

import argparse
import csv
import gc
from functools import partial
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import torch
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed

from src.core.rcmpsp import parse_rcmp
from src.environments.multi_instance import (
    DEFAULT_INSTANCES_ROOT,
    make_splits,
    partition_instance_catalog,
    write_splits,
)
from src.environments.observation import build_static_graph_cache
from src.scripts.train_ppo import resolve_device
from src.training.ablation import EvaluationPoint, PeriodicMakespanCallback
from src.training.callbacks import RCMPSPMetricsCallback
from src.training.environments import make_multi_env, make_vector_env
from src.training.ppo import create_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=6_400_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=96)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gin-layers", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--ppo-epochs", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--eval-split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=DEFAULT_INSTANCES_ROOT,
        help="directory containing the .rcmp training instances",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppo_ablation"))
    parser.add_argument("--splits", type=Path, default=Path("outputs/ppo_ablation/splits.json"))
    return parser.parse_args()


def write_curve(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    fields = [
        "run",
        "seed",
        "gin_layers",
        "ppo_epochs",
        "timesteps",
        "training_seconds",
        "fps",
        "mean_makespan",
        "min_makespan",
        "max_makespan",
    ]
    with path.open("w", newline="", encoding="ascii") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    runs: list[tuple[dict[str, int | str], list[EvaluationPoint]]],
) -> float:
    common_target = max(min(point.mean_makespan for point in points) for _, points in runs)
    fields = [
        "run",
        "seed",
        "gin_layers",
        "ppo_epochs",
        "final_mean_makespan",
        "best_mean_makespan",
        "final_fps",
        "common_target_makespan",
        "steps_to_target",
        "seconds_to_target",
    ]
    rows = []
    for metadata, points in runs:
        reached = next(
            point for point in points if point.mean_makespan <= common_target
        )
        rows.append(
            {
                **metadata,
                "final_mean_makespan": points[-1].mean_makespan,
                "best_mean_makespan": min(point.mean_makespan for point in points),
                "final_fps": points[-1].fps,
                "common_target_makespan": common_target,
                "steps_to_target": reached.timesteps,
                "seconds_to_target": reached.training_seconds,
            }
        )
    with path.open("w", newline="", encoding="ascii") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return common_target


def main() -> None:
    args = parse_args()
    positive_values = [
        args.total_timesteps,
        args.eval_freq,
        args.n_envs,
        args.n_steps,
        args.batch_size,
        args.torch_threads,
        args.torch_interop_threads,
        *args.gin_layers,
        *args.ppo_epochs,
    ]
    if any(value < 1 for value in positive_values):
        raise ValueError("all numeric training parameters must be positive")
    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size:
        raise ValueError("batch-size must divide n-envs * n-steps")

    args.device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = make_splits(args.instances_root)
    write_splits(args.splits, root=args.instances_root)
    evaluation_paths = splits[args.eval_split]
    if not evaluation_paths:
        raise ValueError(f"{args.eval_split} split is empty")
    train_paths = splits["train"]
    if args.n_envs > len(train_paths):
        raise ValueError(
            f"--n-envs ({args.n_envs}) cannot exceed the number of training "
            f"instances ({len(train_paths)})"
        )
    catalog_paths = list(dict.fromkeys(path for paths in splits.values() for path in paths))
    instances_by_path = {path: parse_rcmp(path) for path in catalog_paths}
    train_instances = [instances_by_path[path] for path in train_paths]
    all_instances = list(instances_by_path.values())
    max_activities = max(len(instance.activities) for instance in all_instances)
    max_resources = max(instance.resource_count for instance in all_instances)
    reference_env = SimpleNamespace(
        max_activities=max_activities,
        max_resources=max_resources,
    )
    training_cache = build_static_graph_cache(
        train_instances,
        max_activities=max_activities,
        max_resources=max_resources,
    )

    curve_rows: list[dict[str, int | float | str]] = []
    completed_runs: list[tuple[dict[str, int | str], list[EvaluationPoint]]] = []
    for gin_layers, ppo_epochs, seed in product(
        args.gin_layers, args.ppo_epochs, args.seeds
    ):
        run_name = f"gin{gin_layers}_epochs{ppo_epochs}_seed{seed}"
        run_dir = args.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"starting {run_name}")
        set_random_seed(seed)
        worker_catalogs = partition_instance_catalog(train_paths, args.n_envs)
        env_fns = [
            partial(
                make_multi_env,
                worker_paths,
                seed + rank,
                max_activities=max_activities,
                max_resources=max_resources,
                instance_indices=worker_indices,
                catalog_size=len(train_paths),
            )
            for rank, (worker_paths, worker_indices) in enumerate(worker_catalogs)
        ]
        env = make_vector_env(env_fns, parallel=args.n_envs > 1)
        model = create_ppo(
            env,
            instances=train_instances,
            seed=seed,
            device=args.device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=ppo_epochs,
            gin_layers=gin_layers,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        evaluator = PeriodicMakespanCallback(
            evaluation_paths=evaluation_paths,
            reference_env=reference_env,
            training_cache=training_cache,
            eval_freq=args.eval_freq,
            seed=seed,
        )
        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=CallbackList([RCMPSPMetricsCallback(), evaluator]),
                progress_bar=False,
            )
            model.save(str(run_dir / "final_model"))
        finally:
            env.close()

        metadata: dict[str, int | str] = {
            "run": run_name,
            "seed": seed,
            "gin_layers": gin_layers,
            "ppo_epochs": ppo_epochs,
        }
        completed_runs.append((metadata, evaluator.points))
        for point in evaluator.points:
            curve_rows.append(
                {
                    **metadata,
                    "timesteps": point.timesteps,
                    "training_seconds": point.training_seconds,
                    "fps": point.fps,
                    "mean_makespan": point.mean_makespan,
                    "min_makespan": point.min_makespan,
                    "max_makespan": point.max_makespan,
                }
            )
        write_curve(args.output_dir / "learning_curve.csv", curve_rows)
        del model, env
        gc.collect()

    target = write_summary(args.output_dir / "summary.csv", completed_runs)
    print(f"common target mean makespan: {target:.3f}")
    print(f"curve: {args.output_dir / 'learning_curve.csv'}")
    print(f"summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
