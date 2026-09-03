"""Train the project TD3 policy on one instance or a fixed j30 split."""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from src.core.rcmpsp import parse_rcmp
from src.environments.multi_instance import make_splits, write_splits
from src.environments.observation import observation_size
from src.training.td3 import baseline_makespans, create_td3, evaluate_paths, run_policy_episode
from src.training.callbacks import RCMPSPMetricsCallback
from src.training.environments import make_multi_env, make_single_env, make_vector_env
from src.visualization.gantt import plot_gantt


DEFAULT_INSTANCE = Path("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")


def resolve_device(requested: str) -> str:
    """Resolve and validate the learner device before constructing SB3."""
    requested = requested.strip().lower()
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            # Keep training usable on CPU-only hosts even when a CUDA device was
            # requested by a shared/default launch configuration.
            print(
                f"CUDA device {requested!r} is unavailable; falling back to CPU. "
                "Install a CUDA-enabled PyTorch build and check the scheduler allocation "
                "if GPU execution is required."
            )
            device = "cpu"
        else:
            device = requested
        if device.startswith("cuda:"):
            try:
                index = torch.device(device).index
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"invalid CUDA device: {device}") from exc
            if index is None or index < 0 or index >= torch.cuda.device_count():
                raise ValueError(
                    f"{device} is unavailable; visible CUDA devices: {torch.cuda.device_count()}"
                )
    elif requested == "cpu":
        device = requested
    else:
        raise ValueError("--device must be one of: auto, cpu, cuda, cuda:N")

    if device.startswith("cuda"):
        index = torch.cuda.current_device() if device == "cuda" else torch.device(device).index
        index = 0 if index is None else index
        print(
            f"device={device}; GPU={torch.cuda.get_device_name(index)}; "
            f"CUDA runtime={torch.version.cuda}"
        )
    else:
        print(f"device={device}; CUDA available={torch.cuda.is_available()}")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=6_400_000,
        help="Total transitions collected across all vectorized environments.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="Parallel environments for --mode multi; tune to available CPU cores.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Environment steps between TD3 updates; 4 improves CPU FPS with fewer updates.",
    )
    parser.add_argument("--device", default="auto", help="PyTorch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads for the TD3 learner; environment workers remain separate processes.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--splits", type=Path, default=Path("splits.json"))
    parser.add_argument("--gantt", type=Path)
    parser.add_argument("--eval-freq", type=int, default=25_000, help="Evaluation interval in transitions.")
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=10, help="SB3 training log interval in episodes.")
    return parser.parse_args()


def best_checkpoint_or_final(model, output_dir: Path, device: str):
    """Load the validation-selected checkpoint, falling back to the final model."""
    best_model = output_dir / "best_model.zip"
    if not best_model.exists():
        print("best_model.zip was not created; evaluating final_model instead")
        return model
    print(f"evaluating validation-selected checkpoint: {best_model}")
    return model.__class__.load(str(best_model), device=device)


def evaluate_and_save_results(model, splits, seed: int, reference_env, output_dir: Path) -> Path:
    """Evaluate every split instance and persist the per-instance comparison."""
    rows = []
    for split, paths in splits.items():
        td3_by_name = dict(evaluate_paths(model, paths, seed, reference_env))
        for path in paths:
            instance = parse_rcmp(path)
            baselines = baseline_makespans(instance, seed)
            td3_makespan = td3_by_name[Path(path).name]
            project_count = len({project for project, _ in instance.activities})
            row = {
                "split": split,
                "instance": Path(path).name,
                "projects": project_count,
                "td3_makespan": td3_makespan,
            }
            for name, makespan in baselines.items():
                row[f"{name}_makespan"] = makespan
                row[f"td3_minus_{name}"] = td3_makespan - makespan
                row[f"td3_vs_{name}_pct"] = 100.0 * (td3_makespan - makespan) / makespan
            rows.append(row)
            print(
                f"{split} {row['instance']}: td3={td3_makespan:.0f} "
                f"fifo={baselines['fifo']} ({row['td3_vs_fifo_pct']:+.2f}%) "
                f"random={baselines['random']} ({row['td3_vs_random_pct']:+.2f}%)"
            )

    if not rows:
        print("no validation/test instances configured; skipping per-instance evaluation")
        return output_dir / "per_instance_results.csv"
    result_path = output_dir / "per_instance_results.csv"
    with result_path.open("w", newline="", encoding="ascii") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-instance results: {result_path}")

    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        print(
            f"{split}: td3_mean={np.mean([row['td3_makespan'] for row in split_rows]):.2f} "
            f"fifo_mean={np.mean([row['fifo_makespan'] for row in split_rows]):.2f} "
            f"random_mean={np.mean([row['random_makespan'] for row in split_rows]):.2f}"
        )
    return result_path


def train_single(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or Path("outputs/td3_single")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_single_env(args.instance)
    eval_env = make_single_env(args.instance)
    model = create_td3(
        env,
        seed=args.seed,
        buffer_size=args.buffer_size or 200_000,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        device=args.device,
        tensorboard_log=str(output_dir / "tensorboard"),
    )
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList([RCMPSPMetricsCallback(), callback])
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, log_interval=args.log_interval)
    model.save(str(output_dir / "final_model"))

    evaluation_model = best_checkpoint_or_final(model, output_dir, args.device)
    info = run_policy_episode(evaluation_model, eval_env, seed=args.seed)
    instance = parse_rcmp(args.instance)
    print(f"TD3 makespan={info['makespan']}; baselines={baseline_makespans(instance, args.seed)}")
    gantt = args.gantt or output_dir / "schedule.png"
    output = plot_gantt(instance, info["schedule"], gantt, title=f"TD3 schedule (Cmax={info['makespan']})")
    print(f"gantt: {output}")


def train_multi(args: argparse.Namespace) -> None:
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    output_dir = args.output_dir or Path("outputs/td3_multi")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = make_splits()
    write_splits(args.splits)
    # Only metadata is needed here; constructing another full environment would
    # allocate schedule/resource buffers without participating in training.
    train_instances = [parse_rcmp(path) for path in splits["train"]]
    reference_env = SimpleNamespace(
        max_activities=max(len(instance.activities) for instance in train_instances),
        max_resources=max(instance.resource_count for instance in train_instances),
        max_horizon=max(
            sum(activity.duration for activity in instance.activities.values())
            for instance in train_instances
        ),
    )
    env_fns = [partial(make_multi_env, splits["train"], args.seed + rank) for rank in range(args.n_envs)]
    env = make_vector_env(env_fns, parallel=args.n_envs > 1)
    buffer_size = args.buffer_size or 200_000
    observation_dim = observation_size(reference_env.max_activities, reference_env.max_resources)
    action_dim = reference_env.max_activities
    bytes_per_transition = (2 * observation_dim + action_dim + 3) * np.dtype(np.float32).itemsize
    print(f"state_dim={env.observation_space.shape[0]} action_dim={env.action_space.shape[0]} buffer_size={buffer_size} estimated_replay={buffer_size * bytes_per_transition / 1024**3:.2f} GiB")
    model = create_td3(
        env,
        seed=args.seed,
        buffer_size=buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        device=args.device,
        tensorboard_log=str(output_dir / "tensorboard"),
    )
    print(f"parallel_envs={args.n_envs} total_timesteps={args.total_timesteps}")
    eval_env = None
    try:
        callbacks = [RCMPSPMetricsCallback()]
        if splits["validation"]:
            eval_env_fn = partial(
                make_multi_env,
                splits["validation"],
                args.seed + 10_000,
                max_activities=reference_env.max_activities,
                max_resources=reference_env.max_resources,
                max_horizon=reference_env.max_horizon,
            )
            # One subprocess is sufficient because evaluation is deterministic.
            eval_env = make_vector_env([eval_env_fn], parallel=args.n_envs > 1)
            callbacks.append(EvalCallback(
                eval_env,
                best_model_save_path=str(output_dir),
                log_path=str(output_dir),
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            ))
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks),
            log_interval=args.log_interval,
            progress_bar=False,
        )
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()
    model.save(str(output_dir / "final_model"))

    evaluation_model = best_checkpoint_or_final(model, output_dir, args.device)
    evaluate_and_save_results(evaluation_model, splits, args.seed, reference_env, output_dir)


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    if args.total_timesteps < 1:
        raise ValueError("--total-timesteps must be at least 1")
    if args.buffer_size is not None and args.buffer_size < 1:
        raise ValueError("--buffer-size must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.train_freq < 1:
        raise ValueError("--train-freq must be at least 1")
    if args.eval_freq < 1:
        raise ValueError("--eval-freq must be at least 1")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be at least 1")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be at least 1")
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError("--torch-threads must be at least 1")
        torch.set_num_threads(args.torch_threads)
    set_random_seed(args.seed)
    if args.mode == "single":
        train_single(args)
    else:
        train_multi(args)


if __name__ == "__main__":
    main()
