"""Train the project TD3 policy on one instance or a fixed j30 split."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.core.rcmpsp import parse_rcmp
from src.environments.multi_instance import MultiInstanceRCMPSPEnv, make_splits, write_splits
from src.environments.sb3_env import make_sb3_env
from src.training.td3 import baseline_makespans, create_td3, evaluate_paths, run_policy_episode
from src.visualization.gantt import plot_gantt


DEFAULT_INSTANCE = Path("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Parallel environments for --mode multi; use 1 for single-process debugging.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--splits", type=Path, default=Path("splits.json"))
    parser.add_argument("--gantt", type=Path)
    return parser.parse_args()


def make_multi_env(instance_paths: list[str], seed: int) -> MultiInstanceRCMPSPEnv:
    """Create one independently seeded environment for a vectorized worker."""
    return MultiInstanceRCMPSPEnv(instance_paths, seed=seed)


def train_single(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or Path("outputs/td3_single")
    env = make_sb3_env(args.instance)
    eval_env = make_sb3_env(args.instance)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_td3(env, seed=args.seed, buffer_size=args.buffer_size or 1_000_000)
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir),
        eval_freq=max(64, env.unwrapped.activity_count * 10),
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=args.episodes * env.unwrapped.activity_count, callback=callback)
    model.save(str(output_dir / "final_model"))

    info = run_policy_episode(model, eval_env, seed=args.seed)
    instance = parse_rcmp(args.instance)
    print(f"TD3 makespan={info['makespan']}; baselines={baseline_makespans(instance, args.seed)}")
    gantt = args.gantt or output_dir / "schedule.png"
    output = plot_gantt(instance, info["schedule"], gantt, title=f"TD3 schedule (Cmax={info['makespan']})")
    print(f"gantt: {output}")


def train_multi(args: argparse.Namespace) -> None:
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    output_dir = args.output_dir or Path("outputs/td3_multi")
    splits = make_splits()
    write_splits(args.splits)
    reference_env = make_multi_env(splits["train"], args.seed)
    env_fns = [partial(make_multi_env, splits["train"], args.seed + rank) for rank in range(args.n_envs)]
    # spawn avoids unsafe forks after PyTorch has initialized its worker threads.
    env = DummyVecEnv(env_fns) if args.n_envs == 1 else SubprocVecEnv(env_fns, start_method="spawn")
    buffer_size = args.buffer_size or 2_000
    bytes_per_transition = (2 * reference_env.observation_space.shape[0] + reference_env.action_space.shape[0] + 3) * np.dtype(np.float32).itemsize
    print(f"state_dim={env.observation_space.shape[0]} action_dim={env.action_space.shape[0]} buffer_size={buffer_size} estimated_replay={buffer_size * bytes_per_transition / 1024**3:.2f} GiB")
    model = create_td3(env, seed=args.seed, buffer_size=buffer_size)
    total_timesteps = args.episodes * reference_env.max_activities
    print(f"parallel_envs={args.n_envs} total_timesteps={total_timesteps}")
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
    finally:
        env.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / "final_model"))

    for label, paths in splits.items():
        td3 = evaluate_paths(model, paths, args.seed, reference_env)
        baselines = [baseline_makespans(parse_rcmp(path), args.seed) for path in paths]
        baseline_mean = {name: float(np.mean([result[name] for result in baselines])) for name in ("fifo", "shortest", "random")}
        print(f"{label}: td3_mean={np.mean([makespan for _, makespan in td3]):.2f} baseline_mean={baseline_mean}")


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    if args.mode == "single":
        train_single(args)
    else:
        train_multi(args)


if __name__ == "__main__":
    main()
