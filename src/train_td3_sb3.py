"""Train TD3 with Stable-Baselines3 on one RCMPSP instance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from gantt import plot_gantt
from rcmpsp import generate_schedule, parse_rcmp, priority_fifo, priority_shortest_duration, random_priorities
from sb3_env import make_sb3_env


def evaluate(model, env, episodes: int = 1) -> tuple[float, object]:
    makespans = []
    best_schedule = None
    for _ in range(episodes):
        observation, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
        makespans.append(float(info["makespan"]))
        best_schedule = info["schedule"]
    return float(np.mean(makespans)), best_schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path, nargs="?", default=Path("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/td3_sb3"))
    parser.add_argument("--gantt", type=Path, default=Path("outputs/mp_j30_a2_nr1_td3_sb3_gantt.png"))
    args = parser.parse_args()

    set_random_seed(args.seed)
    env = make_sb3_env(args.instance)
    eval_env = make_sb3_env(args.instance)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(np.zeros(n_actions), 0.1 * np.ones(n_actions))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=100,
        batch_size=100,
        tau=5e-3,
        gamma=0.99,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        action_noise=action_noise,
        policy_kwargs={"net_arch": [256, 256, 128]},
        tensorboard_log=str(args.output_dir / "tensorboard"),
        seed=args.seed,
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=str(args.output_dir), log_path=str(args.output_dir),
        eval_freq=max(64, env.unwrapped.activity_count * 10), deterministic=True, render=False,
    )
    model.learn(total_timesteps=args.episodes * env.unwrapped.activity_count, callback=eval_callback)
    model.save(str(args.output_dir / "final_model"))

    mean_makespan, schedule = evaluate(model, eval_env)
    instance = parse_rcmp(args.instance)
    baselines = {
        "fifo": generate_schedule(instance, priority_fifo).makespan,
        "shortest": generate_schedule(instance, priority_shortest_duration).makespan,
        "random": generate_schedule(instance, random_priorities(instance, args.seed)).makespan,
    }
    print(f"SB3 TD3 makespan={mean_makespan:.0f}; baselines={baselines}")
    plot_gantt(instance, schedule, args.gantt, title=f"SB3 TD3 schedule (Cmax={mean_makespan:.0f})")


if __name__ == "__main__":
    main()
