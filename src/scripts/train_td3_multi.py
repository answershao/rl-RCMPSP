"""Train SB3 TD3 on a fixed multi-instance j30 split and evaluate it."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from src.core.rcmpsp import generate_schedule, parse_rcmp, priority_fifo, priority_shortest_duration, random_priorities
from src.environments.multi_instance import MultiInstanceRCMPSPEnv, make_splits, write_splits


def evaluate(model, paths, seed, reference):
    results = []
    for offset, path in enumerate(paths):
        env = MultiInstanceRCMPSPEnv([path], seed=seed + offset,
                                     max_activities=reference.max_activities,
                                     max_resources=reference.max_resources,
                                     max_horizon=reference.max_horizon)
        obs, _ = env.reset(seed=seed + offset)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
        results.append((Path(path).name, info["makespan"]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--buffer-size", type=int, default=2_000,
        help="replay capacity; 2,000 transitions use about 0.64 GiB for this padded j30 state",
    )
    parser.add_argument("--splits", type=Path, default=Path("splits.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/td3_multi"))
    args = parser.parse_args()
    set_random_seed(args.seed)
    splits = make_splits()
    write_splits(args.splits)
    env = MultiInstanceRCMPSPEnv(splits["train"], seed=args.seed)
    n_actions = env.action_space.shape[0]
    bytes_per_transition = (2 * env.observation_space.shape[0] + n_actions + 3) * np.dtype(np.float32).itemsize
    print(
        f"state_dim={env.observation_space.shape[0]} action_dim={n_actions} "
        f"buffer_size={args.buffer_size} estimated_replay={args.buffer_size * bytes_per_transition / 1024**3:.2f} GiB"
    )
    model = TD3("MlpPolicy", env, learning_rate=3e-4, buffer_size=args.buffer_size, learning_starts=100, batch_size=100, tau=5e-3, gamma=0.99, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5, action_noise=NormalActionNoise(np.zeros(n_actions), 0.1 * np.ones(n_actions)), policy_kwargs={"net_arch": [256, 256, 128]}, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.episodes * env.max_activities, progress_bar=False)
    args.output.parent.mkdir(parents=True, exist_ok=True); model.save(str(args.output))
    for label, paths in (("train", splits["train"]), ("validation", splits["validation"]), ("test", splits["test"])):
        td3 = evaluate(model, paths, args.seed, env)
        baseline = []
        for path in paths:
            instance = parse_rcmp(path)
            baseline.append((Path(path).name, generate_schedule(instance, priority_fifo).makespan, generate_schedule(instance, priority_shortest_duration).makespan, generate_schedule(instance, random_priorities(instance, args.seed)).makespan))
        baseline_values = np.asarray([row[1:] for row in baseline], dtype=np.float64)
        print(label, "td3_mean=", np.mean([value for _, value in td3]), "baseline_mean=", np.mean(baseline_values, axis=0))


if __name__ == "__main__":
    main()
