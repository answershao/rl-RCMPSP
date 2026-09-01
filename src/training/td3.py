"""Shared Stable-Baselines3 TD3 setup and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from src.core.rcmpsp import (
    Instance,
    generate_schedule,
    priority_fifo,
    priority_shortest_duration,
    random_priorities,
)
from src.training.features import RCMPSPStructuredExtractor


class PolicyEnvironment(Protocol):
    action_space: object

    def reset(self, *, seed: int | None = None): ...

    def step(self, action): ...


def create_td3(env, *, seed: int, buffer_size: int) -> TD3:
    """Create the common TD3 configuration used by all experiments."""
    action_count = env.action_space.shape[0]
    # Multi-instance observations are padded to the largest instance.  Encode
    # activity records with shared weights before the TD3 MLP to avoid a huge
    # dense input layer while preserving one output priority per activity.
    base_env = env.envs[0] if hasattr(env, "envs") else env
    max_activities = getattr(base_env, "max_activities", action_count)
    max_resources = getattr(
        base_env,
        "max_resources",
        getattr(getattr(base_env, "unwrapped", None), "resource_count", None),
    )
    if max_resources is None:
        # Works for SubprocVecEnv, where worker environments are not exposed.
        obs_dim = int(env.observation_space.shape[0])
        numerator = obs_dim - 4 * max_activities - 1
        if numerator <= 0 or numerator % (max_activities + 1):
            raise ValueError("cannot infer RCMPSP resource count from observation shape")
        max_resources = numerator // (max_activities + 1)
    return TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=buffer_size,
        learning_starts=100,
        batch_size=100,
        # The scheduling environment is inexpensive relative to a TD3 update.
        # One update per four collected transitions substantially improves
        # wall-clock throughput while retaining off-policy replay learning.
        train_freq=4,
        gradient_steps=1,
        tau=5e-3,
        gamma=0.99,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        action_noise=NormalActionNoise(np.zeros(action_count), 0.1 * np.ones(action_count)),
        policy_kwargs={
            "net_arch": [64, 64],
            "features_extractor_class": RCMPSPStructuredExtractor,
            "features_extractor_kwargs": {
                "max_activities": max_activities,
                "max_resources": max_resources,
                "embedding_dim": 4,
                "hidden_dim": 16,
                "global_dim": 8,
            },
        },
        seed=seed,
        verbose=1,
    )


def run_policy_episode(model: TD3, env, *, seed: int | None = None) -> dict:
    """Run one deterministic policy episode and return its terminal info."""
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
    return info


def baseline_makespans(instance: Instance, seed: int) -> dict[str, int]:
    """Evaluate the deterministic scheduling baselines for one instance."""
    return {
        "fifo": generate_schedule(instance, priority_fifo).makespan,
        "shortest": generate_schedule(instance, priority_shortest_duration).makespan,
        "random": generate_schedule(instance, random_priorities(instance, seed)).makespan,
    }


def evaluate_paths(model: TD3, paths: list[str], seed: int, reference_env) -> list[tuple[str, float]]:
    """Evaluate a padded multi-instance policy once for each supplied path."""
    from src.environments.multi_instance import MultiInstanceRCMPSPEnv

    results = []
    for offset, path in enumerate(paths):
        env = MultiInstanceRCMPSPEnv(
            [path],
            seed=seed + offset,
            max_activities=reference_env.max_activities,
            max_resources=reference_env.max_resources,
            max_horizon=reference_env.max_horizon,
        )
        results.append((Path(path).name, float(run_policy_episode(model, env, seed=seed + offset)["makespan"])))
    return results
