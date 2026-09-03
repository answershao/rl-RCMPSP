"""PPO configuration and evaluation helpers for RCMPSP experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from stable_baselines3 import PPO

from src.core.rcmpsp import (
    Instance,
    generate_schedule,
    priority_fifo,
    priority_shortest_duration,
    random_priorities,
)
from src.environments.observation import MAX_SUCCESSORS
from src.training.features import RCMPSPStructuredExtractor


class PolicyEnvironment(Protocol):
    def reset(self, *, seed: int | None = None): ...

    def step(self, action): ...


def create_ppo(
    env,
    *,
    seed: int,
    device: str = "auto",
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 4,
    graph_layers: int = 0,
    tensorboard_log: str | None = None,
) -> PPO:
    """Create a PPO learner with precedence-message-passing features.

    Actions are continuous activity-priority vectors, decoded into feasible
    schedules by the environment.
    """
    if n_steps < 1 or batch_size < 1 or n_epochs < 1 or graph_layers < 0:
        raise ValueError("n_steps, batch_size, n_epochs, and graph_layers must be non-negative")
    action_count = env.action_space.shape[0]
    base_env = env.envs[0] if hasattr(env, "envs") else env
    max_activities = getattr(base_env, "max_activities", action_count)
    max_resources = getattr(base_env, "max_resources", None)
    if max_resources is None:
        obs_dim = int(env.observation_space.shape[0])
        numerator = obs_dim - (6 + MAX_SUCCESSORS) * max_activities - 1
        max_resources = numerator // (max_activities + 1)
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "features_extractor_class": RCMPSPStructuredExtractor,
            "features_extractor_kwargs": {
                "max_activities": max_activities,
                "max_resources": max_resources,
                "max_successors": getattr(base_env, "max_successors", MAX_SUCCESSORS),
                # Scatter/gather message passing dominates CPU PPO updates on
                # these small instances. Keep it configurable for ablations.
                "graph_layers": graph_layers,
                "embedding_dim": 8,
                "hidden_dim": 32,
                "global_dim": 16,
            },
        },
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )


def run_policy_episode(model: PPO, env: PolicyEnvironment, *, seed: int | None = None) -> dict:
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


def evaluate_paths(
    model: PPO, paths: list[str], seed: int, reference_env
) -> list[tuple[str, float]]:
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
        info = run_policy_episode(model, env, seed=seed + offset)
        results.append((Path(path).name, float(info["makespan"])))
    return results
