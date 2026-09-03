"""Fixed train/validation/test splits and a padded multi-instance environment."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.rcmpsp import parse_rcmp
from src.environments.rcmpsp_env import RCMPSPEnv
from src.environments.observation import (
    MAX_SUCCESSORS,
    flatten_observation,
    observation_size,
)

DEFAULT_INSTANCES_ROOT = Path("data/MPLIB2_train_10_500_5")


def make_splits(root: str | Path = DEFAULT_INSTANCES_ROOT) -> dict[str, list[str]]:
    """Return every RCMP file in a prepared training-data directory."""
    files = sorted(Path(root).glob("*.rcmp"))
    if not files:
        raise ValueError(f"no .rcmp instances found under {root}")
    return {"train": [str(path) for path in files], "validation": [], "test": []}


def write_splits(
    path: str | Path = "splits.json",
    root: str | Path = DEFAULT_INSTANCES_ROOT,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_splits(root), indent=2) + "\n")
    return path


def partition_instance_catalog(
    paths: Sequence[str], parts: int
) -> list[tuple[list[str], list[int]]]:
    """Split a training catalog into per-worker subsets with global indices."""
    if parts < 1:
        raise ValueError("parts must be positive")
    if not paths:
        raise ValueError("paths must not be empty")
    if parts > len(paths):
        raise ValueError("parts cannot exceed the number of paths")
    return [
        (
            [str(paths[index]) for index in range(part, len(paths), parts)],
            list(range(part, len(paths), parts)),
        )
        for part in range(parts)
    ]


class MultiInstanceRCMPSPEnv(gym.Env[np.ndarray, int]):
    """Sample one instance per episode with fixed padded observation and action spaces."""

    def __init__(self, instances: list[str | Path], seed: int | None = None,
                 max_activities: int | None = None, max_resources: int | None = None,
                 instance_indices: list[int] | None = None,
                 catalog_size: int | None = None):
        super().__init__()
        if not instances:
            raise ValueError("instances must not be empty")
        self.instance_paths = [Path(path) for path in instances]
        self.instances = [parse_rcmp(path) for path in self.instance_paths]
        self.instance_indices = (
            list(range(len(self.instances))) if instance_indices is None else instance_indices
        )
        self.catalog_size = len(self.instances) if catalog_size is None else catalog_size
        if len(self.instance_indices) != len(self.instances):
            raise ValueError("instance_indices must match instances")
        if any(not 0 <= index < self.catalog_size for index in self.instance_indices):
            raise ValueError("instance_indices must refer to the static graph catalog")
        self.max_activities = max([max_activities or 0] + [len(instance.activities) for instance in self.instances])
        self.max_resources = max([max_resources or 0] + [instance.resource_count for instance in self.instances])
        self.max_successors = MAX_SUCCESSORS
        self.action_space = spaces.Discrete(self.max_activities)
        feature_size = observation_size(self.max_activities, self.max_resources)
        self.observation_space = spaces.Box(0.0, 1.0, (feature_size,), dtype=np.float32)
        # Reuse environment objects across episodes; reset only clears mutable
        # scheduling state and avoids repeated allocation of large buffers.
        self._envs = [RCMPSPEnv(instance) for instance in self.instances]
        self._env: RCMPSPEnv | None = None
        self._flat_buffers: dict[int, np.ndarray] = {}
        self._capacity_scales = {
            index: np.maximum(np.asarray(instance.capacities, dtype=np.float32), 1.0)
            for index, instance in enumerate(self.instances)
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = int(self.np_random.integers(len(self.instances)))
        self._env = self._envs[index]
        observation, info = self._env.reset(seed=seed)
        self._active_index = index
        return self._encode(observation), {**info, "instance": self.instance_paths[index].name}

    def step(self, action):
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.max_activities})")
        observation, reward, terminated, truncated, info = self._env.step(int(action))
        return self._encode(observation), reward, terminated, truncated, info

    @property
    def active_env(self) -> RCMPSPEnv:
        if self._env is None:
            raise RuntimeError("environment has not been reset")
        return self._env

    def _encode(self, observation):
        env = self.active_env
        buffer = self._flat_buffers.get(self._active_index)
        if buffer is None:
            buffer = np.zeros(
                observation_size(self.max_activities, self.max_resources), dtype=np.float32
            )
            self._flat_buffers[self._active_index] = buffer
        return flatten_observation(
            observation,
            env.instance.capacities,
            env.horizon,
            instance_index=self.instance_indices[self._active_index],
            catalog_size=self.catalog_size,
            max_activities=self.max_activities,
            max_resources=self.max_resources,
            capacity_scale=self._capacity_scales[self._active_index],
            out=buffer,
        )
