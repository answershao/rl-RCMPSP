"""Fixed train/validation/test splits and a padded multi-instance environment."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.rcmpsp import parse_rcmp
from src.environments.rcmpsp_env import RCMPSPEnv
from src.environments.sb3_env import FlattenRCMPSPObservation
from src.rl.td3 import flatten_observation


def make_splits(root: str | Path = "MPSPLIB/RCMP") -> dict[str, list[str]]:
    """Return a deterministic 3/1/1 split within each j30 project scale."""
    files = sorted(Path(root).glob("mp_j30_*.rcmp"))
    groups = {
        scale: [path for path in files if f"mp_j30_{scale}_" in path.name]
        for scale in ("a2", "a5", "a10", "a20")
    }
    if any(len(paths) < 5 for paths in groups.values()):
        raise ValueError("expected five instances for each j30 scale: a2, a5, a10, a20")
    splits = {"train": [], "validation": [], "test": []}
    for paths in groups.values():
        splits["train"].extend(str(path) for path in paths[:3])
        splits["validation"].append(str(paths[3]))
        splits["test"].append(str(paths[4]))
    return splits


def write_splits(path: str | Path = "splits.json") -> Path:
    path = Path(path)
    path.write_text(json.dumps(make_splits(), indent=2) + "\n")
    return path


class MultiInstanceRCMPSPEnv(gym.Env[np.ndarray, np.ndarray]):
    """Sample one instance per episode while exposing one fixed Box space."""

    def __init__(self, instances: list[str | Path], seed: int | None = None,
                 max_activities: int | None = None, max_resources: int | None = None,
                 max_horizon: int | None = None):
        super().__init__()
        if not instances:
            raise ValueError("instances must not be empty")
        self.instance_paths = [Path(path) for path in instances]
        self.instances = [parse_rcmp(path) for path in self.instance_paths]
        self.max_activities = max([max_activities or 0] + [len(instance.activities) for instance in self.instances])
        self.max_resources = max([max_resources or 0] + [instance.resource_count for instance in self.instances])
        self.max_horizon = max([max_horizon or 0] + [sum(a.duration for a in instance.activities.values()) for instance in self.instances])
        self.action_space = spaces.Box(-1.0, 1.0, (self.max_activities,), dtype=np.float32)
        feature_size = self.max_activities * 4 + self.max_activities * self.max_resources + self.max_resources + 1 + self.max_activities
        self.observation_space = spaces.Box(0.0, 1.0, (feature_size,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._env: RCMPSPEnv | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = int(self.np_random.integers(len(self.instances))) if seed is not None else int(self._rng.integers(len(self.instances)))
        self._env = RCMPSPEnv(self.instances[index])
        observation, info = self._env.reset(seed=seed)
        self._active_index = index
        return self._pad(observation), {**info, "instance": self.instance_paths[index].name}

    def step(self, action):
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}")
        n = self._env.activity_count
        observation, reward, terminated, truncated, info = self._env.step(action[:n])
        return self._pad(observation), reward, terminated, truncated, info

    @property
    def active_env(self) -> RCMPSPEnv:
        if self._env is None:
            raise RuntimeError("environment has not been reset")
        return self._env

    def _pad(self, observation):
        env = self.active_env
        n, r = env.activity_count, env.resource_count
        result = np.zeros(self.observation_space.shape, dtype=np.float32)
        encoded = flatten_observation(observation, env.instance.capacities, env.horizon)
        # Rebuild the flattened vector with fixed activity/resource axes.
        offset = 0
        for values in (observation["activity_status"] / 2.0, observation["precedence_satisfied"], observation["durations"] / max(env.horizon, 1), observation["eligible_mask"]):
            result[offset:offset + n] = values; offset += self.max_activities
        demands = observation["resource_demands"].astype(np.float32) / np.maximum(np.asarray(env.instance.capacities), 1)
        result[offset:offset + self.max_activities * self.max_resources].reshape(self.max_activities, self.max_resources)[:n, :r] = demands; offset += self.max_activities * self.max_resources
        result[offset:offset + r] = observation["remaining_capacity"] / np.maximum(np.asarray(env.instance.capacities), 1); offset += self.max_resources
        result[offset] = observation["current_time"][0] / max(env.horizon, 1)
        return result
