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
from src.environments.observation import flatten_observation, observation_size


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
        feature_size = observation_size(self.max_activities, self.max_resources)
        self.observation_space = spaces.Box(0.0, 1.0, (feature_size,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._env: RCMPSPEnv | None = None
        self._flat_buffers: dict[int, np.ndarray] = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = int(self.np_random.integers(len(self.instances))) if seed is not None else int(self._rng.integers(len(self.instances)))
        self._env = RCMPSPEnv(self.instances[index])
        observation, info = self._env.reset(seed=seed)
        self._active_index = index
        return self._encode(observation), {**info, "instance": self.instance_paths[index].name}

    def step(self, action):
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}")
        n = self._env.activity_count
        observation, reward, terminated, truncated, info = self._env.step(action[:n])
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
            n = env.activity_count
            r = env.resource_count
            capacities = np.maximum(np.asarray(env.instance.capacities, dtype=np.float32), 1.0)
            activity_offset = 0
            precedence_offset = self.max_activities
            duration_offset = 2 * self.max_activities
            eligible_offset = 3 * self.max_activities
            demand_offset = 4 * self.max_activities
            buffer[duration_offset : duration_offset + n] = (
                env._durations.astype(np.float32) / max(env.horizon, 1)
            )
            demand_view = buffer[
                demand_offset : demand_offset + self.max_activities * self.max_resources
            ].reshape(self.max_activities, self.max_resources)
            demand_view[:n, :r] = env._demands.astype(np.float32) / capacities
            self._flat_buffers[self._active_index] = buffer

        n = env.activity_count
        r = env.resource_count
        buffer[: self.max_activities] = 0.0
        buffer[:n] = observation["activity_status"] / 2.0
        buffer[self.max_activities : 2 * self.max_activities] = 0.0
        buffer[self.max_activities : self.max_activities + n] = observation["precedence_satisfied"]
        eligible_offset = 3 * self.max_activities
        buffer[eligible_offset : eligible_offset + self.max_activities] = 0.0
        buffer[eligible_offset : eligible_offset + n] = observation["eligible_mask"]
        remaining_offset = 4 * self.max_activities + self.max_activities * self.max_resources
        buffer[remaining_offset : remaining_offset + self.max_resources] = 0.0
        buffer[remaining_offset : remaining_offset + r] = (
            observation["remaining_capacity"]
            / np.maximum(np.asarray(env.instance.capacities, dtype=np.float32), 1.0)
        )
        buffer[remaining_offset + self.max_resources] = observation["current_time"][0] / max(env.horizon, 1)
        return buffer
