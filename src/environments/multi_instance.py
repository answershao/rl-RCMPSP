"""Fixed train/validation/test splits and a padded multi-instance environment."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.rcmpsp import parse_rcmp
from src.environments.rcmpsp_env import RCMPSPEnv
from src.environments.observation import (
    MAX_SUCCESSORS,
    ObservationTopology,
    build_observation_topology,
    flatten_observation,
    observation_size,
)


def make_splits(root: str | Path = "MPSPLIB/RCMP") -> dict[str, list[str]]:
    """Return all five j30/a2 instances as the training set."""
    files = sorted(Path(root).glob("mp_j30_*.rcmp"))
    paths = [path for path in files if "mp_j30_a2_" in path.name]
    if len(paths) < 5:
        raise ValueError("expected five j30/a2 instances")
    return {"train": [str(path) for path in paths[:5]], "validation": [], "test": []}


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
        self.max_successors = MAX_SUCCESSORS
        self.max_horizon = max([max_horizon or 0] + [sum(a.duration for a in instance.activities.values()) for instance in self.instances])
        self.action_space = spaces.Box(-1.0, 1.0, (self.max_activities,), dtype=np.float32)
        feature_size = observation_size(self.max_activities, self.max_resources)
        self.observation_space = spaces.Box(0.0, 1.0, (feature_size,), dtype=np.float32)
        # Reuse environment objects across episodes; reset only clears mutable
        # scheduling state and avoids repeated allocation of large buffers.
        self._envs = [RCMPSPEnv(instance) for instance in self.instances]
        self._env: RCMPSPEnv | None = None
        self._flat_buffers: dict[int, np.ndarray] = {}
        self._topologies: dict[int, ObservationTopology] = {}
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
            self._flat_buffers[self._active_index] = buffer
        topology = self._topologies.get(self._active_index)
        if topology is None:
            topology = build_observation_topology(
                env.instance,
                env.activity_ids,
                max_activities=self.max_activities,
                horizon=env.horizon,
            )
            self._topologies[self._active_index] = topology
        return flatten_observation(
            observation,
            env.instance.capacities,
            env.horizon,
            max_activities=self.max_activities,
            max_resources=self.max_resources,
            successor_indices=topology.successor_indices,
            successor_counts=topology.successor_counts,
            downstream_durations=topology.downstream_durations,
            capacity_scale=self._capacity_scales[self._active_index],
            out=buffer,
        )
