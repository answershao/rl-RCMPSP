"""Gymnasium environment for priority-based RCMPSP schedule construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rcmpsp import ActivityId, Instance, Schedule, parse_rcmp, serial_sgs_insert, validate_schedule


class RCMPSPEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Construct a serial SSGS schedule using continuous activity priorities.

    One step selects the highest-priority precedence-eligible activity and
    inserts it at its earliest resource-feasible time. The episode therefore
    has exactly one scheduling decision per activity.
    """

    metadata = {"render_modes": []}

    def __init__(self, instance: Instance | str | Path):
        super().__init__()
        self.instance = parse_rcmp(instance) if isinstance(instance, (str, Path)) else instance
        self.activity_ids = tuple(sorted(self.instance.activities))
        self.activity_index = {activity_id: i for i, activity_id in enumerate(self.activity_ids)}
        self.activity_count = len(self.activity_ids)
        self.resource_count = self.instance.resource_count
        self.horizon = sum(activity.duration for activity in self.instance.activities.values())

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.activity_count,), dtype=np.float32
        )
        int_max = np.iinfo(np.int32).max
        capacities = np.asarray(self.instance.capacities, dtype=np.int32)
        max_duration = max((activity.duration for activity in self.instance.activities.values()), default=0)
        self.observation_space = spaces.Dict(
            {
                "activity_status": spaces.Box(0, 2, (self.activity_count,), dtype=np.int8),
                "precedence_satisfied": spaces.MultiBinary(self.activity_count),
                "durations": spaces.Box(0, max(max_duration, 1), (self.activity_count,), dtype=np.int32),
                "resource_demands": spaces.Box(
                    0, int_max, (self.activity_count, self.resource_count), dtype=np.int32
                ),
                "remaining_capacity": spaces.Box(
                    np.zeros(self.resource_count, dtype=np.int32), capacities, dtype=np.int32
                ),
                "current_time": spaces.Box(0, max(self.horizon, 1), (1,), dtype=np.int32),
                "eligible_mask": spaces.MultiBinary(self.activity_count),
            }
        )

        self._durations = np.asarray(
            [self.instance.activities[item].duration for item in self.activity_ids], dtype=np.int32
        )
        self._demands = np.asarray(
            [self.instance.activities[item].demand for item in self.activity_ids], dtype=np.int32
        )
        self.starts: dict[ActivityId, int] = {}
        self.finishes: dict[ActivityId, int] = {}
        self.usage: list[list[int]] = []
        self.current_time = 0
        self._terminated = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.starts = {}
        self.finishes = {}
        self.usage = []
        self.current_time = 0
        self._terminated = False
        observation = self._observation()
        return observation, {"eligible_mask": observation["eligible_mask"].copy()}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("step() called after the episode terminated; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be float32 with shape {self.action_space.shape} in [-1, 1]")

        eligible = self._eligible_ids()
        if not eligible:
            raise RuntimeError("precedence graph is cyclic or has a missing predecessor")
        chosen = max(
            eligible,
            key=lambda item: (float(action[self.activity_index[item]]), -self.activity_index[item]),
        )

        old_time = self.current_time
        start, finish = serial_sgs_insert(
            self.instance, chosen, self.starts, self.finishes, self.usage
        )
        self.current_time = max(self.finishes.values(), default=0)
        reward = float(-(self.current_time - old_time))
        self._terminated = len(self.starts) == self.activity_count

        observation = self._observation()
        info: dict[str, Any] = {
            "chosen_activity": chosen,
            "start": start,
            "finish": finish,
            "makespan": self.current_time,
            "eligible_mask": observation["eligible_mask"].copy(),
        }
        if self._terminated:
            info["schedule"] = self.schedule
        return observation, reward, self._terminated, False, info

    @property
    def schedule(self) -> Schedule:
        """Return and validate the completed schedule."""
        if not self._terminated:
            raise RuntimeError("the schedule is only available after episode termination")
        schedule = Schedule(dict(self.starts), dict(self.finishes), self.current_time)
        validate_schedule(self.instance, schedule)
        return schedule

    def _eligible_ids(self) -> list[ActivityId]:
        return [
            activity_id
            for activity_id in self.activity_ids
            if activity_id not in self.starts
            and all(pred in self.finishes for pred in self.instance.predecessors[activity_id])
        ]

    def _observation(self) -> dict[str, np.ndarray]:
        eligible = set(self._eligible_ids()) if not self._terminated else set()
        status = np.zeros(self.activity_count, dtype=np.int8)
        for activity_id, finish in self.finishes.items():
            status[self.activity_index[activity_id]] = 2 if self._terminated or finish < self.current_time else 1

        precedence = np.asarray(
            [
                all(pred in self.finishes for pred in self.instance.predecessors[item])
                for item in self.activity_ids
            ],
            dtype=np.int8,
        )
        eligible_mask = np.asarray(
            [item in eligible for item in self.activity_ids], dtype=np.int8
        )
        # The latest occupied interval gives a useful capacity signal at the
        # partial schedule frontier; at time zero no resource is occupied.
        frontier_usage = self.usage[self.current_time - 1] if self.current_time > 0 else [0] * self.resource_count
        remaining = np.asarray(self.instance.capacities, dtype=np.int32) - np.asarray(
            frontier_usage, dtype=np.int32
        )
        return {
            "activity_status": status,
            "precedence_satisfied": precedence,
            "durations": self._durations.copy(),
            "resource_demands": self._demands.copy(),
            "remaining_capacity": remaining,
            "current_time": np.asarray([self.current_time], dtype=np.int32),
            "eligible_mask": eligible_mask,
        }
