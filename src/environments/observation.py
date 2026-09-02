"""Observation encoding helpers shared by the Gymnasium environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.core.rcmpsp import Instance


MAX_SUCCESSORS = 3


@dataclass(frozen=True)
class ObservationLayout:
    """Named slices for the flattened RCMPSP observation contract."""

    max_activities: int
    max_resources: int
    max_successors: int = MAX_SUCCESSORS

    def __post_init__(self) -> None:
        if min(self.max_activities, self.max_resources, self.max_successors) < 1:
            raise ValueError("observation dimensions must be positive")

    @property
    def activity_status(self) -> slice:
        return slice(0, self.max_activities)

    @property
    def precedence_satisfied(self) -> slice:
        return slice(self.activity_status.stop, self.activity_status.stop + self.max_activities)

    @property
    def durations(self) -> slice:
        return slice(self.precedence_satisfied.stop, self.precedence_satisfied.stop + self.max_activities)

    @property
    def eligible_mask(self) -> slice:
        return slice(self.durations.stop, self.durations.stop + self.max_activities)

    @property
    def resource_demands(self) -> slice:
        return slice(self.eligible_mask.stop, self.eligible_mask.stop + self.max_activities * self.max_resources)

    @property
    def successor_indices(self) -> slice:
        return slice(self.resource_demands.stop, self.resource_demands.stop + self.max_activities * self.max_successors)

    @property
    def successor_counts(self) -> slice:
        return slice(self.successor_indices.stop, self.successor_indices.stop + self.max_activities)

    @property
    def downstream_durations(self) -> slice:
        return slice(self.successor_counts.stop, self.successor_counts.stop + self.max_activities)

    @property
    def global_features(self) -> slice:
        return slice(self.downstream_durations.stop, self.downstream_durations.stop + self.max_resources + 1)

    @property
    def remaining_capacity(self) -> slice:
        return slice(self.global_features.start, self.global_features.stop - 1)

    @property
    def current_time(self) -> int:
        return self.global_features.stop - 1

    @property
    def size(self) -> int:
        return self.global_features.stop


@dataclass(frozen=True)
class ObservationTopology:
    """Static precedence features used by flattened RCMPSP observations."""

    successor_indices: np.ndarray
    successor_counts: np.ndarray
    downstream_durations: np.ndarray


def build_observation_topology(
    instance: Instance,
    activity_ids: tuple[tuple[int, int], ...],
    *,
    max_activities: int,
    horizon: int,
) -> ObservationTopology:
    """Precompute normalized precedence features for an instance.

    The result is independent of scheduling state and can be safely reused for
    every observation in an episode.
    """
    activity_count = len(activity_ids)
    if max_activities < activity_count:
        raise ValueError("max_activities cannot be smaller than the instance")

    activity_index = {activity_id: index for index, activity_id in enumerate(activity_ids)}
    successor_indices = np.zeros((max_activities, MAX_SUCCESSORS), dtype=np.float32)
    successor_counts = np.zeros(max_activities, dtype=np.float32)
    downstream_durations = np.zeros(max_activities, dtype=np.float32)
    longest_path: dict[tuple[int, int], int] = {}

    def downstream_duration(activity_id: tuple[int, int]) -> int:
        if activity_id not in longest_path:
            activity = instance.activities[activity_id]
            longest_path[activity_id] = activity.duration + max(
                (downstream_duration(successor) for successor in activity.successors), default=0
            )
        return longest_path[activity_id]

    for index, activity_id in enumerate(activity_ids):
        successors = instance.activities[activity_id].successors
        if len(successors) > MAX_SUCCESSORS:
            raise ValueError(f"activity {activity_id} has too many successors")
        successor_counts[index] = len(successors) / MAX_SUCCESSORS
        downstream_durations[index] = downstream_duration(activity_id) / max(horizon, 1)
        for slot, successor in enumerate(successors):
            successor_indices[index, slot] = (activity_index[successor] + 1) / max_activities

    return ObservationTopology(successor_indices, successor_counts, downstream_durations)


def flatten_observation(
    observation: Mapping[str, np.ndarray],
    capacities: tuple[int, ...],
    horizon: int,
    *,
    max_activities: int | None = None,
    max_resources: int | None = None,
    successor_indices: np.ndarray | None = None,
    successor_counts: np.ndarray | None = None,
    downstream_durations: np.ndarray | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Flatten and normalize an observation, optionally padding its axes."""
    activity_count = observation["activity_status"].size
    resource_count = len(capacities)
    max_activities = max_activities or activity_count
    max_resources = max_resources or resource_count
    if max_activities < activity_count or max_resources < resource_count:
        raise ValueError("padding dimensions cannot be smaller than the observation")

    layout = ObservationLayout(max_activities, max_resources)
    size = layout.size
    if out is None:
        result = np.zeros(size, dtype=np.float32)
    else:
        if out.shape != (size,) or out.dtype != np.float32:
            raise ValueError("out must be a float32 array with the flattened observation shape")
        result = out
        result.fill(0.0)
    status = observation["activity_status"].astype(np.float32) / 2.0
    precedence = observation["precedence_satisfied"].astype(np.float32)
    durations = observation["durations"].astype(np.float32) / max(horizon, 1)
    eligible = observation["eligible_mask"].astype(np.float32)
    demands = observation["resource_demands"].astype(np.float32)
    demands /= np.maximum(np.asarray(capacities, dtype=np.float32), 1.0)
    remaining = observation["remaining_capacity"].astype(np.float32)
    remaining /= np.maximum(np.asarray(capacities, dtype=np.float32), 1.0)
    current_time = observation["current_time"].astype(np.float32) / max(horizon, 1)

    for field, values in (
        (layout.activity_status, status),
        (layout.precedence_satisfied, precedence),
        (layout.durations, durations),
        (layout.eligible_mask, eligible),
    ):
        result[field.start : field.start + activity_count] = values
    result[layout.resource_demands].reshape(
        max_activities, max_resources
    )[:activity_count, :resource_count] = demands
    if successor_indices is not None:
        expected_shape = (max_activities, layout.max_successors)
        if successor_indices.shape != expected_shape:
            raise ValueError(f"expected successor indices with shape {expected_shape}")
        result[layout.successor_indices] = successor_indices.ravel()
    if successor_counts is not None:
        if successor_counts.shape != (max_activities,):
            raise ValueError(f"expected successor counts with shape {(max_activities,)}")
        result[layout.successor_counts] = successor_counts
    if downstream_durations is not None:
        if downstream_durations.shape != (max_activities,):
            raise ValueError(f"expected downstream durations with shape {(max_activities,)}")
        result[layout.downstream_durations] = downstream_durations
    result[layout.remaining_capacity.start : layout.remaining_capacity.start + resource_count] = remaining
    result[layout.current_time] = current_time[0]
    return result


def observation_size(activity_count: int, resource_count: int) -> int:
    """Return the vector length for a flattened RCMPSP observation."""
    return ObservationLayout(activity_count, resource_count).size
