"""Observation encoding helpers shared by the Gymnasium environments."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def flatten_observation(
    observation: Mapping[str, np.ndarray],
    capacities: tuple[int, ...],
    horizon: int,
    *,
    max_activities: int | None = None,
    max_resources: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Flatten and normalize an observation, optionally padding its axes."""
    activity_count = observation["activity_status"].size
    resource_count = len(capacities)
    max_activities = max_activities or activity_count
    max_resources = max_resources or resource_count
    if max_activities < activity_count or max_resources < resource_count:
        raise ValueError("padding dimensions cannot be smaller than the observation")

    size = observation_size(max_activities, max_resources)
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

    offset = 0
    for values in (status, precedence, durations, eligible):
        result[offset : offset + activity_count] = values
        offset += max_activities
    result[offset : offset + max_activities * max_resources].reshape(
        max_activities, max_resources
    )[:activity_count, :resource_count] = demands
    offset += max_activities * max_resources
    result[offset : offset + resource_count] = remaining
    offset += max_resources
    result[offset] = current_time[0]
    return result


def observation_size(activity_count: int, resource_count: int) -> int:
    """Return the vector length for a flattened RCMPSP observation."""
    return activity_count * 4 + activity_count * resource_count + resource_count + 1
