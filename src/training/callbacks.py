"""Stable-Baselines3 callbacks for RCMPSP-specific training diagnostics."""

from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


TERMINAL_METRICS = (
    "makespan",
    "normalized_makespan",
    "resource_utilization",
    "activity_count",
    "episode_makespan_penalty",
    "episode_utilization_bonus",
    "episode_fifo_relative_bonus",
    "episode_invalid_action_penalty",
    "episode_reward",
)


class RCMPSPMetricsCallback(BaseCallback):
    """Log rolling task metrics from terminal environment ``info`` dictionaries."""

    def __init__(self, window_size: int = 100):
        super().__init__()
        self.window_size = window_size
        self._values = {name: deque(maxlen=window_size) for name in TERMINAL_METRICS}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # ``makespan`` is the partial schedule frontier on ordinary
            # transitions. Monitor adds ``episode`` only when it terminates.
            if "episode" not in info:
                continue
            for name, values in self._values.items():
                value = info.get(name)
                if value is not None:
                    values.append(float(value))
        return True

    def _on_rollout_end(self) -> None:
        for name, values in self._values.items():
            if values:
                self.logger.record(f"rcmpsp/{name}_mean", float(np.mean(values)))
