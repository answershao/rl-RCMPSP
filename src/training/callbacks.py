"""Stable-Baselines3 callbacks for RCMPSP-specific training diagnostics."""

from __future__ import annotations

from collections import deque
from pathlib import Path

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

    def __init__(
        self,
        window_size: int = 100,
        checkpoint_dir: str | Path | None = None,
        early_stop_patience: int = 0,
        makespan_min_delta: float = 0.0,
    ):
        super().__init__()
        if early_stop_patience < 0 or makespan_min_delta < 0:
            raise ValueError("early-stop values must be non-negative")
        self.window_size = window_size
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.early_stop_patience = early_stop_patience
        self.makespan_min_delta = makespan_min_delta
        self.best_makespan = np.inf
        self._bad_makespan_rollouts = 0
        self._stop_requested = False
        self._values = {name: deque(maxlen=window_size) for name in TERMINAL_METRICS}

    def _on_step(self) -> bool:
        if self._stop_requested:
            return False
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

        makespans = self._values["makespan"]
        if makespans:
            makespan_mean = float(np.mean(makespans))
            improved = makespan_mean < self.best_makespan - self.makespan_min_delta
            if improved:
                self.best_makespan = makespan_mean
                self._bad_makespan_rollouts = 0
                if self.checkpoint_dir is not None:
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(self.checkpoint_dir / "best_model"))
                self.logger.record("rcmpsp/best_makespan", self.best_makespan)
            elif self.early_stop_patience:
                self._bad_makespan_rollouts += 1
                if self._bad_makespan_rollouts >= self.early_stop_patience:
                    self._stop_requested = True
                    self.logger.record("rcmpsp/early_stop", 1)
                    print(
                        "Early stopping: no makespan improvement for "
                        f"{self.early_stop_patience} rollouts"
                    )
    def _on_training_end(self) -> None:
        return None
