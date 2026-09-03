"""Periodic deterministic evaluation for PPO architecture ablations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.observation import StaticGraphCache
from src.training.features import SharedDirectedGINExtractor
from src.training.ppo import evaluate_paths


@dataclass(frozen=True)
class EvaluationPoint:
    timesteps: int
    training_seconds: float
    fps: float
    mean_makespan: float
    min_makespan: float
    max_makespan: float


class PeriodicMakespanCallback(BaseCallback):
    """Evaluate makespan while excluding evaluation overhead from training time."""

    def __init__(
        self,
        *,
        evaluation_paths: list[str],
        reference_env,
        training_cache: StaticGraphCache,
        eval_freq: int,
        seed: int,
    ) -> None:
        super().__init__()
        if not evaluation_paths:
            raise ValueError("evaluation_paths must not be empty")
        if eval_freq < 1:
            raise ValueError("eval_freq must be positive")
        self.evaluation_paths = evaluation_paths
        self.reference_env = reference_env
        self.training_cache = training_cache
        self.eval_freq = eval_freq
        self.seed = seed
        self.points: list[EvaluationPoint] = []
        self._started_at = 0.0
        self._evaluation_seconds = 0.0
        self._next_evaluation = eval_freq

    def _on_training_start(self) -> None:
        self._started_at = perf_counter()
        self._record_evaluation()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_evaluation:
            self._record_evaluation()
            self._next_evaluation = (
                self.num_timesteps // self.eval_freq + 1
            ) * self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if not self.points or self.points[-1].timesteps != self.num_timesteps:
            self._record_evaluation()

    def _record_evaluation(self) -> None:
        training_seconds = (
            0.0
            if self.num_timesteps == 0
            else perf_counter() - self._started_at - self._evaluation_seconds
        )
        evaluation_started = perf_counter()
        extractor = self.model.policy.features_extractor
        if not isinstance(extractor, SharedDirectedGINExtractor):
            raise TypeError("ablation requires SharedDirectedGINExtractor")
        try:
            results = evaluate_paths(
                self.model,
                self.evaluation_paths,
                self.seed,
                self.reference_env,
            )
        finally:
            extractor.set_static_cache(self.training_cache)
        self._evaluation_seconds += perf_counter() - evaluation_started
        makespans = np.asarray([makespan for _, makespan in results], dtype=np.float64)
        fps = self.num_timesteps / training_seconds if training_seconds > 0 else 0.0
        point = EvaluationPoint(
            timesteps=self.num_timesteps,
            training_seconds=training_seconds,
            fps=fps,
            mean_makespan=float(makespans.mean()),
            min_makespan=float(makespans.min()),
            max_makespan=float(makespans.max()),
        )
        self.points.append(point)
        self.logger.record("ablation/eval_mean_makespan", point.mean_makespan)
        self.logger.record("ablation/training_fps", point.fps)
