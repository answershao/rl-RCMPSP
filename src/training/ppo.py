"""PPO configuration and evaluation helpers for RCMPSP experiments."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Protocol

import numpy as np
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

from src.core.rcmpsp import (
    Instance,
    parse_rcmp,
)
from src.environments.observation import MAX_SUCCESSORS, build_static_graph_cache
from src.training.features import GINActorCriticHeads, SharedDirectedGINExtractor


class PolicyEnvironment(Protocol):
    def reset(self, *, seed: int | None = None): ...

    def step(self, action): ...


class GINActorCriticPolicy(ActorCriticPolicy):
    """PPO policy with a shared GIN trunk and independent task-specific heads."""

    def _build_mlp_extractor(self) -> None:
        extractor = self.features_extractor
        if not isinstance(extractor, SharedDirectedGINExtractor):
            raise TypeError("GINActorCriticPolicy requires SharedDirectedGINExtractor")
        self.mlp_extractor = GINActorCriticHeads(
            max_activities=extractor.max_activities,
            embedding_dim=extractor.embedding_dim,
            global_dim=extractor.global_dim,
            mixed_precision=extractor.mixed_precision,
        )

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        # The actor head already emits one masked logit per discrete action and
        # the critic head already emits a scalar value.
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        if self.ortho_init:
            self.features_extractor.apply(
                partial(self.init_weights, gain=np.sqrt(2))
            )
            self.mlp_extractor.apply(
                partial(self.init_weights, gain=np.sqrt(2))
            )
            self.mlp_extractor.actor[-1].apply(
                partial(self.init_weights, gain=0.01)
            )
            self.mlp_extractor.critic[-1].apply(
                partial(self.init_weights, gain=1.0)
            )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


def create_ppo(
    env,
    *,
    instances: list[Instance],
    seed: int,
    device: str = "auto",
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 2,
    gin_layers: int = 2,
    mixed_precision: str = "none",
    torch_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    tensorboard_log: str | None = None,
) -> PPO:
    """Create PPO with a shared directed GIN and masked discrete actor.

    The actor selects one eligible activity and the environment inserts it at
    its earliest precedence- and resource-feasible start time.
    """
    if n_steps < 1 or batch_size < 1 or n_epochs < 1 or gin_layers < 1:
        raise ValueError("n_steps, batch_size, n_epochs, and gin_layers must be positive")
    if not instances:
        raise ValueError("instances must not be empty")
    if mixed_precision not in {"none", "bf16", "fp16"}:
        raise ValueError("mixed_precision must be one of: none, bf16, fp16")
    action_count = int(env.action_space.n)
    base_env = env.envs[0] if hasattr(env, "envs") else env
    max_activities = action_count
    max_resources = int(env.observation_space.shape[0]) - 3 * max_activities - 2
    if max_resources < max(instance.resource_count for instance in instances):
        raise ValueError("environment observation cannot represent all instance resources")
    max_successors = getattr(base_env, "max_successors", MAX_SUCCESSORS)
    static_cache = build_static_graph_cache(
        instances,
        max_activities=max_activities,
        max_resources=max_resources,
        max_successors=max_successors,
    )
    model = PPO(
        GINActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            "features_extractor_class": SharedDirectedGINExtractor,
            "features_extractor_kwargs": {
                "max_activities": max_activities,
                "max_resources": max_resources,
                "static_cache": static_cache,
                "max_successors": max_successors,
                "gin_layers": gin_layers,
                "embedding_dim": 32,
                "hidden_dim": 64,
                "global_dim": 16,
                "mixed_precision": mixed_precision,
            },
        },
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
    if torch_compile:
        # Compile only the compute-heavy modules. Keeping the SB3 policy itself
        # unwrapped preserves its save/load and callback interfaces.
        model.policy.features_extractor.compile(mode=compile_mode, dynamic=True)
        model.policy.mlp_extractor.compile(mode=compile_mode, dynamic=True)
    return model


def run_policy_episode(model: PPO, env: PolicyEnvironment, *, seed: int | None = None) -> dict:
    """Run one deterministic policy episode and return its terminal info."""
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
    return info


def evaluate_paths(
    model: PPO, paths: list[str], seed: int, reference_env
) -> list[tuple[str, float]]:
    """Evaluate a padded multi-instance policy once for each supplied path."""
    from src.environments.multi_instance import MultiInstanceRCMPSPEnv

    if not paths:
        return []
    extractor = model.policy.features_extractor
    if not isinstance(extractor, SharedDirectedGINExtractor):
        raise TypeError("model does not use the RCMPSP static graph cache")
    evaluation_instances = [parse_rcmp(path) for path in paths]
    extractor.set_static_cache(
        build_static_graph_cache(
            evaluation_instances,
            max_activities=reference_env.max_activities,
            max_resources=reference_env.max_resources,
            max_successors=extractor.max_successors,
        )
    )
    results = []
    for offset, path in enumerate(paths):
        env = MultiInstanceRCMPSPEnv(
            [path],
            max_activities=reference_env.max_activities,
            max_resources=reference_env.max_resources,
            instance_indices=[offset],
            catalog_size=extractor.instance_count,
        )
        info = run_policy_episode(model, env, seed=seed + offset)
        results.append((Path(path).name, float(info["makespan"])))
    return results
