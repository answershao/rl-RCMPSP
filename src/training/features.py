"""Structured feature extractors for padded RCMPSP observations."""

from __future__ import annotations

import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RCMPSPStructuredExtractor(BaseFeaturesExtractor):
    """Encode padded per-activity features with shared weights.

    The environment keeps a flat Box for SB3 compatibility.  This extractor
    reverses that layout into activity records, applies the same small MLP to
    every activity, and appends the global capacity/time features.  Bias-free
    activity layers ensure all-zero padding stays zero without needing an
    explicit activity-count feature.
    """

    def __init__(
        self,
        observation_space,
        *,
        max_activities: int,
        max_resources: int,
        embedding_dim: int = 8,
        hidden_dim: int = 32,
        global_dim: int = 16,
    ) -> None:
        input_dim = 4 * max_activities + max_activities * max_resources + max_resources + 1
        if observation_space.shape != (input_dim,):
            raise ValueError(
                f"expected flattened RCMPSP observation {(input_dim,)}, "
                f"got {observation_space.shape}"
            )
        features_dim = max_activities * embedding_dim + global_dim
        super().__init__(observation_space, features_dim)
        self.max_activities = max_activities
        self.max_resources = max_resources
        self.embedding_dim = embedding_dim

        activity_input_dim = 4 + max_resources
        self.activity_encoder = nn.Sequential(
            nn.Linear(activity_input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim, bias=False),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(max_resources + 1, global_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n = self.max_activities
        r = self.max_resources
        status = observations[:, 0:n]
        precedence = observations[:, n : 2 * n]
        durations = observations[:, 2 * n : 3 * n]
        eligible = observations[:, 3 * n : 4 * n]
        demand_start = 4 * n
        demands = observations[:, demand_start : demand_start + n * r].reshape(-1, n, r)
        global_start = demand_start + n * r
        global_features = observations[:, global_start : global_start + r + 1]

        activity_features = th.cat(
            [status.unsqueeze(-1), precedence.unsqueeze(-1), durations.unsqueeze(-1),
             eligible.unsqueeze(-1), demands],
            dim=-1,
        )
        embeddings = self.activity_encoder(activity_features.reshape(-1, 4 + r))
        embeddings = embeddings.reshape(-1, n * self.embedding_dim)
        return th.cat([embeddings, self.global_encoder(global_features)], dim=1)
