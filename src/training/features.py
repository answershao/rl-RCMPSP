"""Structured feature extractors for padded RCMPSP observations."""

from __future__ import annotations

import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.environments.observation import ObservationLayout


class RCMPSPStructuredExtractor(BaseFeaturesExtractor):
    """Encode padded per-activity features with shared weights.

    The environment keeps a flat Box for SB3 compatibility.  This extractor
    reverses that layout into activity records, applies a shared MLP, and
    propagates messages across precedence edges before appending global
    capacity/time features. Bias-free activity layers keep zero padding zero.
    """

    def __init__(
        self,
        observation_space,
        *,
        max_activities: int,
        max_resources: int,
        max_successors: int = 3,
        graph_layers: int = 0,
        embedding_dim: int = 8,
        hidden_dim: int = 32,
        global_dim: int = 16,
    ) -> None:
        self.layout = ObservationLayout(max_activities, max_resources, max_successors)
        input_dim = self.layout.size
        if observation_space.shape != (input_dim,):
            raise ValueError(
                f"expected flattened RCMPSP observation {(input_dim,)}, "
                f"got {observation_space.shape}"
            )
        features_dim = max_activities * embedding_dim + global_dim
        super().__init__(observation_space, features_dim)
        self.max_activities = max_activities
        self.max_resources = max_resources
        self.max_successors = max_successors
        self.embedding_dim = embedding_dim

        activity_input_dim = 6 + max_resources
        self.activity_encoder = nn.Sequential(
            nn.Linear(activity_input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim, bias=False),
            nn.ReLU(),
        )
        self.graph_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * embedding_dim, embedding_dim, bias=False),
                    nn.ReLU(),
                )
                for _ in range(graph_layers)
            ]
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(max_resources + 1, global_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n = self.max_activities
        r = self.max_resources
        layout = self.layout
        status = observations[:, layout.activity_status]
        precedence = observations[:, layout.precedence_satisfied]
        durations = observations[:, layout.durations]
        eligible = observations[:, layout.eligible_mask]
        demands = observations[:, layout.resource_demands].reshape(-1, n, r)
        successors = observations[:, layout.successor_indices].reshape(-1, n, self.max_successors)
        successor_counts = observations[:, layout.successor_counts]
        downstream_durations = observations[:, layout.downstream_durations]
        global_features = observations[:, layout.global_features]

        activity_features = th.cat(
            [status.unsqueeze(-1), precedence.unsqueeze(-1), durations.unsqueeze(-1),
             eligible.unsqueeze(-1), successor_counts.unsqueeze(-1),
             downstream_durations.unsqueeze(-1), demands],
            dim=-1,
        )
        embeddings = self.activity_encoder(activity_features.reshape(-1, 6 + r))
        embeddings = embeddings.reshape(-1, n, self.embedding_dim)

        if not self.graph_layers:
            return th.cat(
                [embeddings.reshape(-1, n * self.embedding_dim), self.global_encoder(global_features)],
                dim=1,
            )

        # Slots store (successor index + 1) / n; zero is a padding slot.
        successor_indices = th.round(successors * n).long() - 1
        valid = (successor_indices >= 0).unsqueeze(-1).to(embeddings.dtype)
        successor_indices = successor_indices.clamp(min=0, max=n - 1)
        batch_size = embeddings.shape[0]
        flat_indices = successor_indices.reshape(batch_size, -1)

        for graph_layer in self.graph_layers:
            successor_embeddings = embeddings.gather(
                1,
                flat_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim),
            ).reshape(batch_size, n, self.max_successors, self.embedding_dim)
            successor_sum = (successor_embeddings * valid).sum(dim=2)
            successor_mean = successor_sum / valid.sum(dim=2).clamp(min=1.0)

            predecessor_sum = th.zeros_like(embeddings)
            predecessor_degree = th.zeros(
                batch_size, n, 1, dtype=embeddings.dtype, device=embeddings.device
            )
            source_messages = (
                embeddings.unsqueeze(2).expand(-1, -1, self.max_successors, -1) * valid
            ).reshape(batch_size, -1, self.embedding_dim)
            target_indices = flat_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            predecessor_sum.scatter_add_(1, target_indices, source_messages)
            predecessor_degree.scatter_add_(
                1, flat_indices.unsqueeze(-1), valid.reshape(batch_size, -1, 1)
            )
            predecessor_mean = predecessor_sum / predecessor_degree.clamp(min=1.0)
            embeddings = graph_layer(th.cat([embeddings, successor_mean, predecessor_mean], dim=-1))

        return th.cat(
            [embeddings.reshape(-1, n * self.embedding_dim), self.global_encoder(global_features)],
            dim=1,
        )
