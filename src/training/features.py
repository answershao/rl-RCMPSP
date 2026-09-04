"""Shared directed-GIN encoder and independent actor/critic heads."""

from __future__ import annotations

from contextlib import nullcontext

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.environments.observation import ObservationLayout, StaticGraphCache


class DirectedGINLayer(nn.Module):
    """GIN update with separate predecessor and successor relations."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.epsilon = nn.Parameter(th.zeros(1))
        self.predecessor_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.successor_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.update = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        embeddings: th.Tensor,
        predecessor_sum: th.Tensor,
        successor_sum: th.Tensor,
    ) -> th.Tensor:
        aggregate = (
            (1.0 + self.epsilon) * embeddings
            + self.predecessor_projection(predecessor_sum)
            + self.successor_projection(successor_sum)
        )
        return self.update(aggregate)


class SharedDirectedGINExtractor(BaseFeaturesExtractor):
    """Encode an RCMPSP graph once for both policy and value estimation."""

    def __init__(
        self,
        observation_space,
        *,
        max_activities: int,
        max_resources: int,
        static_cache: StaticGraphCache,
        max_successors: int = 3,
        gin_layers: int = 3,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        global_dim: int = 16,
        mixed_precision: str = "none",
    ) -> None:
        if gin_layers < 1:
            raise ValueError("gin_layers must be positive")
        self.layout = ObservationLayout(max_activities, max_resources)
        if observation_space.shape != (self.layout.size,):
            raise ValueError(
                f"expected flattened RCMPSP observation {(self.layout.size,)}, "
                f"got {observation_space.shape}"
            )
        features_dim = max_activities * embedding_dim + global_dim + 2 * max_activities
        super().__init__(observation_space, features_dim)
        self.max_activities = max_activities
        self.max_resources = max_resources
        self.max_successors = max_successors
        self.embedding_dim = embedding_dim
        self.global_dim = global_dim
        self.mixed_precision = mixed_precision
        self.set_static_cache(static_cache)

        self.global_encoder = nn.Sequential(
            nn.Linear(max_resources + 1, global_dim),
            nn.ReLU(),
        )
        activity_input_dim = 6 + max_resources + global_dim
        self.activity_encoder = nn.Sequential(
            nn.Linear(activity_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.gin_layers = nn.ModuleList(
            DirectedGINLayer(embedding_dim, hidden_dim) for _ in range(gin_layers)
        )

    def set_static_cache(self, static_cache: StaticGraphCache) -> None:
        """Replace immutable instance data, for example before a new evaluation split."""
        self.instance_count = static_cache.instance_count
        self.instance_names = static_cache.instance_names
        expected_node_shape = (self.instance_count, self.max_activities)
        if static_cache.durations.shape != expected_node_shape:
            raise ValueError("static cache does not match max_activities")
        if static_cache.resource_demands.shape != (
            self.instance_count, self.max_activities, self.max_resources
        ):
            raise ValueError("static cache does not match max_resources")
        if static_cache.successor_indices.shape != (
            self.instance_count, self.max_activities, self.max_successors
        ):
            raise ValueError("static cache does not match max_successors")
        arrays = {
            "static_durations": static_cache.durations,
            "static_resource_demands": static_cache.resource_demands,
            "static_successor_indices": static_cache.successor_indices,
            "static_successor_counts": static_cache.successor_counts,
            "static_downstream_durations": static_cache.downstream_durations,
            "static_activity_mask": static_cache.activity_mask,
        }
        device = self.static_durations.device if hasattr(self, "static_durations") else None
        for name, array in arrays.items():
            tensor = th.from_numpy(array).to(device=device)
            if name in self._buffers:
                setattr(self, name, tensor)
            else:
                self.register_buffer(name, tensor)

    def _autocast(self, tensor: th.Tensor):
        if not tensor.is_cuda or self.mixed_precision == "none":
            return nullcontext()
        dtype = th.bfloat16 if self.mixed_precision == "bf16" else th.float16
        return th.autocast(device_type="cuda", dtype=dtype)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        with self._autocast(observations):
            features = self._forward(observations)
        # Keep distributions and PPO losses in FP32; autocast is only needed
        # for the encoder's linear algebra.
        return features.float()

    def _forward(self, observations: th.Tensor) -> th.Tensor:
        n = self.max_activities
        layout = self.layout
        status = observations[:, layout.activity_status]
        precedence = observations[:, layout.precedence_satisfied]
        eligible = observations[:, layout.eligible_mask]
        instance_indices = (
            th.round(observations[:, layout.instance_index] * self.instance_count).long() - 1
        ).clamp(min=0, max=self.instance_count - 1)
        durations = self.static_durations[instance_indices]
        demands = self.static_resource_demands[instance_indices]
        successors = self.static_successor_indices[instance_indices]
        successor_counts = self.static_successor_counts[instance_indices]
        downstream_durations = self.static_downstream_durations[instance_indices]
        activity_mask = self.static_activity_mask[instance_indices]
        global_embedding = self.global_encoder(observations[:, layout.global_features])

        global_by_activity = global_embedding.unsqueeze(1).expand(-1, n, -1)
        activity_features = th.cat(
            [
                status.unsqueeze(-1),
                precedence.unsqueeze(-1),
                durations.unsqueeze(-1),
                eligible.unsqueeze(-1),
                successor_counts.unsqueeze(-1),
                downstream_durations.unsqueeze(-1),
                demands,
                global_by_activity,
            ],
            dim=-1,
        )
        embeddings = self.activity_encoder(activity_features)
        valid_nodes = activity_mask.unsqueeze(-1)
        embeddings = embeddings * valid_nodes

        valid_edges = (successors >= 0).unsqueeze(-1).to(embeddings.dtype)
        successor_indices = successors.clamp(min=0, max=n - 1)
        batch_size = embeddings.shape[0]
        flat_indices = successor_indices.reshape(batch_size, -1)

        for layer in self.gin_layers:
            successor_embeddings = embeddings.gather(
                1,
                flat_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim),
            ).reshape(batch_size, n, self.max_successors, self.embedding_dim)
            successor_sum = (successor_embeddings * valid_edges).sum(dim=2)

            predecessor_sum = th.zeros_like(embeddings)
            source_messages = (
                embeddings.unsqueeze(2).expand(-1, -1, self.max_successors, -1)
                * valid_edges
            ).reshape(batch_size, -1, self.embedding_dim)
            predecessor_sum.scatter_add_(
                1,
                flat_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim),
                source_messages,
            )
            embeddings = layer(embeddings, predecessor_sum, successor_sum) * valid_nodes

        return th.cat(
            [
                embeddings.reshape(-1, n * self.embedding_dim),
                global_embedding,
                activity_mask,
                eligible,
            ],
            dim=1,
        )


class GINActorCriticHeads(nn.Module):
    """Independent node-scoring actor and graph-pooling critic."""

    def __init__(
        self,
        *,
        max_activities: int,
        embedding_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        mixed_precision: str = "none",
    ) -> None:
        super().__init__()
        self.max_activities = max_activities
        self.embedding_dim = embedding_dim
        self.global_dim = global_dim
        self.mixed_precision = mixed_precision
        self.latent_dim_pi = max_activities
        self.latent_dim_vf = 1
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * embedding_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _autocast(self, tensor: th.Tensor):
        if not tensor.is_cuda or self.mixed_precision == "none":
            return nullcontext()
        dtype = th.bfloat16 if self.mixed_precision == "bf16" else th.float16
        return th.autocast(device_type="cuda", dtype=dtype)

    def _unpack(
        self, features: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        n = self.max_activities
        embedding_stop = n * self.embedding_dim
        global_stop = embedding_stop + self.global_dim
        mask_stop = global_stop + n
        embeddings = features[:, :embedding_stop].reshape(-1, n, self.embedding_dim)
        global_embedding = features[:, embedding_stop:global_stop]
        activity_mask = features[:, global_stop:mask_stop]
        eligible = features[:, mask_stop:mask_stop + n]
        return embeddings, global_embedding, activity_mask, eligible

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        with self._autocast(features):
            embeddings, _, activity_mask, eligible = self._unpack(features)
            logits = self.actor(embeddings).squeeze(-1)
            legal = (activity_mask > 0.5) & (eligible > 0.5)
            logits = logits.masked_fill(~legal, th.finfo(logits.dtype).min)
        return logits.float()

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        with self._autocast(features):
            embeddings, global_embedding, activity_mask, _ = self._unpack(features)
            valid = activity_mask.unsqueeze(-1)
            embedding_sum = (embeddings * valid).sum(dim=1)
            embedding_mean = embedding_sum / valid.sum(dim=1).clamp(min=1.0)
            graph_embedding = th.cat([embedding_sum, embedding_mean, global_embedding], dim=1)
            values = self.critic(graph_embedding)
        return values.float()

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
