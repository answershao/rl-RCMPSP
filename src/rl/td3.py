"""Small, dependency-light TD3 implementation for the RCMPSP environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn


def flatten_observation(observation: Mapping[str, np.ndarray], capacities: tuple[int, ...], horizon: int) -> np.ndarray:
    """Encode the structured environment observation as a normalized vector."""
    cap = np.asarray(capacities, dtype=np.float32)
    horizon_value = float(max(horizon, 1))
    status = np.asarray(observation["activity_status"], dtype=np.float32) / 2.0
    precedence = np.asarray(observation["precedence_satisfied"], dtype=np.float32)
    durations = np.asarray(observation["durations"], dtype=np.float32) / horizon_value
    demands = np.asarray(observation["resource_demands"], dtype=np.float32) / np.maximum(cap, 1.0)
    remaining = np.asarray(observation["remaining_capacity"], dtype=np.float32) / np.maximum(cap, 1.0)
    time = np.asarray(observation["current_time"], dtype=np.float32) / horizon_value
    eligible = np.asarray(observation["eligible_mask"], dtype=np.float32)
    return np.concatenate((status, precedence, durations, demands.ravel(), remaining, time, eligible)).astype(np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000, seed: int | None = None):
        self.data = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        self.data.append((np.asarray(state, dtype=np.float32), np.asarray(action, dtype=np.float32), float(reward), np.asarray(next_state, dtype=np.float32), float(done)))

    def sample(self, batch_size: int, device: torch.device):
        indices = self.rng.integers(0, len(self.data), size=batch_size)
        batch = [self.data[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return tuple(torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device) for values in (states, actions, rewards, next_states, dones))

    def __len__(self) -> int:
        return len(self.data)


def _mlp(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, outputs))


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = _mlp(state_dim, action_dim)

    def forward(self, state):
        return torch.tanh(self.net(state))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = _mlp(state_dim + action_dim, 1)

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1))


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 5e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    batch_size: int = 100


class TD3:
    def __init__(self, state_dim: int, action_dim: int, config: TD3Config | None = None, device: str = "cpu"):
        self.config = config or TD3Config()
        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.config.critic_lr)
        self.updates = 0

    def select_action(self, state: np.ndarray, noise: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))[0].cpu().numpy()
        if noise:
            action = action + np.random.normal(0.0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, replay: ReplayBuffer) -> dict[str, float] | None:
        if len(replay) < self.config.batch_size:
            return None
        states, actions, rewards, next_states, dones = replay.sample(self.config.batch_size, self.device)
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.config.policy_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target = rewards.unsqueeze(-1) + self.config.gamma * (1 - dones.unsqueeze(-1)) * torch.minimum(self.critic1_target(next_states, next_actions), self.critic2_target(next_states, next_actions))
        q1, q2 = self.critic1(states, actions), self.critic2(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
        self.updates += 1
        result = {"critic_loss": float(critic_loss.item())}
        if self.updates % self.config.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            result["actor_loss"] = float(actor_loss.item())
        return result

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1 - self.config.tau).add_(self.config.tau * source_param.data)
