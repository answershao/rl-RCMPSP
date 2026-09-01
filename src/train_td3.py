"""Train and evaluate TD3 on one j30 RCMPSP instance."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from gantt import plot_gantt
from rcmpsp import generate_schedule, parse_rcmp, priority_fifo, priority_shortest_duration, random_priorities
from rcmpsp_env import RCMPSPEnv
from td3 import ReplayBuffer, TD3, flatten_observation


def run_episode(env, agent=None, replay=None, train=True, seed=None, exploration_noise=0.1):
    observation, _ = env.reset(seed=seed)
    state = flatten_observation(observation, env.instance.capacities, env.horizon)
    total = 0.0
    terminated = False
    while not terminated:
        action = env.action_space.sample() if agent is None else agent.select_action(state, exploration_noise if train else 0.0)
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = flatten_observation(next_observation, env.instance.capacities, env.horizon)
        if replay is not None:
            replay.add(state, action, reward, next_state, terminated or truncated)
            agent.update(replay)
        state, total = next_state, total + reward
    return -total, info["schedule"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path, nargs="?", default=Path("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gantt", type=Path, default=Path("outputs/mp_j30_a2_nr1_td3_gantt.png"))
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    env = RCMPSPEnv(args.instance); probe, _ = env.reset(seed=args.seed)
    state_dim = flatten_observation(probe, env.instance.capacities, env.horizon).size
    agent = TD3(state_dim, env.activity_count)
    replay = ReplayBuffer(seed=args.seed)
    best = None
    for episode in range(args.episodes):
        makespan, schedule = run_episode(env, agent, replay, True, args.seed + episode)
        best = (makespan, schedule) if best is None or makespan < best[0] else best
        if (episode + 1) % max(1, args.episodes // 10) == 0:
            print(f"episode={episode + 1} makespan={makespan:.0f} best={best[0]:.0f} replay={len(replay)}")
    fifo = generate_schedule(env.instance, priority_fifo).makespan
    shortest = generate_schedule(env.instance, priority_shortest_duration).makespan
    random_baseline = generate_schedule(env.instance, random_priorities(env.instance, args.seed)).makespan
    print(f"baselines fifo={fifo} shortest={shortest} random={random_baseline}")
    print(f"td3_best_makespan={best[0]:.0f}")
    plot_gantt(env.instance, best[1], args.gantt, title=f"TD3 schedule (Cmax={best[0]:.0f})")


if __name__ == "__main__":
    main()
