"""Run one random-priority Gymnasium episode and write its legal Gantt chart."""

from pathlib import Path

from gantt import plot_gantt
from rcmpsp_env import RCMPSPEnv


INSTANCE = Path("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")
OUTPUT = Path("outputs/mp_j30_a2_nr1_random_env_gantt.png")


def main() -> None:
    env = RCMPSPEnv(INSTANCE)
    _, _ = env.reset(seed=7)
    env.action_space.seed(7)
    terminated = False
    total_reward = 0.0
    steps = 0
    while not terminated:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        if truncated:
            raise RuntimeError("unexpected truncation")
        total_reward += reward
        steps += 1

    schedule = env.schedule
    output = plot_gantt(env.instance, schedule, OUTPUT)
    print(f"instance={env.instance.name} steps={steps} makespan={schedule.makespan}")
    print(f"total_reward={total_reward:.1f} gantt={output}")


if __name__ == "__main__":
    main()
