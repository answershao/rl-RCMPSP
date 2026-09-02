from pathlib import Path
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from src.core.rcmpsp import generate_schedule, parse_rcmp, validate_schedule
from src.environments.rcmpsp_env import RCMPSPEnv, RESOURCE_UTILIZATION_WEIGHT


INSTANCE = Path("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")


class RcmpspEnvTest(unittest.TestCase):
    def test_gymnasium_interface(self) -> None:
        check_env(RCMPSPEnv(INSTANCE), skip_render_check=True)

    def test_random_episode_has_legal_schedule_and_telescoping_reward(self) -> None:
        env = RCMPSPEnv(INSTANCE)
        observation, info = env.reset(seed=11)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIn("eligible_mask", info)

        terminated = False
        total_reward = 0.0
        steps = 0
        rng = np.random.default_rng(11)
        while not terminated:
            action = rng.uniform(-1.0, 1.0, env.action_space.shape).astype(np.float32)
            observation, reward, terminated, truncated, terminal_info = env.step(action)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertFalse(truncated)
            total_reward += reward
            steps += 1

        schedule = env.schedule
        validate_schedule(env.instance, schedule)
        self.assertEqual(steps, len(env.instance.activities))
        expected_return = (
            -schedule.makespan / env.horizon
            + RESOURCE_UTILIZATION_WEIGHT * env._resource_utilization(schedule.makespan)
        )
        self.assertAlmostEqual(total_reward, expected_return)
        self.assertTrue(np.all(observation["activity_status"] == 2))
        self.assertEqual(terminal_info["makespan"], schedule.makespan)
        self.assertEqual(terminal_info["normalized_makespan"], schedule.makespan / env.horizon)
        self.assertGreaterEqual(terminal_info["resource_utilization"], 0.0)
        self.assertLessEqual(terminal_info["resource_utilization"], 1.0)
        self.assertEqual(terminal_info["activity_count"], env.activity_count)
        self.assertAlmostEqual(terminal_info["episode_makespan_penalty"], -schedule.makespan / env.horizon)
        self.assertAlmostEqual(
            terminal_info["episode_utilization_bonus"],
            RESOURCE_UTILIZATION_WEIGHT * env._resource_utilization(schedule.makespan),
        )
        self.assertAlmostEqual(terminal_info["episode_reward"], total_reward)

    def test_incremental_decoder_matches_batch_ssgs(self) -> None:
        instance = parse_rcmp(INSTANCE)
        env = RCMPSPEnv(instance)
        env.reset(seed=23)
        priorities = np.random.default_rng(23).uniform(
            -1.0, 1.0, env.action_space.shape
        ).astype(np.float32)

        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(priorities)

        priority_map = {
            activity_id: float(priorities[index])
            for index, activity_id in enumerate(env.activity_ids)
        }
        self.assertEqual(env.schedule, generate_schedule(instance, priority_map))


if __name__ == "__main__":
    unittest.main()
