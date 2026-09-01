import unittest
import numpy as np

from src.environments.multi_instance import MultiInstanceRCMPSPEnv, make_splits


class MultiInstanceTest(unittest.TestCase):
    def test_fixed_padding_and_episode(self):
        splits = make_splits()
        env = MultiInstanceRCMPSPEnv(splits["train"][:2], seed=3)
        obs, _ = env.reset(seed=3)
        self.assertTrue(env.observation_space.contains(obs))
        terminated = False
        while not terminated:
            obs, _, terminated, truncated, _ = env.step(np.zeros(env.max_activities, dtype=np.float32))
            self.assertFalse(truncated)
        self.assertTrue(env.observation_space.contains(obs))


if __name__ == "__main__":
    unittest.main()
