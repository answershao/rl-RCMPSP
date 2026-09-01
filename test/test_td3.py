import unittest
import numpy as np
import torch

from src.environments.rcmpsp_env import RCMPSPEnv
from src.rl.td3 import ReplayBuffer, TD3, flatten_observation


class Td3Test(unittest.TestCase):
    def test_update_and_action_shapes(self):
        env = RCMPSPEnv("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")
        obs, _ = env.reset(seed=1)
        state = flatten_observation(obs, env.instance.capacities, env.horizon)
        agent = TD3(state.size, env.activity_count)
        self.assertEqual(agent.select_action(state).shape, env.action_space.shape)
        replay = ReplayBuffer(1000, seed=1)
        for _ in range(4): replay.add(state, np.zeros(env.activity_count, dtype=np.float32), 0.0, state, False)
        agent.config.batch_size = 4
        result = agent.update(replay)
        self.assertIn("critic_loss", result)
        self.assertTrue(torch.isfinite(torch.tensor(result["critic_loss"])))


if __name__ == "__main__":
    unittest.main()
