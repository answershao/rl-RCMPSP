import unittest

from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from src.environments.rcmpsp_env import RCMPSPEnv
from src.environments.sb3_env import make_sb3_env


class Sb3Test(unittest.TestCase):
    def test_adapter_and_short_learning_run(self):
        base = RCMPSPEnv("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")
        check_env(base, warn=True, skip_render_check=True)
        env = make_sb3_env("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")
        actions = env.action_space.shape[0]
        model = TD3("MlpPolicy", env, learning_starts=10, batch_size=8,
                    action_noise=NormalActionNoise(np.zeros(actions), 0.1 * np.ones(actions)),
                    policy_kwargs={"net_arch": [32, 32]}, seed=1, verbose=0)
        model.learn(total_timesteps=16, progress_bar=False)
        observation, _ = env.reset(seed=2)
        action, _ = model.predict(observation)
        self.assertTrue(env.action_space.contains(action))


if __name__ == "__main__":
    unittest.main()
