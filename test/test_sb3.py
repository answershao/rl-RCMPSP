import unittest

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.environments.rcmpsp_env import RCMPSPEnv
from src.environments.sb3_env import make_sb3_env
from src.training.environments import make_multi_env, make_single_env, make_vector_env


class Sb3Test(unittest.TestCase):
    def test_training_environment_factories(self):
        single = make_single_env("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")
        observation, _ = single.reset(seed=1)
        self.assertTrue(single.observation_space.contains(observation))

        vector_env = make_vector_env(
            [lambda: make_multi_env(["MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp"], seed=1)],
            parallel=False,
        )
        try:
            observation = vector_env.reset()
            self.assertTrue(vector_env.observation_space.contains(observation[0]))
        finally:
            vector_env.close()

    def test_adapter_and_short_learning_run(self):
        base = RCMPSPEnv("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")
        check_env(base, warn=True, skip_render_check=True)
        env = make_sb3_env("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")
        model = PPO(
            "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
            policy_kwargs={"net_arch": [32, 32]}, seed=1, verbose=0,
        )
        model.learn(total_timesteps=16, progress_bar=False)
        observation, _ = env.reset(seed=2)
        action, _ = model.predict(observation)
        self.assertTrue(env.action_space.contains(action))


if __name__ == "__main__":
    unittest.main()
