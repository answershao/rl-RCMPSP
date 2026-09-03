import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.environments.observation import ObservationLayout
from src.environments.rcmpsp_env import RCMPSPEnv
from src.environments.sb3_env import make_sb3_env
from src.training.environments import make_multi_env, make_single_env, make_vector_env
from src.training.ppo import create_ppo


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
        model = create_ppo(
            env, instances=[base.instance], n_steps=8, batch_size=8,
            n_epochs=1, gin_layers=1,
            seed=1, device="cpu",
        )
        model.learn(total_timesteps=16, progress_bar=False)
        observation, _ = env.reset(seed=2)
        action, _ = model.predict(observation)
        self.assertTrue(env.action_space.contains(action))

        observation_tensor, _ = model.policy.obs_to_tensor(observation)
        distribution = model.policy.get_distribution(observation_tensor)
        probabilities = distribution.distribution.probs.detach().cpu().numpy()[0]
        base_env = env.unwrapped
        layout = ObservationLayout(base_env.activity_count, base_env.resource_count)
        eligible = observation[layout.eligible_mask] > 0.5
        np.testing.assert_array_equal(probabilities[~eligible], 0.0)
        self.assertAlmostEqual(float(probabilities[eligible].sum()), 1.0)
        self.assertTrue(model.policy.share_features_extractor)
        actor_parameters = {id(item) for item in model.policy.mlp_extractor.actor.parameters()}
        critic_parameters = {id(item) for item in model.policy.mlp_extractor.critic.parameters()}
        self.assertTrue(actor_parameters.isdisjoint(critic_parameters))

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model.save(model_path)
            restored = PPO.load(model_path, device="cpu")
            restored_action, _ = restored.predict(observation, deterministic=True)
            expected_action, _ = model.predict(observation, deterministic=True)
            np.testing.assert_array_equal(restored_action, expected_action)


if __name__ == "__main__":
    unittest.main()
