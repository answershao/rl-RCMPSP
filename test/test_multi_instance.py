import unittest
import numpy as np

from src.environments.multi_instance import MultiInstanceRCMPSPEnv, make_splits
from src.environments.observation import MAX_SUCCESSORS, ObservationLayout
from src.environments.sb3_env import make_sb3_env


class MultiInstanceTest(unittest.TestCase):
    def test_fixed_padding_and_episode(self):
        splits = make_splits()
        env = MultiInstanceRCMPSPEnv(splits["train"][:2], seed=3)
        obs, _ = env.reset(seed=3)
        self.assertTrue(env.observation_space.contains(obs))

    def test_observation_contains_successor_indices(self):
        env = MultiInstanceRCMPSPEnv(["MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp"], seed=3)
        observation, _ = env.reset(seed=3)
        active = env.active_env
        activity_id = active.activity_ids[0]
        successors = active.instance.activities[activity_id].successors
        layout = ObservationLayout(env.max_activities, env.max_resources)
        successor_view = observation[layout.successor_indices].reshape(
            env.max_activities, MAX_SUCCESSORS
        )
        expected = [(active.activity_index[item] + 1) / env.max_activities for item in successors]
        np.testing.assert_allclose(successor_view[0, : len(successors)], expected)
        terminated = False
        while not terminated:
            obs, _, terminated, truncated, _ = env.step(np.zeros(env.max_activities, dtype=np.float32))
            self.assertFalse(truncated)
        self.assertTrue(env.observation_space.contains(obs))

    def test_unpadded_encoding_matches_single_instance_wrapper(self):
        path = "MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp"
        multi_env = MultiInstanceRCMPSPEnv([path], seed=3)
        single_env = make_sb3_env(path)
        multi_observation, _ = multi_env.reset(seed=7)
        single_observation, _ = single_env.reset(seed=7)
        np.testing.assert_allclose(multi_observation, single_observation)

    def test_observation_layout_covers_each_feature_once(self):
        layout = ObservationLayout(max_activities=5, max_resources=2)
        fields = (
            layout.activity_status,
            layout.precedence_satisfied,
            layout.durations,
            layout.eligible_mask,
            layout.resource_demands,
            layout.successor_indices,
            layout.successor_counts,
            layout.downstream_durations,
            layout.global_features,
        )
        self.assertEqual(fields[0].start, 0)
        self.assertEqual(fields[-1].stop, layout.size)
        self.assertTrue(all(first.stop == second.start for first, second in zip(fields, fields[1:])))


if __name__ == "__main__":
    unittest.main()
