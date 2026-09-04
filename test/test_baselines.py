from pathlib import Path
import tempfile
import unittest

from scripts.baselines import find_instances, instance_sort_key


class BaselinesTest(unittest.TestCase):
    def test_instance_sort_key_uses_numeric_set_and_instance_numbers(self) -> None:
        paths = [
            Path("MPLIB2_Set2_0.rcmp"),
            Path("MPLIB2_Set1_1002.rcmp"),
            Path("MPLIB2_Set1_99.rcmp"),
            Path("MPLIB2_Set1_9.rcmp"),
            Path("other.rcmp"),
        ]

        ordered = sorted(paths, key=instance_sort_key)

        self.assertEqual(
            [path.name for path in ordered],
            [
                "MPLIB2_Set1_9.rcmp",
                "MPLIB2_Set1_99.rcmp",
                "MPLIB2_Set1_1002.rcmp",
                "MPLIB2_Set2_0.rcmp",
                "other.rcmp",
            ],
        )

    def test_find_instances_uses_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "MPLIB2_Set2_0.rcmp",
                "MPLIB2_Set1_1002.rcmp",
                "MPLIB2_Set1_99.rcmp",
            ):
                (root / name).touch()

            self.assertEqual(
                [path.name for path in find_instances(root)],
                [
                    "MPLIB2_Set1_99.rcmp",
                    "MPLIB2_Set1_1002.rcmp",
                    "MPLIB2_Set2_0.rcmp",
                ],
            )


if __name__ == "__main__":
    unittest.main()
