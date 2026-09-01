from pathlib import Path
import unittest

from src.core.rcmpsp import generate_schedule, parse_rcmp, priority_fifo, priority_shortest_duration, random_priorities, validate_schedule
from src.visualization.aon import plot_aon
from src.visualization.gantt import plot_gantt


INSTANCE = Path("MPSPLIB/RCMP/mp_j30_a2_nr1.rcmp")


class RcmpspTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.instance = parse_rcmp(INSTANCE)

    def test_parser_reads_expected_structure(self) -> None:
        self.assertEqual(len(self.instance.activities), 64)
        self.assertEqual(self.instance.resource_count, 6)
        self.assertEqual(self.instance.capacities, (32, 62, 25, 25, 25, 25))
        self.assertEqual(self.instance.activities[(1, 1)].duration, 0)
        self.assertEqual(self.instance.activities[(1, 32)].duration, 0)
        for activity in self.instance.activities.values():
            for successor in activity.successors:
                self.assertIn(activity.id, self.instance.predecessors[successor])

    def test_baselines_generate_valid_schedules(self) -> None:
        priorities = [
            priority_fifo,
            priority_shortest_duration,
            random_priorities(self.instance, seed=7),
        ]
        for priority in priorities:
            schedule = generate_schedule(self.instance, priority)
            validate_schedule(self.instance, schedule)
            self.assertEqual(len(schedule.starts), 64)
            self.assertGreater(schedule.makespan, 0)

    def test_random_priority_is_reproducible(self) -> None:
        first = generate_schedule(self.instance, random_priorities(self.instance, seed=19))
        second = generate_schedule(self.instance, random_priorities(self.instance, seed=19))
        self.assertEqual(first, second)

    def test_gantt_chart_is_written(self) -> None:
        output = Path("test-output-gantt.png")
        try:
            schedule = generate_schedule(self.instance, priority_fifo)
            result = plot_gantt(self.instance, schedule, output)
            self.assertEqual(result, output)
            self.assertGreater(output.stat().st_size, 0)
        finally:
            if output.exists():
                output.unlink()

    def test_aon_chart_is_written(self) -> None:
        output = Path("test-output-aon.png")
        try:
            schedule = generate_schedule(self.instance, priority_fifo)
            result = plot_aon(self.instance, output, schedule=schedule)
            self.assertEqual(result, output)
            self.assertGreater(output.stat().st_size, 0)
        finally:
            if output.exists():
                output.unlink()


if __name__ == "__main__":
    unittest.main()
