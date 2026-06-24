import unittest

from src.app.main import temporal_nav_target


class AppTemporalTest(unittest.TestCase):
    def test_temporal_nav_target_moves_to_next_period(self):
        periods = ["2024-Q1", "2024-Q2", "2024-Q3"]

        self.assertEqual(temporal_nav_target(periods, "2024-Q2", "next"), "2024-Q3")


if __name__ == "__main__":
    unittest.main()
