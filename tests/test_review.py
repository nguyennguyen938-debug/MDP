"""Checks for the statistics and the LinUCB baseline added in response to the reviews."""

from __future__ import annotations

import unittest

from scripts.review_numbers import mcnemar_exact, wilson
from testgenrl.actions import Action
from testgenrl.cli import build_parser
from testgenrl.planners import LinUCBPlanner, planner_from_name


class ReviewStatsTests(unittest.TestCase):
    def test_wilson_interval(self) -> None:
        lo, hi = wilson(4, 22)
        self.assertAlmostEqual(lo, 0.07307, places=4)
        self.assertAlmostEqual(hi, 0.38517, places=4)

    def test_mcnemar_exact_is_two_sided(self) -> None:
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(5, 0), 0.0625)
        self.assertAlmostEqual(mcnemar_exact(0, 5), 0.0625)
        self.assertAlmostEqual(mcnemar_exact(10, 0), 2 / 1024)


class LinUCBTests(unittest.TestCase):
    def test_tries_each_valid_action_before_repeating_one(self) -> None:
        planner = LinUCBPlanner()
        mask = [True, True, True, True, False, False, False, True]
        picks = []
        for _ in range(4):
            action = planner.select([0.5] * 24, mask)
            picks.append(action)
            planner.update(action, 0.0)
        self.assertEqual(picks, [0, 1, 2, 3])

    def test_stops_only_once_every_valid_action_scores_below_zero(self) -> None:
        planner = LinUCBPlanner()
        mask = [True, False, False, False, False, False, False, True]
        self.assertEqual(planner.select([0.1] * 24, mask), int(Action.GENERATE_BOUNDARY_TESTS))
        for _ in range(50):
            planner.update(int(Action.GENERATE_BOUNDARY_TESTS), -5.0)
        self.assertEqual(planner.select([0.1] * 24, mask), int(Action.STOP))

    def test_cli_and_factory_accept_linucb(self) -> None:
        self.assertIsInstance(planner_from_name("linucb"), LinUCBPlanner)
        args = build_parser().parse_args(["run", "--planner", "linucb", "--subjects", "s", "--out", "o"])
        self.assertEqual(args.planner, "linucb")
