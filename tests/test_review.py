"""Checks for the statistics and the LinUCB baseline added in response to the reviews."""

from __future__ import annotations

import unittest

from testgenrl.actions import Action
from testgenrl.cli import build_parser
from testgenrl.planners import LinUCBPlanner, planner_from_name


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
