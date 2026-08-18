#!/usr/bin/env python3
"""Regression tests for the 2026-07-10 TXF strategy audit fixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


sys.path.insert(0, "/Users/guichenxiang/taifutures_strategy")
sys.path.insert(0, "/Users/guichenxiang/txf_backtest")

import audited_strategy_evolution as audit
import shioaji_paper_test as paper
import strategy_optimization as optimization
import txf_strategy_monitor as monitor


class StrategySafetyTests(unittest.TestCase):
    def test_basis_holiday_mapping_does_not_forward_fill_stale_true(self) -> None:
        trigger = pd.Series(
            [True, False, False],
            index=pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-07"]),
        )
        trading_day = pd.Series(pd.to_datetime(["2025-04-02", "2025-04-07"]))
        mapped = audit.map_prior_exchange_day_signal(trigger, trading_day)
        self.assertEqual(mapped.tolist(), [1.0, 0.0])

    def test_evening_and_early_bars_map_to_next_day_session(self) -> None:
        dt = pd.Series(
            pd.to_datetime(
                [
                    "2026-07-03 13:45",
                    "2026-07-03 15:00",
                    "2026-07-06 04:59",
                    "2026-07-06 08:45",
                ]
            )
        )
        got = audit.assign_trading_day(dt).dt.strftime("%Y-%m-%d").tolist()
        self.assertEqual(got, ["2026-07-03", "2026-07-06", "2026-07-06", "2026-07-06"])

    def test_monthly_stop_keeps_trigger_bar_loss(self) -> None:
        bars = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-01-02 09:00", "2026-01-02 09:01"]),
                "ema_position": [1.0, 1.0],
                "price_change": [-10.0, 0.0],
            }
        )
        exposure, stopped = monitor.apply_monthly_stop(
            bars, base=0.0, addon=1.0, threshold=-0.08, start_price=100.0
        )
        self.assertEqual(exposure.tolist(), [1.0, 0.0])
        self.assertEqual(stopped.tolist(), [True, True])

    def test_oos_does_not_affect_selection_score(self) -> None:
        frame = pd.DataFrame(
            {
                "strategy": ["same-train-bad-oos", "same-train-good-oos"],
                "train_sharpe": [1.0, 1.0],
                "train_maxdd": [-0.2, -0.2],
                "avg_exposure_full": [1.0, 1.0],
                "oos_sharpe": [-9.0, 9.0],
            }
        )
        ranked = optimization.add_rank_columns(frame)
        self.assertTrue((ranked["oos_used_for_selection"] == False).all())  # noqa: E712
        self.assertAlmostEqual(ranked["robust_score"].iloc[0], ranked["robust_score"].iloc[1])

    def test_partial_day_is_rejected_for_basis_close(self) -> None:
        bars = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2026-06-24 13:45", "2026-06-25 08:45", "2026-06-25 12:43"]
                ),
                "Close": [45000.0, 45100.0, 45200.0],
            }
        )
        close, _ = monitor.futures_daily_series(bars, Path("/definitely/missing.csv"))
        self.assertIn(pd.Timestamp("2026-06-24"), close.index)
        self.assertNotIn(pd.Timestamp("2026-06-25"), close.index)

    def test_paper_default_leverage_is_core_one_x(self) -> None:
        with mock.patch.object(sys, "argv", ["shioaji_paper_test.py"]):
            args = paper.parse_args()
        self.assertEqual(args.target_leverage, 1.0)


if __name__ == "__main__":
    unittest.main()
