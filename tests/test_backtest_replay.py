import tempfile
import unittest
from unittest.mock import patch

from backtest.replay import build_replay_ledgers, export_replay_ledgers_to_parquet, run_one_bar_replay_backtest


class BacktestReplayTests(unittest.TestCase):
    def test_one_bar_replay_counts_trade_and_pnl(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.1000},
                "trade_signals": {"EURUSD": {"signal": "BUY", "entry_price": 1.1000}},
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.1011},
                "trade_signals": {},
            },
        ]

        summary = run_one_bar_replay_backtest(cycles, fee_bps=0.0, slippage_bps=0.0)

        self.assertEqual(summary.total_trades, 1)
        self.assertAlmostEqual(summary.win_rate, 1.0)
        self.assertGreater(summary.gross_pnl, 0.0)
        self.assertEqual(summary.skipped_trades, 0)
        self.assertEqual(summary.equity_curve_mode, "additive")

    def test_sell_signal_can_profit_when_price_drops(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"XAUUSD": 3000.0},
                "trade_signals": {"XAUUSD": {"signal": "SELL", "entry_price": 3000.0}},
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"XAUUSD": 2970.0},
                "trade_signals": {},
            },
        ]

        result = build_replay_ledgers(cycles)

        self.assertEqual(result.summary.total_trades, 1)
        self.assertAlmostEqual(result.trade_ledger[0]["gross_return"], 0.01)
        self.assertAlmostEqual(result.summary.win_rate, 1.0)
        self.assertEqual(result.trade_ledger[0]["decision_id"], result.decision_ledger[0]["decision_id"])

    def test_fee_and_slippage_are_exposed_as_explicit_cost_return(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.0},
                "trade_signals": {"EURUSD": {"signal": "BUY", "entry_price": 1.0}},
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.0010},
                "trade_signals": {},
            },
        ]

        result = build_replay_ledgers(cycles, fee_bps=2.0, slippage_bps=3.0)

        trade = result.trade_ledger[0]
        self.assertAlmostEqual(trade["gross_return"], 0.001)
        self.assertGreater(trade["cost_return"], 0.0)
        self.assertAlmostEqual(trade["net_return"], trade["gross_return"] - trade["cost_return"])
        self.assertAlmostEqual(result.summary.net_pnl, trade["net_return"])

    def test_missing_entry_price_falls_back_to_current_actual_price(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"GBPUSD": 1.2500},
                "trade_signals": {"GBPUSD": {"signal": "BUY"}},
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"GBPUSD": 1.2625},
                "trade_signals": {},
            },
        ]

        result = build_replay_ledgers(cycles)

        self.assertTrue(result.trade_ledger[0]["used_entry_fallback"])
        self.assertEqual(result.trade_ledger[0]["entry_price_source"], "latest_actual_prices")
        self.assertAlmostEqual(result.trade_ledger[0]["entry_price_raw"], 1.25)
        self.assertAlmostEqual(result.decision_ledger[0]["entry_price_raw"], 1.25)

    def test_missing_exit_price_is_skipped_and_recorded_in_decision_ledger(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"USDJPY": 150.0},
                "trade_signals": {"USDJPY": {"signal": "BUY", "entry_price": 150.0}},
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {},
                "trade_signals": {},
            },
        ]

        result = build_replay_ledgers(cycles)

        self.assertEqual(result.summary.total_trades, 0)
        self.assertEqual(result.summary.skipped_trades, 1)
        self.assertEqual(result.decision_ledger[0]["skip_reason"], "missing_exit_price")
        self.assertFalse(result.decision_ledger[0]["actually_executed"])
        self.assertEqual(result.decision_ledger[0]["blocked_by"], ["exit_price_available"])
        self.assertEqual(result.decision_ledger[0]["gate_pass_mask"], [1, 1, 0, 1])

    def test_multi_cycle_multi_symbol_builds_additive_equity_curve_and_drawdown(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.0, "GBPUSD": 2.0},
                "trade_signals": {
                    "EURUSD": {"signal": "BUY", "entry_price": 1.0},
                    "GBPUSD": {"signal": "SELL", "entry_price": 2.0},
                },
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.1, "GBPUSD": 2.1},
                "trade_signals": {
                    "EURUSD": {"signal": "BUY", "entry_price": 1.1},
                    "GBPUSD": {"signal": "HOLD"},
                },
            },
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.045, "GBPUSD": 2.05},
                "trade_signals": {},
            },
        ]

        result = build_replay_ledgers(cycles)

        self.assertEqual(result.summary.total_trades, 3)
        self.assertEqual(len(result.summary.equity_curve), 3)
        self.assertGreater(result.summary.max_drawdown, 0.0)
        self.assertEqual(result.decision_ledger[-1]["action"], "HOLD")
        self.assertEqual(result.decision_ledger[-1]["skip_reason"], "hold_signal")
        self.assertEqual(result.summary.equity_curve_mode, "additive")

    def test_compounding_equity_curve_mode_is_supported(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.0},
                "trade_signals": {"EURUSD": {"signal": "BUY", "entry_price": 1.0}},
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.1},
                "trade_signals": {"EURUSD": {"signal": "BUY", "entry_price": 1.1}},
            },
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.045},
                "trade_signals": {},
            },
        ]

        additive = build_replay_ledgers(cycles, equity_curve_mode="additive")
        compounding = build_replay_ledgers(cycles, equity_curve_mode="compounding")

        self.assertEqual(additive.summary.equity_curve_mode, "additive")
        self.assertEqual(compounding.summary.equity_curve_mode, "compounding")
        self.assertNotEqual(additive.summary.equity_curve, compounding.summary.equity_curve)
        self.assertGreaterEqual(compounding.summary.equity_curve[0], 1.0)

    def test_effective_execution_prices_make_buy_less_optimistic(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.2000},
                "trade_signals": {"EURUSD": {"signal": "BUY", "entry_price": 1.2000}},
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.2120},
                "trade_signals": {},
            },
        ]

        result = build_replay_ledgers(cycles, fee_bps=1.0, slippage_bps=2.0)
        trade = result.trade_ledger[0]

        self.assertGreater(trade["entry_price_effective"], trade["entry_price_raw"])
        self.assertLess(trade["exit_price_effective"], trade["exit_price_raw"])
        self.assertGreater(trade["cost_return"], 0.0)
        self.assertLess(trade["net_return"], trade["gross_return"])

    def test_invalid_equity_curve_mode_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "equity_curve_mode"):
            build_replay_ledgers([], equity_curve_mode="invalid")

    def test_export_reports_missing_parquet_engine_cleanly(self):
        result = build_replay_ledgers(
            [
                {"timestamp": "2026-01-01T00:00:00Z", "latest_actual_prices": {"EURUSD": 1.0}, "trade_signals": {"EURUSD": {"signal": "BUY", "entry_price": 1.0}}},
                {"timestamp": "2026-01-02T00:00:00Z", "latest_actual_prices": {"EURUSD": 1.1}, "trade_signals": {}},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch("backtest.replay.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Parquet export"):
                export_replay_ledgers_to_parquet(result, tmpdir, run_metadata={"run_id": "demo"})

    def test_sprint2_feature_fields_multi_horizon_labels_and_gate_attribution(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.2000},
                "trade_signals": {
                    "EURUSD": {
                        "signal": "HOLD",
                        "preferred_action": "BUY",
                        "entry_price": 1.2000,
                        "blocked_by": ["confidence_gate"],
                        "action_mask": {"can_buy": False, "can_sell": True, "confidence_gate": False},
                        "feature_model": {
                            "rls_confidence": 0.41,
                            "deviation_score": 0.08,
                            "kalman_zscore": -1.2,
                            "dcc_correlation": 0.67,
                            "predicted_return": 0.015,
                            "pred_var": 0.0025,
                            "spread": 0.0002,
                            "regime_label": "trend_up",
                        },
                    }
                },
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.2060},
                "trade_signals": {},
            },
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.1940},
                "trade_signals": {},
            },
            {
                "timestamp": "2026-01-04T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.2180},
                "trade_signals": {},
            },
            {
                "timestamp": "2026-01-05T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.2100},
                "trade_signals": {},
            },
            {
                "timestamp": "2026-01-06T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 1.2240},
                "trade_signals": {},
            },
        ]

        result = build_replay_ledgers(cycles)
        decision = result.decision_ledger[0]

        self.assertEqual(result.summary.total_trades, 0)
        self.assertEqual(result.summary.skipped_trades, 1)
        self.assertEqual(decision["preferred_action"], "BUY")
        self.assertEqual(decision["action"], "HOLD")
        self.assertEqual(decision["skip_reason"], "blocked_by_gate")
        self.assertIn("confidence_gate", decision["blocked_by"])
        self.assertEqual(decision["rls_confidence"], 0.41)
        self.assertEqual(decision["regime_label"], "trend_up")
        self.assertFalse(decision["can_buy"])
        self.assertTrue(decision["can_sell"])
        self.assertIsNotNone(decision["pnl_1"])
        self.assertIsNotNone(decision["pnl_3"])
        self.assertIsNotNone(decision["pnl_5"])
        self.assertGreater(decision["max_favorable"], decision["max_adverse"])
        self.assertTrue(decision["hit_1"])
        self.assertTrue(decision["hit_3"])
        self.assertTrue(decision["hit_5"])
        self.assertIn("confidence_gate", result.metadata["gate_fields"])
        self.assertEqual(result.metadata["hold_reason_summary"]["blocked_by_gate"], 1)
        self.assertEqual(result.metadata["blocked_by_summary"]["confidence_gate"], 1)
        self.assertEqual(result.gate_attribution[0]["gate"], "confidence_gate")
        self.assertEqual(result.gate_attribution[0]["unblocked_trade_count"], 1)
        self.assertGreater(result.gate_attribution[0]["mean_impact"], 0.0)

    def test_custom_horizons_and_gate_masks_stay_consistent(self):
        cycles = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"XAUUSD": 3000.0},
                "trade_signals": {
                    "XAUUSD": {
                        "signal": "SELL",
                        "entry_price": 3000.0,
                        "gates": {"deviation_gate": True, "news_gate": True},
                    }
                },
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"XAUUSD": 2990.0},
                "trade_signals": {},
            },
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "latest_actual_prices": {"XAUUSD": 2970.0},
                "trade_signals": {},
            },
        ]

        result = build_replay_ledgers(cycles, horizons=(1, 2))
        decision = result.decision_ledger[0]

        self.assertIn("deviation_gate", decision["gate_fields"])
        self.assertIn("news_gate", decision["gate_fields"])
        self.assertEqual(len(decision["gate_fields"]), len(decision["gate_pass_mask"]))
        self.assertIsNotNone(decision["pnl_1"])
        self.assertIsNone(decision["pnl_3"])
        self.assertEqual(result.metadata["horizons"], [1, 2])


if __name__ == "__main__":
    unittest.main()
