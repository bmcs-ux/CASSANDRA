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

    def test_fee_and_slippage_are_charged_on_both_sides(self):
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

        self.assertAlmostEqual(result.trade_ledger[0]["gross_return"], 0.001)
        self.assertAlmostEqual(result.trade_ledger[0]["net_return"], 0.0)
        self.assertAlmostEqual(result.summary.net_pnl, 0.0)

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
        self.assertAlmostEqual(result.trade_ledger[0]["entry_price"], 1.25)

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

    def test_multi_cycle_multi_symbol_builds_equity_curve_and_drawdown(self):
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


if __name__ == "__main__":
    unittest.main()
