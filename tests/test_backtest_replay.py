import unittest

from backtest.replay import run_one_bar_replay_backtest


class BacktestReplayTests(unittest.TestCase):
    def test_one_bar_replay_counts_trade_and_pnl(self):
        cycles = [
            {
                "latest_actual_prices": {"EURUSD": 1.1000},
                "trade_signals": {"EURUSD": {"signal": "BUY", "entry_price": 1.1000}},
            },
            {
                "latest_actual_prices": {"EURUSD": 1.1011},
                "trade_signals": {},
            },
        ]

        summary = run_one_bar_replay_backtest(cycles, fee_bps=0.0, slippage_bps=0.0)

        self.assertEqual(summary.total_trades, 1)
        self.assertAlmostEqual(summary.win_rate, 1.0)
        self.assertGreater(summary.gross_pnl, 0.0)


if __name__ == "__main__":
    unittest.main()
