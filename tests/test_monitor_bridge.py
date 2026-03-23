import importlib
import sys
import unittest

import pandas as pd

import adapters.dummy_MetaTrader5 as dummy_mt5

sys.modules.setdefault("MetaTrader5", dummy_mt5)
sys.modules.setdefault("mt5linux", dummy_mt5)
MT5Adapter = importlib.import_module("adapters.mt5_adapter").MT5Adapter
from backtest.monitor_bridge import (
    normalize_monitor_cycle_for_replay,
    normalize_monitor_cycles_for_replay,
)


class MonitorBridgeTests(unittest.TestCase):
    def setUp(self):
        dummy_mt5.reset_simulation()
        dummy_mt5.shutdown()

    def test_normalize_monitor_cycle_populates_replay_fields(self):
        raw_cycle = {
            "timestamp": "2026-01-01T00:00:00Z",
            "latest_actual_prices": {"EURUSD": 1.1000},
            "trade_signals": {
                "EURUSD": {
                    "signal": "HOLD",
                    "entry_price": 1.1000,
                    "reason": "Consensus/Kalman gate blocked BUY",
                }
            },
            "rls_health": {"fx_major": {"confidence": 0.83, "pred_var": 0.0004}},
            "parameter_deviations": {"fx_major": 0.03},
            "rls_forecast": {"EURUSD": {"rls_expected_return_pct": 1.4}},
            "dcc_metrics": {"fx_major": {"contagion_score": 0.62}},
            "kalman_metrics": {"EURUSD": {"innovation_zscore": 0.7, "trend": "UP"}},
            "consensus_metrics": {"EURUSD": {"signal_d1": "BUY", "signal_h1": "BUY", "kalman_trend": "UP"}},
            "news_status": {"is_restricted": False},
            "global_metrics": {"global_confidence": 0.83},
        }

        normalized = normalize_monitor_cycle_for_replay(raw_cycle, symbol_to_group={"EURUSD": "fx_major"})
        signal = normalized["trade_signals"]["EURUSD"]

        self.assertEqual(signal["preferred_action"], "BUY")
        self.assertEqual(signal["signal"], "HOLD")
        self.assertIn("consensus_gate", signal["blocked_by"])
        self.assertFalse(signal["action_mask"]["consensus_gate"])
        self.assertEqual(signal["feature_model"]["rls_confidence"], 0.83)
        self.assertEqual(signal["feature_model"]["deviation_score"], 0.03)
        self.assertEqual(signal["feature_model"]["kalman_zscore"], 0.7)
        self.assertEqual(signal["feature_model"]["dcc_correlation"], 0.62)
        self.assertAlmostEqual(signal["feature_model"]["predicted_return"], 0.014)
        self.assertEqual(signal["feature_model"]["pred_var"], 0.0004)
        self.assertEqual(signal["feature_model"]["regime_label"], "UP")

    def test_normalize_monitor_cycles_keeps_cycle_order(self):
        cycles = [
            {"timestamp": "2026-01-01T00:00:00Z", "latest_actual_prices": {}, "trade_signals": {}},
            {"timestamp": "2026-01-02T00:00:00Z", "latest_actual_prices": {}, "trade_signals": {}},
        ]
        normalized = normalize_monitor_cycles_for_replay(cycles)
        self.assertEqual([row["timestamp"] for row in normalized], [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
        ])

    def test_dummy_historical_playback_is_deterministic_via_adapter(self):
        hist_df = pd.DataFrame(
            {
                "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"],
                "bid": [1.1000, 1.1005],
                "ask": [1.1002, 1.1007],
                "open": [1.0998, 1.1001],
                "high": [1.1003, 1.1008],
                "low": [1.0997, 1.1000],
                "close": [1.1001, 1.1006],
                "tick_volume": [10, 11],
                "volume": [10, 11],
            }
        )
        dummy_mt5.inject_historical_data("EURUSD", hist_df)
        dummy_mt5.set_current_step(1)

        adapter = MT5Adapter(mt5_backend=dummy_mt5)
        self.assertTrue(adapter.initialize())
        self.assertTrue(adapter.login(login=123456, password="demo", server="dummy"))

        tick = adapter.symbol_info_tick("EURUSD")
        account_info = adapter.account_info()
        rates = adapter.copy_rates_range(
            "EURUSD",
            adapter.TIMEFRAME_M1,
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-01T00:01:00Z"),
        )

        self.assertAlmostEqual(tick.bid, 1.1005)
        self.assertAlmostEqual(tick.ask, 1.1007)
        self.assertEqual(account_info.equity, 10000.0)
        self.assertEqual(len(rates), 2)
        self.assertIn("close", rates.dtype.names)


if __name__ == "__main__":
    unittest.main()
