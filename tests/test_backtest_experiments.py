import unittest

from backtest.experiments import (
    build_experiment_grid,
    derive_regime_label,
    evaluate_strategy_config,
    evaluate_walk_forward_grid,
    generate_walk_forward_windows,
    run_replay_experiment_grid,
)
from backtest.replay import build_replay_ledgers


class BacktestExperimentTests(unittest.TestCase):
    def _build_cycle_results(self):
        return [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 100.0},
                "trade_signals": {
                    "EURUSD": {
                        "signal": "HOLD",
                        "preferred_action": "BUY",
                        "entry_price": 100.0,
                        "feature_model": {
                            "rls_confidence": 0.92,
                            "deviation_score": 0.02,
                            "kalman_zscore": 0.5,
                            "dcc_correlation": 0.75,
                            "predicted_return": 0.020,
                            "pred_var": 0.0004,
                            "spread": 0.0002,
                            "regime_label": "trend_up_high_corr",
                        },
                    }
                },
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 102.0},
                "trade_signals": {
                    "EURUSD": {
                        "signal": "HOLD",
                        "preferred_action": "BUY",
                        "entry_price": 102.0,
                        "feature_model": {
                            "rls_confidence": 0.68,
                            "deviation_score": 0.03,
                            "kalman_zscore": 0.7,
                            "dcc_correlation": 0.55,
                            "predicted_return": 0.015,
                            "pred_var": 0.0009,
                            "spread": 0.0003,
                            "regime_label": "trend_up_low_corr",
                        },
                    }
                },
            },
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 103.0},
                "trade_signals": {
                    "EURUSD": {
                        "signal": "HOLD",
                        "preferred_action": "SELL",
                        "entry_price": 103.0,
                        "feature_model": {
                            "rls_confidence": 0.84,
                            "deviation_score": 0.14,
                            "kalman_zscore": 0.6,
                            "dcc_correlation": 0.62,
                            "predicted_return": -0.018,
                            "pred_var": 0.0004,
                            "spread": 0.0004,
                            "regime_label": "ranging_high_vol",
                        },
                    }
                },
            },
            {
                "timestamp": "2026-01-04T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 101.0},
                "trade_signals": {
                    "EURUSD": {
                        "signal": "HOLD",
                        "preferred_action": "BUY",
                        "entry_price": 101.0,
                        "feature_model": {
                            "rls_confidence": 0.42,
                            "deviation_score": 0.02,
                            "kalman_zscore": 0.5,
                            "dcc_correlation": 0.40,
                            "predicted_return": 0.012,
                            "pred_var": 0.0004,
                            "spread": 0.0002,
                            "regime_label": "trend_up_low_conf",
                        },
                    }
                },
            },
            {
                "timestamp": "2026-01-05T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 104.0},
                "trade_signals": {
                    "EURUSD": {
                        "signal": "HOLD",
                        "preferred_action": "SELL",
                        "entry_price": 104.0,
                        "feature_model": {
                            "rls_confidence": 0.88,
                            "deviation_score": 0.04,
                            "kalman_zscore": 1.8,
                            "dcc_correlation": 0.35,
                            "predicted_return": -0.012,
                            "pred_var": 0.0004,
                            "spread": 0.0012,
                            "regime_label": "trend_down_wide_spread",
                        },
                    }
                },
            },
            {
                "timestamp": "2026-01-06T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 106.0},
                "trade_signals": {
                    "EURUSD": {
                        "signal": "HOLD",
                        "preferred_action": "BUY",
                        "entry_price": 106.0,
                        "feature_model": {
                            "rls_confidence": 0.95,
                            "deviation_score": 0.01,
                            "kalman_zscore": 0.4,
                            "dcc_correlation": 0.82,
                            "predicted_return": 0.022,
                            "pred_var": 0.0004,
                            "spread": 0.0002,
                            "regime_label": "trend_up_high_corr",
                        },
                    }
                },
            },
            {
                "timestamp": "2026-01-07T00:00:00Z",
                "latest_actual_prices": {"EURUSD": 108.0},
                "trade_signals": {},
            },
        ]

    def test_build_experiment_grid_enumerates_parameter_combinations(self):
        grid = build_experiment_grid(
            {
                "rls_confidence_entry_threshold": [0.5, 0.8],
                "rls_deviation_threshold": [0.05],
                "kalman_flip_zscore": [1.0, 2.0],
            }
        )

        self.assertEqual(len(grid), 4)
        self.assertEqual(grid[0].name, "cfg_001")
        self.assertEqual(grid[-1].kalman_flip_zscore, 2.0)

    def test_evaluate_strategy_config_applies_thresholds_and_regime_segmentation(self):
        replay_result = build_replay_ledgers(self._build_cycle_results())
        config = build_experiment_grid(
            {
                "rls_confidence_entry_threshold": [0.6],
                "rls_deviation_threshold": [0.10],
                "rls_deviation_close_all_threshold": [0.20],
                "consensus_threshold": [0.5],
                "kalman_flip_zscore": [1.0],
            }
        )[0]

        summary = evaluate_strategy_config(replay_result.decision_ledger, config)

        self.assertEqual(summary["evaluated_decisions"], 6)
        self.assertEqual(summary["executed_trades"], 3)
        self.assertIn("confidence_gate", summary["blocked_by_summary"])
        self.assertIn("deviation_gate", summary["blocked_by_summary"])
        self.assertIn("kalman_flip_gate", summary["blocked_by_summary"])
        self.assertIn("trend_up_high_corr", summary["regime_summary"])
        self.assertGreater(summary["net_return"], 0.0)

    def test_generate_walk_forward_windows_preserves_temporal_order(self):
        replay_result = build_replay_ledgers(self._build_cycle_results())

        windows = generate_walk_forward_windows(replay_result.decision_ledger, train_size=3, test_size=2, step_size=1)

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].train_start, 0)
        self.assertEqual(windows[0].test_start, 3)
        self.assertLess(windows[0].train_end_timestamp, windows[0].test_start_timestamp)

    def test_evaluate_walk_forward_grid_returns_ranking_and_stability_metrics(self):
        replay_result = build_replay_ledgers(self._build_cycle_results())
        configs = build_experiment_grid(
            {
                "rls_confidence_entry_threshold": [0.4, 0.8],
                "rls_deviation_threshold": [0.10],
                "rls_deviation_close_all_threshold": [0.20],
                "consensus_threshold": [0.5],
                "kalman_flip_zscore": [1.0, 2.0],
            }
        )

        report = evaluate_walk_forward_grid(
            replay_result.decision_ledger,
            configs,
            train_size=3,
            test_size=2,
            step_size=1,
        )

        self.assertEqual(report["window_count"], 2)
        self.assertEqual(len(report["configs"]), 4)
        self.assertEqual(len(report["ranking"]), 4)
        self.assertGreaterEqual(report["ranking"][0]["ranking_score"], report["ranking"][-1]["ranking_score"])
        self.assertIn("stability_variance", report["configs"][0]["aggregate"])

    def test_run_replay_experiment_grid_runs_end_to_end(self):
        report = run_replay_experiment_grid(
            self._build_cycle_results(),
            {
                "rls_confidence_entry_threshold": [0.4],
                "rls_deviation_threshold": [0.10],
                "rls_deviation_close_all_threshold": [0.20],
                "consensus_threshold": [0.5],
                "kalman_flip_zscore": [1.5],
            },
            train_size=3,
            test_size=2,
            step_size=1,
        )

        self.assertIn("replay_metadata", report)
        self.assertEqual(report["window_count"], 2)
        self.assertEqual(len(report["ranking"]), 1)

    def test_regime_label_falls_back_to_heuristic_segmentation(self):
        label = derive_regime_label(
            {
                "predicted_return": -0.01,
                "pred_var": 0.02,
                "dcc_correlation": 0.2,
                "spread": 0.002,
            }
        )

        self.assertEqual(label, "high_vol:trend_down:low_corr:wide_spread")


if __name__ == "__main__":
    unittest.main()
