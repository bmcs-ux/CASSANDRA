import tempfile
import unittest
from pathlib import Path

import pandas as pd
import polars as pl

import adapters.dummy_MetaTrader5 as dummy_mt5


class DummyMT5PreloadTests(unittest.TestCase):
    def setUp(self):
        dummy_mt5._GLOBAL_DATA_CACHE = {}

    def test_preload_falls_back_to_directory_discovery_when_registry_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = (
                Path(tmp_dir)
                / "asset_class=forex"
                / "symbol=GBPUSDm"
                / "timeframe=M1"
                / "data_2026_03.parquet"
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "__index__": ["2026-03-01T00:00:00Z", "2026-03-01T00:01:00Z"],
                    "open": [1.20, 1.21],
                    "high": [1.21, 1.22],
                    "low": [1.19, 1.20],
                    "close": [1.205, 1.215],
                    "tick_volume": [11, 12],
                }
            ).write_parquet(parquet_path)

            # Registry sengaja tidak cocok agar memicu fallback scan direktori.
            bad_registry = {"GBPUSD": {"asset_class": "forex", "symbol": "GBPUSD"}}
            dummy_mt5.preload_all_data(tmp_dir, bad_registry)

            self.assertIn("M1", dummy_mt5._GLOBAL_DATA_CACHE)
            self.assertIn("GBPUSDM", dummy_mt5._GLOBAL_DATA_CACHE["M1"])

    def test_copy_rates_range_supports_broker_suffix_symbol_lookup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = (
                Path(tmp_dir)
                / "asset_class=forex"
                / "symbol=GBPUSD"
                / "timeframe=M1"
                / "data_2026_03.parquet"
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "__index__": ["2026-03-01T00:00:00Z", "2026-03-01T00:01:00Z"],
                    "open": [1.30, 1.31],
                    "high": [1.31, 1.32],
                    "low": [1.29, 1.30],
                    "close": [1.305, 1.315],
                    "tick_volume": [13, 14],
                }
            ).write_parquet(parquet_path)

            dummy_mt5.preload_all_data(tmp_dir, {})
            rates = dummy_mt5.copy_rates_range(
                "GBPUSDm",
                dummy_mt5.TIMEFRAME_M1,
                "2026-03-01T00:00:00Z",
                "2026-03-01T00:01:00Z",
            )

            self.assertEqual(len(rates), 2)
            self.assertIn("close", rates.dtype.names)

    def test_copy_rates_range_handles_epoch_seconds_in_time_column(self):
        epoch_start = int(pd.Timestamp("2026-03-01T00:00:00Z").timestamp())
        dummy_mt5._historical_buffer["GBPUSD"] = pd.DataFrame(
            {
                "time": [epoch_start, epoch_start + 60],
                "open": [1.30, 1.31],
                "high": [1.31, 1.32],
                "low": [1.29, 1.30],
                "close": [1.305, 1.315],
                "tick_volume": [10, 12],
            }
        )

        rates = dummy_mt5.copy_rates_range(
            "GBPUSD",
            dummy_mt5.TIMEFRAME_M1,
            "2026-03-01T00:00:00Z",
            "2026-03-01T00:01:00Z",
        )

        self.assertEqual(len(rates), 2)
        self.assertGreater(rates["time"][-1], 1700000000)

    def test_copy_rates_range_handles_epoch_milliseconds_in_time_column(self):
        epoch_start_ms = int(pd.Timestamp("2026-03-01T00:00:00Z").timestamp() * 1000)
        dummy_mt5._historical_buffer["USDJPY"] = pd.DataFrame(
            {
                "time": [epoch_start_ms, epoch_start_ms + 60_000],
                "open": [150.0, 150.1],
                "high": [150.2, 150.3],
                "low": [149.9, 150.0],
                "close": [150.1, 150.2],
                "tick_volume": [20, 22],
            }
        )

        rates = dummy_mt5.copy_rates_range(
            "USDJPY",
            dummy_mt5.TIMEFRAME_M1,
            "2026-03-01T00:00:00Z",
            "2026-03-01T00:01:00Z",
        )

        self.assertEqual(len(rates), 2)
        self.assertGreater(rates["time"][-1], 1700000000)


if __name__ == "__main__":
    unittest.main()
