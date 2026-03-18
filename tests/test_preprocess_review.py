import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import main

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency for local runs
    pl = None


class PreprocessReviewHelpersTests(unittest.TestCase):
    def test_summarize_dataframe_counts_missing(self):
        df = pd.DataFrame(
            {
                "A": [1.0, None, 3.0],
                "B": [None, None, 2.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        summary = main.summarize_dataframe(df, "demo")

        self.assertEqual(summary["shape"], (3, 2))
        self.assertEqual(summary["total_missing"], 3)
        self.assertEqual(summary["missing_per_column"]["B"], 2)

    def test_combine_and_split_mtf_pair_ohlc_roundtrip(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        mtf_pairs = {
            "XAUUSD": pd.DataFrame({"Open": [1, 2], "Close": [3, 4]}, index=idx),
            "BTCUSD": pd.DataFrame({"Open": [5, 6], "Close": [7, 8]}, index=idx),
        }

        combined = main._combine_mtf_pair_ohlc(mtf_pairs)
        rebuilt = main._split_mtf_pair_ohlc(combined, mtf_pairs)

        self.assertIn("XAUUSD_Open", combined.columns)
        self.assertIn("BTCUSD_Close", combined.columns)
        pd.testing.assert_frame_equal(rebuilt["XAUUSD"], mtf_pairs["XAUUSD"])
        pd.testing.assert_frame_equal(rebuilt["BTCUSD"], mtf_pairs["BTCUSD"])

    def test_review_functions_non_interactive(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        mtf = {
            "D1": {
                "XAUUSD": pd.DataFrame({"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3], "Close": [1, 2, 3]}, index=idx)
            }
        }
        fred_df = pd.DataFrame({"EFFRVOL": [1.0, None, 2.0]}, index=idx)

        stream = StringIO()
        reviewed_mtf = main.review_and_confirm_mtf_data(stream, mtf, fred_df, interactive=False)
        reviewed_fred = main.review_and_confirm_fred_data(stream, fred_df, mtf, interactive=False)

        self.assertEqual(set(reviewed_mtf.keys()), {"D1"})
        self.assertEqual(reviewed_fred.shape, fred_df.shape)

    def test_input_menu_uses_first_character(self):
        with patch("builtins.input", return_value="kextra"):
            choice = main._input_menu("prompt", {"k", "b"}, "b")
        self.assertEqual(choice, "k")

    def test_fred_menu_back_cancels_imputation(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        fred_df = pd.DataFrame({"BTC/USD_Close": [1.0, None, 3.0]}, index=idx)
        mtf = {"D1": {"XAUUSD": pd.DataFrame({"Open": [1, 2, 3], "Close": [1, 2, 3]}, index=idx)}}

        def fake_imputation(log_stream, df):
            return df.fillna(99.0), {"dummy": 1}

        with patch("main.apply_loop_berantai_imputation", side_effect=fake_imputation), patch(
            "builtins.input", side_effect=["i", "1", "b", "k"]
        ):
            reviewed = main.review_and_confirm_fred_data(StringIO(), fred_df, mtf, interactive=True)

        pd.testing.assert_frame_equal(reviewed, fred_df)


    @unittest.skipIf(pl is None, "polars is required for parquet persistence tests")
    def test_mtf_menu_save_writes_parquet(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        mtf = {"D1": {"XAUUSD": pd.DataFrame({"Open": [1, 2], "Close": [1, 2]}, index=idx)}}
        fred_df = pd.DataFrame({"EFFRVOL": [1.0, 2.0]}, index=idx)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(main.parameter, "BASE_DATA_DIR", tmpdir), patch("builtins.input", side_effect=["s", "k"]):
            reviewed = main.review_and_confirm_mtf_data(StringIO(), mtf, fred_df, interactive=True)

            saved_files = list(Path(tmpdir).glob("asset_class=*/symbol=*/timeframe=*/*.parquet"))

        self.assertIn("D1", reviewed)
        self.assertEqual(len(saved_files), 1)

    def test_mtf_imputation_uses_special_assets_only_for_imputation(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        mtf = {"D1": {"XAUUSD": pd.DataFrame({"Open": [1, 2, 3], "Close": [1, 2, 3]}, index=idx)}}
        fred_df = pd.DataFrame({"EFFRVOL": [1.0, 2.0, 3.0]}, index=idx)
        special = {
            "D1": {
                "BTC/USD": pd.DataFrame({"Open": [10, 11, 12], "High": [10, 11, 12], "Low": [10, 11, 12], "Close": [10, 11, 12]}, index=idx),
                "BTC/XAU": pd.DataFrame({"Open": [5, 6, 7], "High": [5, 6, 7], "Low": [5, 6, 7], "Close": [5, 6, 7]}, index=idx),
                "BTC/XAG": pd.DataFrame({"Open": [2, 3, 4], "High": [2, 3, 4], "Low": [2, 3, 4], "Close": [2, 3, 4]}, index=idx),
                "XAU/USD": pd.DataFrame({"Open": [1, 1, 1], "High": [1, 1, 1], "Low": [1, 1, 1], "Close": [1, 1, 1]}, index=idx),
                "XAG/USD": pd.DataFrame({"Open": [1, 1, 1], "High": [1, 1, 1], "Low": [1, 1, 1], "Close": [1, 1, 1]}, index=idx),
            }
        }

        def fake_imputation(log_stream, df):
            self.assertIn("BTC/USD_Close", df.columns)
            return df, {"ok": True}

        with patch("main.apply_loop_berantai_imputation", side_effect=fake_imputation), patch(
            "builtins.input", side_effect=["i", "k"]
        ):
            reviewed = main.review_and_confirm_mtf_data(
                StringIO(),
                mtf,
                fred_df,
                interactive=True,
                imputation_assets_by_tf=special,
            )

        self.assertNotIn("BTC/USD", reviewed["D1"])

    @unittest.skipIf(pl is None, "polars is required for parquet persistence tests")
    def test_save_and_load_parquet_roundtrip_preserves_timeframes(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        mtf = {
            "D1": {
                "XAUUSD": pd.DataFrame({"Open": [1.0, 2.0, 3.0], "Close": [1.5, 2.5, 3.5]}, index=idx),
            },
            "H1": {
                "GBPUSD": pd.DataFrame({"Open": [4.0, 5.0, 6.0], "Close": [4.5, 5.5, 6.5]}, index=idx),
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            stream = StringIO()
            main._save_parquet(stream, mtf, tmpdir, "mtf_base_dfs", main.parameter.ASSET_REGISTRY)
            loaded = main._load_parquet_lazy(tmpdir, main.parameter.ASSET_REGISTRY)

        self.assertEqual(set(loaded.keys()), {"D1", "H1"})
        self.assertIn("XAUUSD", loaded["D1"])
        self.assertIn("GBPUSD", loaded["H1"])
        pd.testing.assert_frame_equal(loaded["D1"]["XAUUSD"], mtf["D1"]["XAUUSD"])
        pd.testing.assert_frame_equal(loaded["H1"]["GBPUSD"], mtf["H1"]["GBPUSD"])

    def test_fred_menu_save_writes_pickle(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        fred_df = pd.DataFrame({"EFFRVOL": [1.0, 2.0]}, index=idx)
        mtf = {"D1": {"XAUUSD": pd.DataFrame({"Open": [1, 2], "Close": [1, 2]}, index=idx)}}

        with patch("main._save_pickle") as save_mock, patch("builtins.input", side_effect=["s", "k"]):
            reviewed = main.review_and_confirm_fred_data(StringIO(), fred_df, mtf, interactive=True)

        self.assertEqual(reviewed.shape, fred_df.shape)
        save_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
