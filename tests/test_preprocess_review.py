import unittest
from io import StringIO

import pandas as pd

import main


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


if __name__ == "__main__":
    unittest.main()
