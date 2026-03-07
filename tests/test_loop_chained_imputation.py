import unittest
from io import StringIO

import numpy as np
import pandas as pd

from preprocessing.handle_missing import handle_missing_market_data
from preprocessing.loop_chained_imputation import (
    impute_btc_cross_pairs,
    impute_metals_from_btc_cross,
)


class LoopChainedImputationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                'BTC/USD_Close': [60000.0, 61000.0],
                'XAU/USD_Close': [2000.0, np.nan],
                'XAG/USD_Close': [25.0, np.nan],
                'BTC/XAU_Close': [np.nan, 30.0],
                'BTC/XAG_Close': [2400.0, 2440.0],
                'BTC/XAU_Open': [np.nan, np.nan],
                'BTC/XAU_High': [np.nan, np.nan],
                'BTC/XAU_Low': [np.nan, np.nan],
                'BTC/XAG_Open': [np.nan, np.nan],
                'BTC/XAG_High': [np.nan, np.nan],
                'BTC/XAG_Low': [np.nan, np.nan],
                'XAU/USD_Open': [np.nan, np.nan],
                'XAU/USD_High': [np.nan, np.nan],
                'XAU/USD_Low': [np.nan, np.nan],
                'XAG/USD_Open': [np.nan, np.nan],
                'XAG/USD_High': [np.nan, np.nan],
                'XAG/USD_Low': [np.nan, np.nan],
            }
        )

    def test_impute_btc_cross_pairs(self):
        result, stats = impute_btc_cross_pairs(self.df)

        self.assertAlmostEqual(result.loc[0, 'BTC/XAU_Close'], 30.0)
        self.assertAlmostEqual(result.loc[1, 'BTC/XAG_Close'], 2440.0)
        self.assertAlmostEqual(result.loc[0, 'BTC/XAU_Open'], 30.0)
        self.assertEqual(stats['BTC/XAU']['added'], 1)
        self.assertEqual(stats['BTC/XAG']['added'], 0)

    def test_impute_metals_from_btc_cross(self):
        btc_filled, _ = impute_btc_cross_pairs(self.df)
        result, stats = impute_metals_from_btc_cross(btc_filled)

        self.assertAlmostEqual(result.loc[1, 'XAU/USD_Close'], 2033.3333333333333)
        self.assertAlmostEqual(result.loc[1, 'XAG/USD_Close'], 25.0)
        self.assertAlmostEqual(result.loc[1, 'XAU/USD_High'], result.loc[1, 'XAU/USD_Close'])
        self.assertEqual(stats['XAU/USD']['added'], 1)
        self.assertEqual(stats['XAG/USD']['added'], 1)

    def test_handle_missing_market_data_returns_original_when_required_cols_missing(self):
        stream = StringIO()
        bad_df = pd.DataFrame({'BTC/USD_Close': [1.0]})

        result = handle_missing_market_data(stream, bad_df)

        self.assertTrue(result.equals(bad_df))
        self.assertIn('Imputasi Loop Berantai dilewati', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
