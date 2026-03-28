import unittest

import parameter


class ParameterCompatibilityTests(unittest.TestCase):
    def test_legacy_aliases_exist(self):
        self.assertEqual(parameter.maxlag_test, parameter.maxlag_granger)
        self.assertEqual(parameter.alpha, parameter.alpha_granger)
        self.assertIsInstance(parameter.lookback_days, int)
        self.assertEqual(parameter.base_interval, "1d")

    def test_pairs_symbol_alignment(self):
        self.assertIn("US500", parameter.PAIRS)
        self.assertNotIn("SP500", parameter.PAIRS)
        self.assertEqual(parameter.PAIRS["DXY"], "DXYm")
        self.assertEqual(parameter.PAIRS["GBPUSD"], "GBPUSDm")

    def test_asset_registry_derives_pairs_and_fred_series(self):
        self.assertIn("GBPUSD", parameter.ASSET_REGISTRY)
        self.assertEqual(parameter.ASSET_REGISTRY["XAUUSD"]["asset_class"], "commodities")
        self.assertEqual(parameter.FRED_SERIES["EFFRVOL"], "EFFRVOL")
        self.assertNotIn("EFFRVOL", parameter.PAIRS)
        self.assertEqual(parameter.BASE_DATA_DIR, "/content/drive/MyDrive/books/CASSANDRA/base_data")

    def test_imputation_special_assets_contains_btc_cross(self):
        names = {item["func_pair_name"] for item in parameter.IMPUTATION_SPECIAL_ASSETS}
        self.assertIn("BTC/USD", names)
        self.assertIn("BTC/XAU", names)
        self.assertIn("BTC/XAG", names)

    def test_pickle_cache_defaults_exist(self):
        self.assertTrue(parameter.PKL_CACHE_DIR)
        self.assertTrue(parameter.MTF_BASE_DFS_PKL_NAME.endswith('.pkl'))
        self.assertTrue(parameter.FRED_DF_PKL_NAME.endswith('.pkl'))


if __name__ == '__main__':
    unittest.main()
