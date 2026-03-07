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

    def test_imputation_special_assets_contains_btc_cross(self):
        names = {item["func_pair_name"] for item in parameter.IMPUTATION_SPECIAL_ASSETS}
        self.assertIn("BTC/USD", names)
        self.assertIn("BTC/XAU", names)
        self.assertIn("BTC/XAG", names)


if __name__ == '__main__':
    unittest.main()
