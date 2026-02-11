import unittest

import parameter


class ParameterCompatibilityTests(unittest.TestCase):
    def test_legacy_aliases_exist(self):
        self.assertEqual(parameter.maxlag_test, parameter.maxlag_granger)
        self.assertEqual(parameter.alpha, parameter.alpha_granger)
        self.assertIsInstance(parameter.lookback_days, int)
        self.assertEqual(parameter.base_interval, "1d")


if __name__ == '__main__':
    unittest.main()
