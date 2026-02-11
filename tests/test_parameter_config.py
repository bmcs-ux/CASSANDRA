import unittest

import parameter


class ParameterCompatibilityTests(unittest.TestCase):
    def test_legacy_aliases_exist(self):
        self.assertEqual(parameter.maxlag_test, parameter.maxlag_granger)
        self.assertEqual(parameter.alpha, parameter.alpha_granger)
        self.assertEqual(parameter.lookback_days, parameter.fred_lookback_days)


if __name__ == '__main__':
    unittest.main()
