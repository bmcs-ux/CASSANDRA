import unittest
from io import StringIO

import pandas as pd

from preprocessing.combine_data import combine_log_returns


class CombineLogReturnsTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC')
        self.log_returns = {
            'EURUSD': pd.DataFrame(
                {
                    'Close_Log_Return': [0.1, 0.2, 0.3],
                    'Open_Log_Return': [0.09, 0.19, 0.29],
                    'Noise': [1, 2, 3],
                },
                index=idx,
            ),
            'USDJPY': pd.DataFrame(
                {
                    'Close_Log_Return': [0.01, 0.02, 0.03],
                },
                index=idx,
            ),
        }

    def test_return_type_dict_adds_log_return_alias(self):
        stream = StringIO()

        result = combine_log_returns(stream, self.log_returns, return_type='dict')

        self.assertIn('EURUSD', result)
        self.assertIn('USDJPY', result)
        self.assertIn('Log_Return', result['EURUSD'].columns)
        self.assertEqual(result['EURUSD']['Log_Return'].tolist(), [0.1, 0.2, 0.3])
        self.assertNotIn('Noise', result['EURUSD'].columns)

    def test_return_type_df_prefixes_pair_name(self):
        stream = StringIO()

        result = combine_log_returns(stream, self.log_returns, return_type='df')

        self.assertIn('EURUSD_Close_Log_Return', result.columns)
        self.assertIn('USDJPY_Close_Log_Return', result.columns)
        self.assertNotIn('Noise', result.columns)


if __name__ == '__main__':
    unittest.main()
