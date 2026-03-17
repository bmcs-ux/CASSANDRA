import unittest
from io import StringIO

import pandas as pd

from restored import restore_log_returns_to_price


class RestoreLogReturnsToPriceTests(unittest.TestCase):
    def setUp(self):
        self.idx = pd.date_range('2026-03-04', periods=2, freq='D', tz='UTC')
        self.base_dfs = {
            'GBPUSD': pd.DataFrame(
                {
                    'Open': [1.24, 1.25],
                    'High': [1.26, 1.27],
                    'Low': [1.23, 1.24],
                    'Close': [1.255, 1.256],
                },
                index=pd.date_range('2026-03-01', periods=2, freq='D', tz='UTC'),
            )
        }

    def test_restore_accepts_summary_frame_columns(self):
        forecast_data = {
            'FX_Majors': {
                'endog_names': ['GBPUSD_Close_Log_Return'],
                'interval_forecast': pd.DataFrame(
                    {
                        'mean': [0.01, 0.02],
                        'mean_ci_lower': [0.005, 0.01],
                        'mean_ci_upper': [0.015, 0.03],
                    },
                    index=self.idx,
                ),
            }
        }

        stream = StringIO()
        restored = restore_log_returns_to_price(stream, forecast_data, self.base_dfs)

        self.assertIn('GBPUSD', restored)
        self.assertFalse(restored['GBPUSD'].empty)
        self.assertIn('Close_Mean_Forecast', restored['GBPUSD'].columns)
        self.assertIn('Close_Lower_95CI', restored['GBPUSD'].columns)
        self.assertIn('Close_Upper_95CI', restored['GBPUSD'].columns)

    def test_restore_accepts_custom_renamed_columns(self):
        forecast_data = {
            'FX_Majors': {
                'endog_names': ['GBPUSD_Close_Log_Return'],
                'interval_forecast': pd.DataFrame(
                    {
                        'GBPUSD_Close_Log_Return_Forecast': [0.01, 0.02],
                        'GBPUSD_Close_Log_Return_Lower': [0.005, 0.01],
                        'GBPUSD_Close_Log_Return_Upper': [0.015, 0.03],
                    },
                    index=self.idx,
                ),
            }
        }

        stream = StringIO()
        restored = restore_log_returns_to_price(stream, forecast_data, self.base_dfs)

        self.assertIn('GBPUSD', restored)
        self.assertFalse(restored['GBPUSD'].empty)


if __name__ == '__main__':
    unittest.main()
