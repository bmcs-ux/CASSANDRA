import tempfile
import unittest
from io import StringIO
from pathlib import Path

import pandas as pd

from raw.pair_raw import _resolve_local_csv_path, load_base_data_mtf


class PairRawLocalCsvTests(unittest.TestCase):
    def test_resolve_local_csv_path_prefers_interval_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            target = d / 'combined_data_final_complete_m1.csv'
            other = d / 'combined_data_final_complete_h1.csv'
            target.write_text('idx,EURUSD_Open\n')
            other.write_text('idx,EURUSD_Open\n')

            resolved = _resolve_local_csv_path(str(d / 'combined_data_final_complete.csv'), '1m')

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.name, 'combined_data_final_complete_m1.csv')

    def test_load_base_data_mtf_reads_data_base_csv_with_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            csv_path = d / 'combined_data_final_complete_m1.csv'
            df = pd.DataFrame(
                {
                    'EURUSD_Open': [1.0, 1.1, 1.2],
                    'EURUSD_High': [1.2, 1.3, 1.4],
                    'EURUSD_Low': [0.9, 1.0, 1.1],
                    'EURUSD_Close': [1.1, 1.2, 1.3],
                    'USDJPY_Open': [150.0, 150.1, 150.2],
                    'USDJPY_High': [150.2, 150.3, 150.4],
                    'USDJPY_Low': [149.8, 149.9, 150.0],
                    'USDJPY_Close': [150.1, 150.2, 150.3],
                },
                index=pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC'),
            )
            df.to_csv(csv_path)

            stream = StringIO()
            result = load_base_data_mtf(
                log_stream=stream,
                pairs={'EURUSD': 'EURUSD=X', 'USDJPY': 'USDJPY=X'},
                lookback_days=30,
                base_interval='1m',
                use_local_csv_for_pairs=True,
                base_path=str(d / 'combined_data_final_complete.csv'),
            )

            self.assertIn('EURUSD', result)
            self.assertIn('USDJPY', result)
            self.assertListEqual(list(result['EURUSD'].columns), ['Open', 'High', 'Low', 'Close'])


if __name__ == '__main__':
    unittest.main()
