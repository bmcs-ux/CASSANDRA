import tempfile
import unittest
from datetime import date
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch
import zipfile

import pandas as pd

from raw.pair_raw import _build_exness_urls, _resolve_local_csv_path, load_base_data_mtf


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
                },
                index=pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC'),
            )
            df.to_csv(csv_path)

            stream = StringIO()
            result = load_base_data_mtf(
                log_stream=stream,
                pairs={'EURUSD': 'EURUSD=X'},
                lookback_days=30,
                base_interval='1m',
                use_local_csv_for_pairs=True,
                base_path=str(d / 'combined_data_final_complete.csv'),
            )

            self.assertIn('EURUSD', result)
            self.assertListEqual(list(result['EURUSD'].columns), ['Open', 'High', 'Low', 'Close'])


class PairRawLookbackFilterTests(unittest.TestCase):
    def test_apply_lookback_filter_drops_invalid_timestamp_rows(self):
        from raw.pair_raw import _apply_lookback_filter

        stream = StringIO()
        df = pd.DataFrame(
            {
                'Open': [1.0, 1.1],
                'High': [1.2, 1.3],
                'Low': [0.9, 1.0],
                'Close': [1.1, 1.2],
            },
            index=['invalid-ts', '2024-01-01T00:00:00Z'],
        )

        filtered = _apply_lookback_filter(stream, df, lookback_days=30, pair_name='EURUSD')

        self.assertEqual(len(filtered), 1)
        self.assertIn('invalid timestamps', stream.getvalue())


class PairRawExnessFallbackTests(unittest.TestCase):
    @patch('raw.pair_raw.date')
    def test_build_exness_urls_monthly_and_daily(self, mock_date):
        mock_date.today.return_value = date(2024, 5, 10)
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)

        urls = _build_exness_urls('EURUSD', date(2024, 4, 29), date(2024, 5, 2))

        self.assertTrue(any('Exness_EURUSD_2024_04.zip' in u for u in urls))
        self.assertTrue(any('Exness_EURUSD_2024_05_01.zip' in u for u in urls))
        self.assertTrue(any('Exness_EURUSD_2024_05_02.zip' in u for u in urls))

    @patch('raw.pair_raw.requests.get')
    def test_load_base_data_mtf_uses_exness_fallback(self, mock_get):
        df_ticks = pd.DataFrame(
            {
                'Timestamp': ['2024-01-01T00:00:00Z', '2024-01-01T00:01:00Z', '2024-01-01T00:02:00Z'],
                'Bid': [1.1, 1.2, 1.15],
            }
        )

        csv_bytes = df_ticks.to_csv(index=False).encode()
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('ticks.csv', csv_bytes)

        class Resp:
            def __init__(self, content):
                self.content = content

            def raise_for_status(self):
                return None

        mock_get.return_value = Resp(zip_buffer.getvalue())

        stream = StringIO()
        result = load_base_data_mtf(
            log_stream=stream,
            pairs={'EURUSD': 'EURUSD=X'},
            lookback_days=30,
            base_interval='1m',
            use_local_csv_for_pairs=False,
            base_path='data_base/combined_data_final_complete.csv',
        )

        self.assertIn('EURUSD', result)
        self.assertListEqual(list(result['EURUSD'].columns), ['Open', 'High', 'Low', 'Close'])


if __name__ == '__main__':
    unittest.main()
