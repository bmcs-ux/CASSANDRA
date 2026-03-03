import unittest
from io import StringIO

import numpy as np
import pandas as pd

from fitted_models.def_varx import fit_varx_or_arx


from fitted_models.dcc_garch import DCCGARCH


class DccGarchInterfaceTests(unittest.TestCase):
    def test_dcc_garch_fit_accepts_eps_matrix_without_constructor_params(self):
        rng = np.random.default_rng(7)
        eps = rng.normal(0, 0.01, size=(120, 3))
        cols = ['SERIES_A', 'SERIES_B', 'SERIES_C']

        model = DCCGARCH()
        result = model.fit(eps=eps, column_names=cols, disp=False)

        self.assertIn('dcc_params', result)
        self.assertEqual(model.column_names, cols)
        self.assertEqual(model.get_cov().shape, (120, 3, 3))


class FitVarxOrArxTests(unittest.TestCase):
    def test_fit_single_endog_returns_model_result(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range('2024-01-01', periods=80, freq='D')
        signal = np.sin(np.linspace(0, 8, 80)) * 0.01
        noise = rng.normal(0, 0.002, 80)
        df = pd.DataFrame({'Close_Log_Return': signal + noise}, index=idx)

        stream = StringIO()
        result = fit_varx_or_arx(
            log_stream=stream,
            df_pair=df,
            endog_cols=['Close_Log_Return'],
            exog_cols=[],
            maxlags=2,
        )

        self.assertEqual(result['model_type'], 'SARIMAX(ARX)')
        self.assertIn('Close_Log_Return', result['R2'])
        self.assertIn('fitted_model', result)
        self.assertIsNotNone(result['fitted_model'])

    def test_fit_raises_for_insufficient_observations(self):
        idx = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({'Close_Log_Return': [0.01, 0.0, -0.01, 0.02, 0.01]}, index=idx)

        stream = StringIO()
        with self.assertRaises(ValueError):
            fit_varx_or_arx(
                log_stream=stream,
                df_pair=df,
                endog_cols=['Close_Log_Return'],
                exog_cols=[],
                maxlags=4,
            )


if __name__ == '__main__':
    unittest.main()
