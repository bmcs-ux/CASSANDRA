import importlib.util
import importlib
from pathlib import Path
import unittest
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ExtractSignalsPreferredActionTest(unittest.TestCase):
    def _sample_cycles(self):
        return [
            {
                "timestamp": "2026-03-19T12:20:00+00:00",
                "latest_actual_prices": {"XAUAUD": 6062.51},
                "trade_signals": {
                    "XAUAUD": {
                        "signal": "HOLD",
                        "suggested_action": "BUY",
                        "entry_price": 6062.51,
                        "sl": 6021.24,
                        "tp": 6172.23,
                    }
                },
            },
            {
                "timestamp": "2026-03-19T12:21:00+00:00",
                "latest_actual_prices": {"XAUAUD": 6070.00},
                "trade_signals": {},
            },
        ]

    def test_python_orchestrator_uses_action_alias(self):
        module = _load_module(REPO_ROOT / "python" / "orchestrator.py")
        signals = module.extract_signals(self._sample_cycles())
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["preferred_action"], "BUY")

    def test_root_orchestrator_uses_action_alias(self):
        module = _load_module(REPO_ROOT / "orchestrator.py")
        signals = module.extract_signals(self._sample_cycles())
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["preferred_action"], "BUY")

    def test_python_orchestrator_normalizes_timestamp_to_epoch_ms(self):
        pl = importlib.import_module("polars")
        module = _load_module(REPO_ROOT / "python" / "orchestrator.py")
        ts = datetime(2026, 3, 19, 12, 20, 0, tzinfo=timezone.utc)
        df = pl.DataFrame(
            {
                "Timestamp": [ts],
                "Open": [1.0],
                "High": [1.0],
                "Low": [1.0],
                "Close": [1.0],
            }
        )

        out = module._to_polars(df, "TEST")
        self.assertEqual(str(out.schema["Timestamp"]), "Int64")
        self.assertEqual(out["Timestamp"][0], int(ts.timestamp() * 1000))

    def test_root_orchestrator_normalizes_timestamp_to_epoch_ms(self):
        pl = importlib.import_module("polars")
        module = _load_module(REPO_ROOT / "orchestrator.py")
        ts = datetime(2026, 3, 19, 12, 20, 0, tzinfo=timezone.utc)
        df = pl.DataFrame(
            {
                "Timestamp": [ts],
                "Open": [1.0],
                "High": [1.0],
                "Low": [1.0],
                "Close": [1.0],
            }
        )

        out = module._to_polars(df, "TEST")
        self.assertEqual(str(out.schema["Timestamp"]), "Int64")
        self.assertEqual(out["Timestamp"][0], int(ts.timestamp() * 1000))


if __name__ == "__main__":
    unittest.main()
