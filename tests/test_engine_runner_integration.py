from pathlib import Path

import pandas as pd

import adapters.dummy_MetaTrader5 as dummy_mt5
from adapters.mt5_adapter import MT5Adapter
from backtest.engine_runner import run_backtest_simulation
from monitoring import monitor_for_vps as monitor


def _mini_ohlc_frame():
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-03-01", periods=4, freq="min", tz="UTC"),
            "open": [1.0, 1.01, 1.02, 1.03],
            "high": [1.01, 1.02, 1.03, 1.04],
            "low": [0.99, 1.0, 1.01, 1.02],
            "close": [1.01, 1.02, 1.03, 1.04],
            "tick_volume": [10, 11, 12, 13],
        }
    )


def test_run_single_cycle_backtest_mode_never_sends_live_signal(monkeypatch):
    dummy_mt5.reset_simulation()
    dummy_mt5.inject_historical_data("EURUSD", _mini_ohlc_frame())
    dummy_mt5.set_simulation_step(3)

    monkeypatch.setattr(monitor.parameter, "PAIRS", {"EURUSD": "EURUSD"})
    monkeypatch.setattr(monitor.parameter, "HF_LOOKBACK_DAYS", 1)
    monkeypatch.setattr(monitor.parameter, "HF_BASE_INTERVAL", "1m")

    called = {"count": 0}

    def _forbidden_send(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("send_signal_to_trade_engine should not be called in backtest mode")

    monkeypatch.setattr(monitor, "send_signal_to_trade_engine", _forbidden_send)

    adapter = MT5Adapter(mt5_backend=dummy_mt5)
    result = monitor.run_single_monitoring_cycle(
        mt5_adapter_instance=adapter,
        pipeline_run_id_for_monitor="BTEST_DEMO",
        cycle_count=1,
        is_backtest=True,
    )

    assert result["is_backtest"] is True
    assert called["count"] == 0


def test_engine_runner_passes_is_backtest_true(monkeypatch, tmp_path: Path):
    parquet_dir = tmp_path / "data_base" / "asset_class=forex" / "symbol=EURUSD" / "timeframe=M1"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    (parquet_dir / "data_2026-03.parquet").write_text("fixture")

    monkeypatch.setattr("backtest.engine_runner.pd.read_parquet", lambda _path: _mini_ohlc_frame())

    flags = []

    def _fake_cycle(**kwargs):
        flags.append(kwargs.get("is_backtest"))
        return {
            "cycle_number": kwargs["cycle_count"],
            "timestamp": "2026-03-01T00:00:00Z",
            "latest_actual_prices": {},
            "trade_signals": {},
        }

    monkeypatch.setattr("backtest.engine_runner.monitor.run_single_monitoring_cycle", _fake_cycle)

    results = run_backtest_simulation(data_base_dir=str(tmp_path / "data_base"))

    assert len(results) == 4
    assert flags and all(flag is True for flag in flags)
