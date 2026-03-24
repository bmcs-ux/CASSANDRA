import os
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

from adapters.mt5_adapter import MT5Adapter
import adapters.dummy_MetaTrader5 as mock_mt5
from monitoring import monitor_for_vps as monitor


def _discover_symbol_frames(data_base_dir: str) -> Dict[str, List[str]]:
    frames: Dict[str, List[str]] = {}
    for root, _, files in os.walk(data_base_dir):
        if "symbol=" not in root:
            continue
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            continue
        symbol = root.split("symbol=")[-1].split(os.sep)[0].upper()
        frames.setdefault(symbol, [])
        frames[symbol].extend(os.path.join(root, f) for f in parquet_files)
    return frames


def run_backtest_simulation(data_base_dir: str = "data_base") -> List[Dict[str, Any]]:
    adapter = MT5Adapter(mt5_backend=mock_mt5)

    symbol_frames = _discover_symbol_frames(data_base_dir)
    if not symbol_frames:
        return []

    max_len = 0
    for symbol, file_paths in symbol_frames.items():
        merged = []
        for path in sorted(file_paths):
            df = pd.read_parquet(path)
            if df.empty:
                continue
            merged.append(df)
        if not merged:
            continue

        symbol_df = pd.concat(merged, axis=0).sort_index().reset_index(drop=True)
        mock_mt5.inject_historical_data(symbol, symbol_df)
        max_len = max(max_len, len(symbol_df))

    backtest_id = "BTEST_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    all_cycle_results: List[Dict[str, Any]] = []

    for i in range(max_len):
        mock_mt5.set_simulation_step(i)
        result = monitor.run_single_monitoring_cycle(
            mt5_adapter_instance=adapter,
            pipeline_run_id_for_monitor=backtest_id,
            cycle_count=i + 1,
            is_backtest=True,
        )
        if result:
            all_cycle_results.append(result)

    return all_cycle_results


if __name__ == "__main__":
    replay_results = run_backtest_simulation()
    print(f"[BACKTEST] total cycles: {len(replay_results)}")
