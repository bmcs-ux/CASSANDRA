from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class BacktestSummary:
    total_trades: int
    win_rate: float
    gross_pnl: float
    avg_pnl_per_trade: float


def _signal_direction(signal: str) -> int:
    if signal == "BUY":
        return 1
    if signal == "SELL":
        return -1
    return 0


def run_one_bar_replay_backtest(cycle_results: List[Dict[str, Any]], fee_bps: float = 0.0, slippage_bps: float = 0.0) -> BacktestSummary:
    """
    Backtest sederhana berbasis output monitoring per cycle.

    Aturan:
    - Eksekusi sinyal di akhir cycle t pada harga `entry_price` (fallback ke `latest_actual_prices`).
    - Posisi ditutup di harga aktual cycle t+1 untuk pair yang sama (one-bar holding).
    - Biaya transaksi dihitung dua sisi (entry + exit) dalam basis points.
    """
    total_pnl = 0.0
    total_trades = 0
    wins = 0

    if len(cycle_results) < 2:
        return BacktestSummary(total_trades=0, win_rate=0.0, gross_pnl=0.0, avg_pnl_per_trade=0.0)

    cost_rate = (fee_bps + slippage_bps) / 10_000.0

    for idx in range(len(cycle_results) - 1):
        current_cycle = cycle_results[idx]
        next_cycle = cycle_results[idx + 1]

        trade_signals = current_cycle.get("trade_signals", {})
        next_prices = next_cycle.get("latest_actual_prices", {})
        current_prices = current_cycle.get("latest_actual_prices", {})

        for pair_name, signal_obj in trade_signals.items():
            direction = _signal_direction(str(signal_obj.get("signal", "HOLD")))
            if direction == 0:
                continue

            entry_price = signal_obj.get("entry_price")
            if not entry_price:
                entry_price = current_prices.get(pair_name)

            exit_price = next_prices.get(pair_name)

            if not entry_price or not exit_price:
                continue

            gross_return = direction * ((float(exit_price) - float(entry_price)) / float(entry_price))
            net_return = gross_return - (2.0 * cost_rate)

            total_pnl += net_return
            total_trades += 1
            if net_return > 0:
                wins += 1

    win_rate = (wins / total_trades) if total_trades else 0.0
    avg_pnl = (total_pnl / total_trades) if total_trades else 0.0
    return BacktestSummary(total_trades=total_trades, win_rate=win_rate, gross_pnl=total_pnl, avg_pnl_per_trade=avg_pnl)
