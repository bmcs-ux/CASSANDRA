"""Ledger-first replay backtest helpers for monitoring cycle artifacts.

Sprint 1 freezes a minimal replay contract around ``cycle_results`` entries that
provide ``timestamp``, ``trade_signals``, and ``latest_actual_prices``. The
replay engine records every decision row, derives an executed trade ledger for
BUY/SELL actions that have complete price data, then summarizes baseline
trading and risk KPIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import json
import math

import pandas as pd

SCHEMA_VERSION = "sprint1.v1"


@dataclass
class BacktestSummary:
    total_trades: int
    win_rate: float
    gross_pnl: float
    avg_pnl_per_trade: float
    net_pnl: float = 0.0
    gross_return: float = 0.0
    net_return: float = 0.0
    skipped_trades: int = 0
    max_drawdown: float = 0.0
    equity_curve: List[float] = field(default_factory=list)


@dataclass
class ReplayResult:
    summary: BacktestSummary
    decision_ledger: List[Dict[str, Any]]
    trade_ledger: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def _signal_direction(signal: str) -> int:
    if signal == "BUY":
        return 1
    if signal == "SELL":
        return -1
    return 0


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted):
        return None
    return converted


def _normalize_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _base_decision_row(
    *,
    timestamp: Any,
    next_timestamp: Any,
    symbol: str,
    action: str,
    direction: int,
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _normalize_timestamp(timestamp),
        "next_timestamp": _normalize_timestamp(next_timestamp),
        "symbol": symbol,
        "action": action,
        "preferred_action": action,
        "direction": direction,
        "signal_generated": direction != 0,
        "passed_all_gates": direction != 0,
        "actually_executed": False,
        "blocked_by": [],
        "gate_pass_mask": [],
        "rls_confidence": None, # Akan diisi dari cycle_results
        "deviation_score": None,
        "kalman_zscore": None,
        "dcc_correlation": None,
        "regime_label": None,
        "entry_price": None,
        "exit_price": None,
        "entry_price_source": None,
        "gross_return": None,
        "net_return": None,
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "transaction_cost_bps": float(2.0 * (fee_bps + slippage_bps)),
        "skip_reason": None,
        "used_entry_fallback": False,
    }


def build_replay_ledgers(
    cycle_results: List[Dict[str, Any]],
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> ReplayResult:
    """Build row-level decision and trade ledgers for one-bar replay.

    Rules:
    - execute a BUY/SELL signal at the end of cycle ``t``;
    - prefer ``signal.entry_price`` and fall back to ``latest_actual_prices``;
    - close the position using the same symbol's actual price at cycle ``t+1``;
    - charge fee and slippage on both sides.
    """
    if len(cycle_results) < 2:
        empty_summary = BacktestSummary(total_trades=0, win_rate=0.0, gross_pnl=0.0, avg_pnl_per_trade=0.0)
        return ReplayResult(
            summary=empty_summary,
            decision_ledger=[],
            trade_ledger=[],
            metadata={
                "schema_version": SCHEMA_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cycle_count": len(cycle_results),
                "fee_bps": float(fee_bps),
                "slippage_bps": float(slippage_bps),
            },
        )

    decision_ledger: List[Dict[str, Any]] = []
    trade_ledger: List[Dict[str, Any]] = []
    cost_rate = (fee_bps + slippage_bps) / 10_000.0

    for idx in range(len(cycle_results) - 1):
        current_cycle = cycle_results[idx]
        next_cycle = cycle_results[idx + 1]

        trade_signals = current_cycle.get("trade_signals", {}) or {}
        next_prices = next_cycle.get("latest_actual_prices", {}) or {}
        current_prices = current_cycle.get("latest_actual_prices", {}) or {}
        current_timestamp = current_cycle.get("timestamp")
        next_timestamp = next_cycle.get("timestamp")

        for symbol, signal_obj in trade_signals.items():
            signal_obj = signal_obj or {}
            action = str(signal_obj.get("signal", "HOLD")).upper()
            direction = _signal_direction(action)
            decision_row = _base_decision_row(
                timestamp=current_timestamp,
                next_timestamp=next_timestamp,
                symbol=symbol,
                action=action,
                direction=direction,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )

            if direction == 0:
                decision_row["skip_reason"] = "hold_signal"
                decision_ledger.append(decision_row)
                continue

            explicit_entry_price = _safe_float(signal_obj.get("entry_price"))
            fallback_entry_price = _safe_float(current_prices.get(symbol))
            exit_price = _safe_float(next_prices.get(symbol))

            if explicit_entry_price is not None:
                entry_price = explicit_entry_price
                entry_price_source = "signal"
            else:
                entry_price = fallback_entry_price
                entry_price_source = "latest_actual_prices" if fallback_entry_price is not None else None

            decision_row["entry_price"] = entry_price
            decision_row["exit_price"] = exit_price
            decision_row["entry_price_source"] = entry_price_source
            decision_row["used_entry_fallback"] = explicit_entry_price is None and fallback_entry_price is not None

            if entry_price is None:
                decision_row["skip_reason"] = "missing_entry_price"
                decision_ledger.append(decision_row)
                continue

            if exit_price is None:
                decision_row["skip_reason"] = "missing_exit_price"
                decision_ledger.append(decision_row)
                continue

            gross_return = direction * ((exit_price - entry_price) / entry_price)
            net_return = gross_return - (2.0 * cost_rate)

            decision_row["actually_executed"] = True
            decision_row["skip_reason"] = None
            decision_row["gross_return"] = gross_return
            decision_row["net_return"] = net_return
            decision_ledger.append(decision_row)

            trade_row = {
                "schema_version": SCHEMA_VERSION,
                "timestamp": decision_row["timestamp"],
                "next_timestamp": decision_row["next_timestamp"],
                "symbol": symbol,
                "action": action,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_price_source": entry_price_source,
                "used_entry_fallback": decision_row["used_entry_fallback"],
                "gross_return": gross_return,
                "net_return": net_return,
                "fee_bps": float(fee_bps),
                "slippage_bps": float(slippage_bps),
                "transaction_cost_bps": float(2.0 * (fee_bps + slippage_bps)),
            }
            trade_ledger.append(trade_row)

    summary = summarize_trade_ledger(trade_ledger, decision_ledger)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cycle_count": len(cycle_results),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
    }
    return ReplayResult(summary=summary, decision_ledger=decision_ledger, trade_ledger=trade_ledger, metadata=metadata)


def summarize_trade_ledger(
    trade_ledger: Iterable[Mapping[str, Any]],
    decision_ledger: Optional[Iterable[Mapping[str, Any]]] = None,
) -> BacktestSummary:
    trade_rows = list(trade_ledger)
    decision_rows = list(decision_ledger or [])
    total_trades = len(trade_rows)
    wins = sum(1 for row in trade_rows if float(row["net_return"]) > 0.0)
    gross_return = sum(float(row["gross_return"]) for row in trade_rows)
    net_return = sum(float(row["net_return"]) for row in trade_rows)
    equity_curve = _build_equity_curve(trade_rows)
    max_drawdown = _compute_max_drawdown(equity_curve)
    skipped_trades = sum(1 for row in decision_rows if row.get("signal_generated") and not row.get("actually_executed"))
    win_rate = (wins / total_trades) if total_trades else 0.0
    avg_net_return = (net_return / total_trades) if total_trades else 0.0
    return BacktestSummary(
        total_trades=total_trades,
        win_rate=win_rate,
        gross_pnl=gross_return,
        avg_pnl_per_trade=avg_net_return,
        net_pnl=net_return,
        gross_return=gross_return,
        net_return=net_return,
        skipped_trades=skipped_trades,
        max_drawdown=max_drawdown,
        equity_curve=equity_curve,
    )


def _build_equity_curve(trade_ledger: Iterable[Mapping[str, Any]]) -> List[float]:
    equity = 0.0
    curve: List[float] = []
    for row in trade_ledger:
        equity += float(row["net_return"])
        curve.append(equity)
    return curve


def _compute_max_drawdown(equity_curve: Iterable[float]) -> float:
    peak = 0.0
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return max_drawdown


def run_one_bar_replay_backtest(
    cycle_results: List[Dict[str, Any]],
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestSummary:
    """Compatibility wrapper that returns only Sprint 1 summary KPIs."""
    return build_replay_ledgers(cycle_results, fee_bps=fee_bps, slippage_bps=slippage_bps).summary


def export_replay_ledgers_to_parquet(
    replay_result: ReplayResult,
    output_dir: str,
    run_metadata: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, str]:
    """Persist minimal Sprint 1 ledgers as parquet plus a metadata sidecar.

    A parquet engine (``pyarrow`` or ``fastparquet``) must be installed by the
    execution environment.
    """
    engine = _resolve_parquet_engine()
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        **replay_result.metadata,
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        **(dict(run_metadata or {})),
    }

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    decision_df = pd.DataFrame(_attach_export_metadata(replay_result.decision_ledger, metadata))
    trade_df = pd.DataFrame(_attach_export_metadata(replay_result.trade_ledger, metadata))

    decision_path = target_dir / "decision_ledger.parquet"
    trade_path = target_dir / "trade_ledger.parquet"
    metadata_path = target_dir / "metadata.json"

    decision_df.to_parquet(decision_path, engine=engine, index=False)
    trade_df.to_parquet(trade_path, engine=engine, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "decision_ledger": str(decision_path),
        "trade_ledger": str(trade_path),
        "metadata": str(metadata_path),
    }


def _resolve_parquet_engine() -> str:
    if find_spec("pyarrow"):
        return "pyarrow"
    if find_spec("fastparquet"):
        return "fastparquet"
    raise RuntimeError("Parquet export membutuhkan dependency 'pyarrow' atau 'fastparquet'.")


def _attach_export_metadata(rows: Iterable[Mapping[str, Any]], metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [{**row, **metadata} for row in rows]
