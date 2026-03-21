"""Ledger-first replay backtest helpers for monitoring cycle artifacts.

The replay engine records every decision row, derives an executed trade ledger
for BUY/SELL actions that have complete price data, then summarizes baseline
trading and risk KPIs.

Notes for Sprint 2 preparation:
- parquet export is isolated behind lightweight row-writer helpers so the API is
  not tightly coupled to pandas;
- row construction is split into symbol-level helpers to keep the design
  compatible with future parallel execution per symbol or per cycle batch;
- dataframe-style transformations are intentionally kept on plain row records so
  they can be migrated to Polars incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import json
import math

SCHEMA_VERSION = "sprint1.v2"
DEFAULT_EQUITY_CURVE_MODE = "additive"
SUPPORTED_EQUITY_CURVE_MODES = {"additive", "compounding"}
GATE_FIELDS: Tuple[str, ...] = (
    "signal_present",
    "entry_price_available",
    "exit_price_available",
    "execution_ready",
)


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
    equity_curve_mode: str = DEFAULT_EQUITY_CURVE_MODE


@dataclass
class ReplayResult:
    summary: BacktestSummary
    decision_ledger: List[Dict[str, Any]]
    trade_ledger: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class _ExecutionPrices:
    entry_price_raw: float
    entry_price_effective: float
    exit_price_raw: float
    exit_price_effective: float


@dataclass(frozen=True)
class _CycleContext:
    cycle_index: int
    signal_index: int
    timestamp: Any
    next_timestamp: Any
    symbol: str
    signal_obj: Mapping[str, Any]
    current_prices: Mapping[str, Any]
    next_prices: Mapping[str, Any]
    fee_bps: float
    slippage_bps: float


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


def _decision_id(cycle_index: int, signal_index: int, timestamp: Any, symbol: str) -> str:
    normalized_timestamp = _normalize_timestamp(timestamp) or "unknown-ts"
    return f"{normalized_timestamp}:{cycle_index}:{signal_index}:{symbol}"


def _empty_gate_mask() -> List[int]:
    return [0] * len(GATE_FIELDS)


def _set_gate(mask: List[int], gate_name: str, passed: bool, blocked_by: List[str]) -> None:
    gate_index = GATE_FIELDS.index(gate_name)
    mask[gate_index] = 1 if passed else 0
    if not passed and gate_name not in blocked_by:
        blocked_by.append(gate_name)


def _base_decision_row(
    *,
    decision_id: str,
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
        "decision_id": decision_id,
        "timestamp": _normalize_timestamp(timestamp),
        "next_timestamp": _normalize_timestamp(next_timestamp),
        "symbol": symbol,
        "action": action,
        "preferred_action": action,
        "direction": direction,
        "signal_generated": direction != 0,
        "passed_all_gates": False,
        "actually_executed": False,
        "blocked_by": [],
        "gate_pass_mask": _empty_gate_mask(),
        "gate_fields": list(GATE_FIELDS),
        "rls_confidence": None,
        "deviation_score": None,
        "kalman_zscore": None,
        "dcc_correlation": None,
        "regime_label": None,
        "entry_price": None,
        "exit_price": None,
        "entry_price_raw": None,
        "entry_price_effective": None,
        "exit_price_raw": None,
        "exit_price_effective": None,
        "entry_price_source": None,
        "gross_return": None,
        "cost_return": None,
        "net_return": None,
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "transaction_cost_bps": float(2.0 * (fee_bps + slippage_bps)),
        "skip_reason": None,
        "used_entry_fallback": False,
    }


def _iter_cycle_contexts(
    cycle_results: Sequence[Mapping[str, Any]],
    fee_bps: float,
    slippage_bps: float,
) -> Iterator[_CycleContext]:
    """Yield symbol-level contexts.

    This iterator keeps cycle traversal separate from row construction so the
    same payload can later be distributed to workers per symbol or per cycle
    batch without changing ledger schemas.
    """
    for cycle_index in range(len(cycle_results) - 1):
        current_cycle = cycle_results[cycle_index]
        next_cycle = cycle_results[cycle_index + 1]

        trade_signals = current_cycle.get("trade_signals", {}) or {}
        next_prices = next_cycle.get("latest_actual_prices", {}) or {}
        current_prices = current_cycle.get("latest_actual_prices", {}) or {}
        current_timestamp = current_cycle.get("timestamp")
        next_timestamp = next_cycle.get("timestamp")

        for signal_index, (symbol, signal_obj) in enumerate(trade_signals.items()):
            yield _CycleContext(
                cycle_index=cycle_index,
                signal_index=signal_index,
                timestamp=current_timestamp,
                next_timestamp=next_timestamp,
                symbol=symbol,
                signal_obj=signal_obj or {},
                current_prices=current_prices,
                next_prices=next_prices,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )


def _resolve_entry_price(signal_obj: Mapping[str, Any], current_prices: Mapping[str, Any], symbol: str) -> Tuple[Optional[float], Optional[str], bool]:
    explicit_entry_price = _safe_float(signal_obj.get("entry_price"))
    fallback_entry_price = _safe_float(current_prices.get(symbol))

    if explicit_entry_price is not None:
        return explicit_entry_price, "signal", False
    if fallback_entry_price is not None:
        return fallback_entry_price, "latest_actual_prices", True
    return None, None, False


def _compute_effective_prices(entry_price_raw: float, exit_price_raw: float, direction: int, side_cost_rate: float) -> _ExecutionPrices:
    entry_multiplier = 1.0 + (direction * side_cost_rate)
    exit_multiplier = 1.0 - (direction * side_cost_rate)
    return _ExecutionPrices(
        entry_price_raw=entry_price_raw,
        entry_price_effective=entry_price_raw * entry_multiplier,
        exit_price_raw=exit_price_raw,
        exit_price_effective=exit_price_raw * exit_multiplier,
    )


def _build_rows_for_context(context: _CycleContext) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    action = str(context.signal_obj.get("signal", "HOLD")).upper()
    direction = _signal_direction(action)
    decision_id = _decision_id(context.cycle_index, context.signal_index, context.timestamp, context.symbol)
    decision_row = _base_decision_row(
        decision_id=decision_id,
        timestamp=context.timestamp,
        next_timestamp=context.next_timestamp,
        symbol=context.symbol,
        action=action,
        direction=direction,
        fee_bps=context.fee_bps,
        slippage_bps=context.slippage_bps,
    )

    gate_mask = decision_row["gate_pass_mask"]
    blocked_by = decision_row["blocked_by"]
    _set_gate(gate_mask, "signal_present", direction != 0, blocked_by)

    if direction == 0:
        decision_row["skip_reason"] = "hold_signal"
        return decision_row, None

    entry_price_raw, entry_price_source, used_entry_fallback = _resolve_entry_price(
        context.signal_obj,
        context.current_prices,
        context.symbol,
    )
    exit_price_raw = _safe_float(context.next_prices.get(context.symbol))

    _set_gate(gate_mask, "entry_price_available", entry_price_raw is not None, blocked_by)
    _set_gate(gate_mask, "exit_price_available", exit_price_raw is not None, blocked_by)

    decision_row["entry_price_raw"] = entry_price_raw
    decision_row["exit_price_raw"] = exit_price_raw
    decision_row["entry_price_source"] = entry_price_source
    decision_row["used_entry_fallback"] = used_entry_fallback

    if entry_price_raw is None:
        decision_row["skip_reason"] = "missing_entry_price"
        return decision_row, None

    if exit_price_raw is None:
        decision_row["skip_reason"] = "missing_exit_price"
        return decision_row, None

    side_cost_rate = (context.fee_bps + context.slippage_bps) / 10_000.0
    execution_prices = _compute_effective_prices(entry_price_raw, exit_price_raw, direction, side_cost_rate)
    gross_return = direction * ((execution_prices.exit_price_raw - execution_prices.entry_price_raw) / execution_prices.entry_price_raw)
    effective_return = direction * (
        (execution_prices.exit_price_effective - execution_prices.entry_price_effective)
        / execution_prices.entry_price_effective
    )
    cost_return = gross_return - effective_return
    net_return = gross_return - cost_return

    _set_gate(gate_mask, "execution_ready", True, blocked_by)
    decision_row["passed_all_gates"] = True
    decision_row["actually_executed"] = True
    decision_row["entry_price"] = execution_prices.entry_price_effective
    decision_row["exit_price"] = execution_prices.exit_price_effective
    decision_row["entry_price_effective"] = execution_prices.entry_price_effective
    decision_row["exit_price_effective"] = execution_prices.exit_price_effective
    decision_row["skip_reason"] = None
    decision_row["gross_return"] = gross_return
    decision_row["cost_return"] = cost_return
    decision_row["net_return"] = net_return

    trade_row = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": decision_id,
        "timestamp": decision_row["timestamp"],
        "next_timestamp": decision_row["next_timestamp"],
        "symbol": context.symbol,
        "action": action,
        "direction": direction,
        "entry_price": execution_prices.entry_price_effective,
        "exit_price": execution_prices.exit_price_effective,
        "entry_price_raw": execution_prices.entry_price_raw,
        "entry_price_effective": execution_prices.entry_price_effective,
        "exit_price_raw": execution_prices.exit_price_raw,
        "exit_price_effective": execution_prices.exit_price_effective,
        "entry_price_source": entry_price_source,
        "used_entry_fallback": used_entry_fallback,
        "gross_return": gross_return,
        "cost_return": cost_return,
        "net_return": net_return,
        "fee_bps": float(context.fee_bps),
        "slippage_bps": float(context.slippage_bps),
        "transaction_cost_bps": float(2.0 * (context.fee_bps + context.slippage_bps)),
    }
    return decision_row, trade_row


def build_replay_ledgers(
    cycle_results: List[Dict[str, Any]],
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    equity_curve_mode: str = DEFAULT_EQUITY_CURVE_MODE,
) -> ReplayResult:
    """Build row-level decision and trade ledgers for one-bar replay.

    Rules:
    - execute a BUY/SELL signal at the end of cycle ``t``;
    - prefer ``signal.entry_price`` and fall back to ``latest_actual_prices``;
    - close the position using the same symbol's actual price at cycle ``t+1``;
    - preserve both raw prices and effective execution prices;
    - expose transaction cost explicitly via ``cost_return``;
    - support additive (Sprint 1 default) and compounding equity curves.
    """
    if equity_curve_mode not in SUPPORTED_EQUITY_CURVE_MODES:
        supported = ", ".join(sorted(SUPPORTED_EQUITY_CURVE_MODES))
        raise ValueError(f"equity_curve_mode harus salah satu dari: {supported}.")

    if len(cycle_results) < 2:
        empty_summary = BacktestSummary(
            total_trades=0,
            win_rate=0.0,
            gross_pnl=0.0,
            avg_pnl_per_trade=0.0,
            equity_curve_mode=equity_curve_mode,
        )
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
                "equity_curve_mode": equity_curve_mode,
                "gate_fields": list(GATE_FIELDS),
                "parallelizable_units": ["symbol", "cycle_batch"],
            },
        )

    decision_ledger: List[Dict[str, Any]] = []
    trade_ledger: List[Dict[str, Any]] = []

    for context in _iter_cycle_contexts(cycle_results, fee_bps=fee_bps, slippage_bps=slippage_bps):
        decision_row, trade_row = _build_rows_for_context(context)
        decision_ledger.append(decision_row)
        if trade_row is not None:
            trade_ledger.append(trade_row)

    summary = summarize_trade_ledger(
        trade_ledger,
        decision_ledger,
        equity_curve_mode=equity_curve_mode,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cycle_count": len(cycle_results),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "equity_curve_mode": equity_curve_mode,
        "gate_fields": list(GATE_FIELDS),
        "parallelizable_units": ["symbol", "cycle_batch"],
        "export_backend_candidates": ["pyarrow", "fastparquet", "polars"],
        "dataframe_api_note": "Replay export tetap berbasis row records agar migrasi transformasi/parquet ke Polars bertahap.",
    }
    return ReplayResult(summary=summary, decision_ledger=decision_ledger, trade_ledger=trade_ledger, metadata=metadata)


def summarize_trade_ledger(
    trade_ledger: Iterable[Mapping[str, Any]],
    decision_ledger: Optional[Iterable[Mapping[str, Any]]] = None,
    equity_curve_mode: str = DEFAULT_EQUITY_CURVE_MODE,
) -> BacktestSummary:
    trade_rows = list(trade_ledger)
    decision_rows = list(decision_ledger or [])
    total_trades = len(trade_rows)
    wins = sum(1 for row in trade_rows if float(row["net_return"]) > 0.0)
    gross_return = sum(float(row["gross_return"]) for row in trade_rows)
    net_return = sum(float(row["net_return"]) for row in trade_rows)
    equity_curve = _build_equity_curve(trade_rows, mode=equity_curve_mode)
    max_drawdown = _compute_max_drawdown(equity_curve, mode=equity_curve_mode)
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
        equity_curve_mode=equity_curve_mode,
    )


def _build_equity_curve(trade_ledger: Iterable[Mapping[str, Any]], mode: str = DEFAULT_EQUITY_CURVE_MODE) -> List[float]:
    curve: List[float] = []
    if mode == "additive":
        equity = 0.0
        for row in trade_ledger:
            equity += float(row["net_return"])
            curve.append(equity)
        return curve

    equity = 1.0
    for row in trade_ledger:
        equity *= 1.0 + float(row["net_return"])
        curve.append(equity)
    return curve


def _compute_max_drawdown(equity_curve: Iterable[float], mode: str = DEFAULT_EQUITY_CURVE_MODE) -> float:
    if mode == "additive":
        peak = 0.0
    else:
        peak = 1.0

    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if mode == "additive":
            drawdown = peak - equity
        else:
            drawdown = 0.0 if peak == 0 else (peak - equity) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def run_one_bar_replay_backtest(
    cycle_results: List[Dict[str, Any]],
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    equity_curve_mode: str = DEFAULT_EQUITY_CURVE_MODE,
) -> BacktestSummary:
    """Compatibility wrapper that returns only summary KPIs."""
    return build_replay_ledgers(
        cycle_results,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        equity_curve_mode=equity_curve_mode,
    ).summary


def export_replay_ledgers_to_parquet(
    replay_result: ReplayResult,
    output_dir: str,
    run_metadata: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, str]:
    """Persist replay ledgers as parquet plus a metadata sidecar.

    Export intentionally accepts row dictionaries instead of dataframe objects so
    parquet export and dataframe transformations can later move to Polars
    without changing the public replay API.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        **replay_result.metadata,
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        **(dict(run_metadata or {})),
    }

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    decision_rows = _attach_export_metadata(replay_result.decision_ledger, metadata)
    trade_rows = _attach_export_metadata(replay_result.trade_ledger, metadata)

    decision_path = target_dir / "decision_ledger.parquet"
    trade_path = target_dir / "trade_ledger.parquet"
    metadata_path = target_dir / "metadata.json"

    _write_rows_to_parquet(decision_rows, decision_path)
    _write_rows_to_parquet(trade_rows, trade_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "decision_ledger": str(decision_path),
        "trade_ledger": str(trade_path),
        "metadata": str(metadata_path),
    }


def _write_rows_to_parquet(rows: List[Mapping[str, Any]], output_path: Path) -> None:
    if find_spec("pyarrow"):
        pyarrow = import_module("pyarrow")
        pyarrow_parquet = import_module("pyarrow.parquet")
        table = pyarrow.Table.from_pylist(list(rows))
        pyarrow_parquet.write_table(table, output_path)
        return

    if find_spec("fastparquet"):
        # Fallback untuk environment lama; sengaja diisolasi agar replay tidak
        # hard-coupled ke pandas pada API publiknya.
        pandas = import_module("pandas")
        pandas.DataFrame(list(rows)).to_parquet(output_path, engine="fastparquet", index=False)
        return

    raise RuntimeError("Parquet export membutuhkan dependency 'pyarrow' atau 'fastparquet'.")


def _attach_export_metadata(rows: Iterable[Mapping[str, Any]], metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [{**row, **metadata} for row in rows]
