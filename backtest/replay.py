"""Ledger-first replay backtest helpers for monitoring cycle artifacts.

The replay engine records every decision row, derives an executed trade ledger
for BUY/SELL actions that have complete price data, enriches each decision with
model-feature snapshots plus multi-horizon labels, then summarizes baseline
trading and risk KPIs.

Sprint 2 additions:
- extract feature-model snapshots and gating metadata directly from replay
  artifacts without hard-coupling to a single signal payload schema;
- derive multi-horizon labels (`pnl_1`, `pnl_3`, `pnl_5`, `max_adverse`,
  `max_favorable`) using only data available after cycle ``t``;
- publish gate-attribution summaries so blocked opportunities can be analyzed
  without re-running the replay loop.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import json
import math

SCHEMA_VERSION = "sprint2.v1"
DEFAULT_EQUITY_CURVE_MODE = "additive"
SUPPORTED_EQUITY_CURVE_MODES = {"additive", "compounding"}
DEFAULT_HORIZONS: Tuple[int, ...] = (1, 3, 5)
BASE_GATE_FIELDS: Tuple[str, ...] = (
    "signal_present",
    "entry_price_available",
    "exit_price_available",
    "execution_ready",
)
FEATURE_FIELD_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "rls_confidence": ("rls_confidence", "confidence", "rls_health"),
    "deviation_score": ("deviation_score", "parameter_deviation", "parameter_deviations", "deviation"),
    "kalman_zscore": ("kalman_zscore", "kalman_flip_zscore", "kalman_score"),
    "dcc_correlation": ("dcc_correlation", "correlation", "dcc_corr"),
    "predicted_return": ("predicted_return", "forecast_return", "expected_return"),
    "pred_var": ("pred_var", "prediction_variance", "forecast_variance"),
    "spread": ("spread", "current_spread"),
    "regime_label": ("regime_label", "market_regime", "regime"),
}
FEATURE_CONTAINER_KEYS: Tuple[str, ...] = (
    "features",
    "feature_model",
    "feature_snapshot",
    "model_features",
    "metrics",
)
GATE_CONTAINER_KEYS: Tuple[str, ...] = (
    "action_mask",
    "gate_status",
    "gate_results",
    "gates",
)
BLOCKED_BY_KEYS: Tuple[str, ...] = ("blocked_by", "blocked_gates")
PREFERRED_ACTION_KEYS: Tuple[str, ...] = ("preferred_action", "raw_action", "intended_action", "suggested_action")
PRICE_SOURCE_KEYS: Tuple[str, ...] = ("latest_actual_prices", "actual_prices", "prices")


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
    gate_attribution: List[Dict[str, Any]] = field(default_factory=list)


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
    current_cycle: Mapping[str, Any]
    cycle_results: Sequence[Mapping[str, Any]]
    current_prices: Mapping[str, Any]
    next_prices: Mapping[str, Any]
    fee_bps: float
    slippage_bps: float
    horizons: Tuple[int, ...]


@dataclass(frozen=True)
class _HorizonOutcome:
    labels: Dict[str, Any]
    price_path: List[Dict[str, Any]]


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


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _iter_container_candidates(signal_obj: Mapping[str, Any], current_cycle: Mapping[str, Any], symbol: str) -> Iterator[Mapping[str, Any]]:
    yield signal_obj

    for key in FEATURE_CONTAINER_KEYS + GATE_CONTAINER_KEYS:
        container = _mapping_or_empty(signal_obj.get(key))
        if container:
            yield container

    for cycle_key in ("symbol_features", "feature_models", "feature_snapshots", "signal_features", "symbol_metrics"):
        symbol_map = _mapping_or_empty(current_cycle.get(cycle_key))
        symbol_container = _mapping_or_empty(symbol_map.get(symbol))
        if symbol_container:
            yield symbol_container
            for key in FEATURE_CONTAINER_KEYS + GATE_CONTAINER_KEYS:
                nested = _mapping_or_empty(symbol_container.get(key))
                if nested:
                    yield nested


def _first_present(candidates: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> Any:
    for candidate in candidates:
        for key in keys:
            if key in candidate and candidate.get(key) is not None:
                return candidate.get(key)
    return None


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "pass", "passed", "allow", "allowed"}:
            return True
        if lowered in {"false", "0", "no", "n", "block", "blocked", "fail", "failed"}:
            return False
    return None


def _normalize_blocked_by(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        normalized: List[str] = []
        for item in value:
            if item is None:
                continue
            gate_name = str(item).strip()
            if gate_name and gate_name not in normalized:
                normalized.append(gate_name)
        return normalized
    return []


def _extract_preferred_action(signal_obj: Mapping[str, Any], current_cycle: Mapping[str, Any], symbol: str, final_action: str) -> str:
    candidates = tuple(_iter_container_candidates(signal_obj, current_cycle, symbol))
    preferred = _first_present(candidates, PREFERRED_ACTION_KEYS)
    if preferred is None:
        return final_action
    return str(preferred).upper()


def _extract_feature_snapshot(signal_obj: Mapping[str, Any], current_cycle: Mapping[str, Any], symbol: str) -> Dict[str, Any]:
    candidates = tuple(_iter_container_candidates(signal_obj, current_cycle, symbol))
    features: Dict[str, Any] = {}
    for canonical_name, aliases in FEATURE_FIELD_ALIASES.items():
        value = _first_present(candidates, aliases)
        if canonical_name == "regime_label":
            features[canonical_name] = None if value is None else str(value)
        else:
            features[canonical_name] = _safe_float(value)
    return features


def _extract_gate_evidence(signal_obj: Mapping[str, Any], current_cycle: Mapping[str, Any], symbol: str) -> Tuple[Dict[str, bool], List[str], Optional[bool], Optional[bool]]:
    candidates = tuple(_iter_container_candidates(signal_obj, current_cycle, symbol))
    blocked_by = _normalize_blocked_by(_first_present(candidates, BLOCKED_BY_KEYS))

    can_buy = _normalize_bool(_first_present(candidates, ("can_buy",)))
    can_sell = _normalize_bool(_first_present(candidates, ("can_sell",)))

    non_gate_keys = {
        "signal",
        "entry_price",
        "timestamp",
        "symbol",
        "direction",
        "preferred_direction",
        "feature_model",
        "features",
        "feature_snapshot",
        "model_features",
        "metrics",
        "action_mask",
        "gate_status",
        "gate_results",
        "gates",
        "regime_label",
        "spread",
        "predicted_return",
        "pred_var",
        "rls_confidence",
        "deviation_score",
        "kalman_zscore",
        "dcc_correlation",
        *BLOCKED_BY_KEYS,
        *PREFERRED_ACTION_KEYS,
    }

    gate_results: Dict[str, bool] = {}
    for candidate in candidates:
        for key, value in candidate.items():
            if key in {"can_buy", "can_sell", *non_gate_keys}:
                continue
            normalized = _normalize_bool(value)
            if normalized is not None:
                gate_results.setdefault(str(key), normalized)

    for gate_name in blocked_by:
        gate_results[gate_name] = False
    return gate_results, blocked_by, can_buy, can_sell


def _base_decision_row(
    *,
    decision_id: str,
    timestamp: Any,
    next_timestamp: Any,
    symbol: str,
    action: str,
    preferred_action: str,
    direction: int,
    preferred_direction: int,
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
        "preferred_action": preferred_action,
        "direction": direction,
        "preferred_direction": preferred_direction,
        "signal_generated": preferred_direction != 0,
        "passed_all_gates": False,
        "actually_executed": False,
        "blocked_by": [],
        "gate_results": {},
        "gate_pass_mask": [],
        "gate_fields": [],
        "can_buy": None,
        "can_sell": None,
        "rls_confidence": None,
        "deviation_score": None,
        "kalman_zscore": None,
        "dcc_correlation": None,
        "predicted_return": None,
        "pred_var": None,
        "spread": None,
        "regime_label": None,
        "feature_model": {},
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
        "price_path": [],
        "pnl_1": None,
        "pnl_3": None,
        "pnl_5": None,
        "max_adverse": None,
        "max_favorable": None,
        "hit_1": None,
        "hit_3": None,
        "hit_5": None,
        "t_profit": None,
    }


def _iter_cycle_contexts(
    cycle_results: Sequence[Mapping[str, Any]],
    fee_bps: float,
    slippage_bps: float,
    horizons: Tuple[int, ...],
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
        next_prices = _first_present((_mapping_or_empty(next_cycle),), PRICE_SOURCE_KEYS) or {}
        current_prices = _first_present((_mapping_or_empty(current_cycle),), PRICE_SOURCE_KEYS) or {}
        current_timestamp = current_cycle.get("timestamp")
        next_timestamp = next_cycle.get("timestamp")

        for signal_index, (symbol, signal_obj) in enumerate(trade_signals.items()):
            yield _CycleContext(
                cycle_index=cycle_index,
                signal_index=signal_index,
                timestamp=current_timestamp,
                next_timestamp=next_timestamp,
                symbol=symbol,
                signal_obj=_mapping_or_empty(signal_obj),
                current_cycle=current_cycle,
                cycle_results=cycle_results,
                current_prices=_mapping_or_empty(current_prices),
                next_prices=_mapping_or_empty(next_prices),
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                horizons=horizons,
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


def _apply_gate_result(decision_row: MutableMapping[str, Any], gate_name: str, passed: bool) -> None:
    gate_results = decision_row.setdefault("gate_results", {})
    gate_results[str(gate_name)] = bool(passed)
    if not passed:
        blocked_by = decision_row.setdefault("blocked_by", [])
        if gate_name not in blocked_by:
            blocked_by.append(str(gate_name))


def _compute_horizon_outcomes(
    context: _CycleContext,
    entry_price_raw: Optional[float],
    evaluation_direction: int,
) -> _HorizonOutcome:
    labels: Dict[str, Any] = {
        "pnl_1": None,
        "pnl_3": None,
        "pnl_5": None,
        "max_adverse": None,
        "max_favorable": None,
        "hit_1": None,
        "hit_3": None,
        "hit_5": None,
        "t_profit": None,
    }
    price_path: List[Dict[str, Any]] = []

    if evaluation_direction == 0 or entry_price_raw is None:
        return _HorizonOutcome(labels=labels, price_path=price_path)

    side_cost_rate = (context.fee_bps + context.slippage_bps) / 10_000.0
    returns_by_horizon: Dict[int, float] = {}

    for horizon in context.horizons:
        future_index = context.cycle_index + horizon
        if future_index >= len(context.cycle_results):
            continue

        future_cycle = context.cycle_results[future_index]
        future_prices = _first_present((_mapping_or_empty(future_cycle),), PRICE_SOURCE_KEYS) or {}
        future_price_raw = _safe_float(_mapping_or_empty(future_prices).get(context.symbol))
        if future_price_raw is None:
            continue

        execution_prices = _compute_effective_prices(entry_price_raw, future_price_raw, evaluation_direction, side_cost_rate)
        pnl = evaluation_direction * (
            (execution_prices.exit_price_effective - execution_prices.entry_price_effective)
            / execution_prices.entry_price_effective
        )
        returns_by_horizon[horizon] = pnl
        price_path.append(
            {
                "horizon": horizon,
                "timestamp": _normalize_timestamp(future_cycle.get("timestamp")),
                "exit_price_raw": future_price_raw,
                "pnl": pnl,
            }
        )

    for horizon in context.horizons:
        field_name = f"pnl_{horizon}"
        if field_name in labels:
            labels[field_name] = returns_by_horizon.get(horizon)
            labels[f"hit_{horizon}"] = (returns_by_horizon[horizon] > 0.0) if horizon in returns_by_horizon else None

    observed_returns = list(returns_by_horizon.values())
    if observed_returns:
        labels["max_adverse"] = min(observed_returns)
        labels["max_favorable"] = max(observed_returns)
        labels["t_profit"] = any(value > 0.0 for value in observed_returns)

    return _HorizonOutcome(labels=labels, price_path=price_path)


def _build_trade_row(
    context: _CycleContext,
    decision_row: Mapping[str, Any],
    execution_prices: _ExecutionPrices,
    gross_return: float,
    cost_return: float,
    net_return: float,
) -> Dict[str, Any]:
    trade_row = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": decision_row["decision_id"],
        "timestamp": decision_row["timestamp"],
        "next_timestamp": decision_row["next_timestamp"],
        "symbol": context.symbol,
        "action": decision_row["action"],
        "preferred_action": decision_row["preferred_action"],
        "direction": decision_row["direction"],
        "preferred_direction": decision_row["preferred_direction"],
        "entry_price": execution_prices.entry_price_effective,
        "exit_price": execution_prices.exit_price_effective,
        "entry_price_raw": execution_prices.entry_price_raw,
        "entry_price_effective": execution_prices.entry_price_effective,
        "exit_price_raw": execution_prices.exit_price_raw,
        "exit_price_effective": execution_prices.exit_price_effective,
        "entry_price_source": decision_row["entry_price_source"],
        "used_entry_fallback": decision_row["used_entry_fallback"],
        "gross_return": gross_return,
        "cost_return": cost_return,
        "net_return": net_return,
        "fee_bps": float(context.fee_bps),
        "slippage_bps": float(context.slippage_bps),
        "transaction_cost_bps": float(2.0 * (context.fee_bps + context.slippage_bps)),
        "blocked_by": list(decision_row["blocked_by"]),
        "gate_results": dict(decision_row["gate_results"]),
        "rls_confidence": decision_row["rls_confidence"],
        "deviation_score": decision_row["deviation_score"],
        "kalman_zscore": decision_row["kalman_zscore"],
        "dcc_correlation": decision_row["dcc_correlation"],
        "predicted_return": decision_row["predicted_return"],
        "pred_var": decision_row["pred_var"],
        "spread": decision_row["spread"],
        "regime_label": decision_row["regime_label"],
        "pnl_1": decision_row["pnl_1"],
        "pnl_3": decision_row["pnl_3"],
        "pnl_5": decision_row["pnl_5"],
        "max_adverse": decision_row["max_adverse"],
        "max_favorable": decision_row["max_favorable"],
        "hit_1": decision_row["hit_1"],
        "hit_3": decision_row["hit_3"],
        "hit_5": decision_row["hit_5"],
        "t_profit": decision_row["t_profit"],
    }
    return trade_row


def _build_rows_for_context(context: _CycleContext) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    action = str(context.signal_obj.get("signal", "HOLD")).upper()
    preferred_action = _extract_preferred_action(context.signal_obj, context.current_cycle, context.symbol, action)
    direction = _signal_direction(action)
    preferred_direction = _signal_direction(preferred_action)
    decision_id = _decision_id(context.cycle_index, context.signal_index, context.timestamp, context.symbol)
    decision_row = _base_decision_row(
        decision_id=decision_id,
        timestamp=context.timestamp,
        next_timestamp=context.next_timestamp,
        symbol=context.symbol,
        action=action,
        preferred_action=preferred_action,
        direction=direction,
        preferred_direction=preferred_direction,
        fee_bps=context.fee_bps,
        slippage_bps=context.slippage_bps,
    )

    feature_snapshot = _extract_feature_snapshot(context.signal_obj, context.current_cycle, context.symbol)
    decision_row.update(feature_snapshot)
    decision_row["feature_model"] = feature_snapshot.copy()

    gate_results, blocked_by, can_buy, can_sell = _extract_gate_evidence(context.signal_obj, context.current_cycle, context.symbol)
    decision_row["blocked_by"] = list(blocked_by)
    decision_row["can_buy"] = can_buy if can_buy is not None else (preferred_direction >= 0)
    decision_row["can_sell"] = can_sell if can_sell is not None else (preferred_direction <= 0)
    for gate_name, passed in gate_results.items():
        _apply_gate_result(decision_row, gate_name, passed)

    _apply_gate_result(decision_row, "signal_present", preferred_direction != 0)

    if preferred_direction == 0:
        decision_row["skip_reason"] = "hold_signal"
        return decision_row, None

    entry_price_raw, entry_price_source, used_entry_fallback = _resolve_entry_price(
        context.signal_obj,
        context.current_prices,
        context.symbol,
    )
    exit_price_raw = _safe_float(context.next_prices.get(context.symbol))

    _apply_gate_result(decision_row, "entry_price_available", entry_price_raw is not None)
    _apply_gate_result(decision_row, "exit_price_available", exit_price_raw is not None)

    decision_row["entry_price_raw"] = entry_price_raw
    decision_row["exit_price_raw"] = exit_price_raw
    decision_row["entry_price_source"] = entry_price_source
    decision_row["used_entry_fallback"] = used_entry_fallback

    outcome = _compute_horizon_outcomes(context, entry_price_raw, preferred_direction)
    decision_row.update(outcome.labels)
    decision_row["price_path"] = outcome.price_path

    if entry_price_raw is None:
        decision_row["skip_reason"] = "missing_entry_price"
        return decision_row, None

    if not blocked_by and direction == 0:
        decision_row["skip_reason"] = "hold_signal"
        return decision_row, None

    if blocked_by or any(not passed for gate_name, passed in decision_row["gate_results"].items() if gate_name not in BASE_GATE_FIELDS):
        decision_row["skip_reason"] = "blocked_by_gate"
        return decision_row, None

    if exit_price_raw is None:
        decision_row["skip_reason"] = "missing_exit_price"
        return decision_row, None

    if direction == 0:
        decision_row["skip_reason"] = "hold_signal"
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

    _apply_gate_result(decision_row, "execution_ready", True)
    decision_row["passed_all_gates"] = not decision_row["blocked_by"] and all(decision_row["gate_results"].values())
    decision_row["actually_executed"] = True
    decision_row["entry_price"] = execution_prices.entry_price_effective
    decision_row["exit_price"] = execution_prices.exit_price_effective
    decision_row["entry_price_effective"] = execution_prices.entry_price_effective
    decision_row["exit_price_effective"] = execution_prices.exit_price_effective
    decision_row["skip_reason"] = None
    decision_row["gross_return"] = gross_return
    decision_row["cost_return"] = cost_return
    decision_row["net_return"] = net_return

    trade_row = _build_trade_row(context, decision_row, execution_prices, gross_return, cost_return, net_return)
    return decision_row, trade_row


def _finalize_gate_schema(decision_ledger: List[MutableMapping[str, Any]]) -> List[str]:
    ordered_fields = list(BASE_GATE_FIELDS)
    for row in decision_ledger:
        for gate_name in row.get("gate_results", {}):
            if gate_name not in ordered_fields:
                ordered_fields.append(gate_name)

    for row in decision_ledger:
        gate_results = row.get("gate_results", {})
        row["gate_fields"] = list(ordered_fields)
        row["gate_pass_mask"] = [1 if gate_results.get(gate_name, True) else 0 for gate_name in ordered_fields]
    return ordered_fields


def summarize_gate_attribution(
    decision_ledger: Iterable[Mapping[str, Any]],
    horizon_field: str = "pnl_1",
) -> List[Dict[str, Any]]:
    per_gate_records: Dict[str, List[float]] = {}
    evaluation_count: Dict[str, int] = {}
    unlocked_count: Dict[str, int] = {}

    for row in decision_ledger:
        preferred_direction = int(row.get("preferred_direction") or 0)
        blocked_by = list(row.get("blocked_by") or [])
        label_value = row.get(horizon_field)
        candidate_pnl = float(label_value) if label_value is not None else 0.0
        actual_pnl = float(row.get("net_return")) if row.get("actually_executed") else 0.0

        for gate_name in row.get("gate_fields", []):
            if gate_name in BASE_GATE_FIELDS:
                continue
            evaluation_count[gate_name] = evaluation_count.get(gate_name, 0) + 1
            without_gate_pnl = actual_pnl
            gate_unlocks_trade = preferred_direction != 0 and gate_name in blocked_by and len(blocked_by) == 1
            if gate_unlocks_trade:
                without_gate_pnl = candidate_pnl
                unlocked_count[gate_name] = unlocked_count.get(gate_name, 0) + 1
            impact = without_gate_pnl - actual_pnl
            per_gate_records.setdefault(gate_name, []).append(impact)

    attribution_rows: List[Dict[str, Any]] = []
    for gate_name in sorted(per_gate_records):
        impacts = per_gate_records[gate_name]
        sorted_impacts = sorted(impacts)
        last_index = max(0, math.ceil(0.95 * len(sorted_impacts)) - 1)
        attribution_rows.append(
            {
                "gate": gate_name,
                "horizon_field": horizon_field,
                "observations": len(impacts),
                "evaluated_decisions": evaluation_count.get(gate_name, 0),
                "unblocked_trade_count": unlocked_count.get(gate_name, 0),
                "mean_impact": sum(impacts) / len(impacts),
                "median_impact": sorted_impacts[len(sorted_impacts) // 2],
                "tail_impact_p95": sorted_impacts[last_index],
                "positive_impact_rate": sum(1 for impact in impacts if impact > 0.0) / len(impacts),
            }
        )
    return attribution_rows


def build_replay_ledgers(
    cycle_results: List[Dict[str, Any]],
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    equity_curve_mode: str = DEFAULT_EQUITY_CURVE_MODE,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> ReplayResult:
    """Build row-level decision and trade ledgers for one-bar replay.

    Rules:
    - execute a BUY/SELL signal at the end of cycle ``t``;
    - prefer ``signal.entry_price`` and fall back to ``latest_actual_prices``;
    - close the position using the same symbol's actual price at cycle ``t+1``;
    - preserve both raw prices and effective execution prices;
    - expose transaction cost explicitly via ``cost_return``;
    - enrich each decision row with feature snapshots, gating evidence, and
      multi-horizon labels for downstream analysis.
    """
    if equity_curve_mode not in SUPPORTED_EQUITY_CURVE_MODES:
        supported = ", ".join(sorted(SUPPORTED_EQUITY_CURVE_MODES))
        raise ValueError(f"equity_curve_mode harus salah satu dari: {supported}.")

    normalized_horizons = tuple(sorted({int(horizon) for horizon in horizons if int(horizon) > 0}))
    if not normalized_horizons:
        raise ValueError("horizons harus berisi setidaknya satu horizon positif.")

    base_metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cycle_count": len(cycle_results),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "equity_curve_mode": equity_curve_mode,
        "horizons": list(normalized_horizons),
        "parallelizable_units": ["symbol", "cycle_batch"],
    }

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
                **base_metadata,
                "gate_fields": list(BASE_GATE_FIELDS),
                "gate_attribution_horizon": "pnl_1",
                "gate_attribution_summary": [],
                "hold_reason_summary": {},
                "blocked_by_summary": {},
            },
            gate_attribution=[],
        )

    decision_ledger: List[Dict[str, Any]] = []
    trade_ledger: List[Dict[str, Any]] = []

    for context in _iter_cycle_contexts(
        cycle_results,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        horizons=normalized_horizons,
    ):
        decision_row, trade_row = _build_rows_for_context(context)
        decision_ledger.append(decision_row)
        if trade_row is not None:
            trade_ledger.append(trade_row)

    gate_fields = _finalize_gate_schema(decision_ledger)
    gate_attribution = summarize_gate_attribution(decision_ledger, horizon_field="pnl_1")
    hold_reason_summary = dict(sorted(Counter(row.get("skip_reason") for row in decision_ledger if row.get("skip_reason")).items()))
    blocked_by_summary = dict(sorted(Counter(gate for row in decision_ledger for gate in row.get("blocked_by", [])).items()))

    summary = summarize_trade_ledger(
        trade_ledger,
        decision_ledger,
        equity_curve_mode=equity_curve_mode,
    )
    metadata = {
        **base_metadata,
        "gate_fields": gate_fields,
        "gate_attribution_horizon": "pnl_1",
        "gate_attribution_summary": gate_attribution,
        "hold_reason_summary": hold_reason_summary,
        "blocked_by_summary": blocked_by_summary,
        "export_backend_candidates": ["pyarrow", "fastparquet", "polars"],
        "dataframe_api_note": "Replay export tetap berbasis row records agar migrasi transformasi/parquet ke Polars bertahap.",
    }
    return ReplayResult(
        summary=summary,
        decision_ledger=decision_ledger,
        trade_ledger=trade_ledger,
        metadata=metadata,
        gate_attribution=gate_attribution,
    )


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
