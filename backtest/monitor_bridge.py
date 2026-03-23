"""Helpers to normalize monitoring artifacts into replay-ready cycle_results."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional
import re


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _symbol_feature_lookup(symbol: str, metric_map: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metric_map, Mapping):
        return {}
    if symbol in metric_map and isinstance(metric_map[symbol], Mapping):
        return metric_map[symbol]
    upper_symbol = str(symbol).upper()
    if upper_symbol in metric_map and isinstance(metric_map[upper_symbol], Mapping):
        return metric_map[upper_symbol]
    for key, value in metric_map.items():
        if not isinstance(value, Mapping):
            continue
        key_text = str(key).upper()
        if key_text.endswith(f"::{upper_symbol}") or key_text == upper_symbol:
            return value
    return {}


def _resolve_group_metric(symbol: str, metric_map: Mapping[str, Any], symbol_to_group: Optional[Mapping[str, str]] = None) -> Any:
    direct = _symbol_feature_lookup(symbol, metric_map)
    if direct:
        return direct
    if not symbol_to_group:
        return {}
    group = symbol_to_group.get(symbol) or symbol_to_group.get(str(symbol).upper())
    if group is None:
        return {}
    group_key = str(group)
    if group_key in metric_map:
        return metric_map[group_key]
    for prefix_key in (f"H1::{group_key}", f"M1::{group_key}", f"D1::{group_key}"):
        if prefix_key in metric_map:
            return metric_map[prefix_key]
    return {}


def _preferred_action_from_signal(signal_obj: Mapping[str, Any], consensus_metric: Mapping[str, Any], forecast_metric: Mapping[str, Any]) -> str:
    explicit = signal_obj.get("preferred_action")
    if explicit is not None:
        return str(explicit).upper()

    signal = str(signal_obj.get("signal", "HOLD")).upper()
    if signal in {"BUY", "SELL"}:
        return signal

    signal_d1 = str(consensus_metric.get("signal_d1", "")).upper()
    signal_h1 = str(consensus_metric.get("signal_h1", "")).upper()
    if signal_d1 in {"BUY", "SELL"}:
        return signal_d1
    if signal_h1 in {"BUY", "SELL"}:
        return signal_h1

    forecast_ret = _safe_float(forecast_metric.get("rls_expected_return_pct"))
    if forecast_ret is not None:
        if forecast_ret > 0:
            return "BUY"
        if forecast_ret < 0:
            return "SELL"
    return "HOLD"


def _build_action_mask(signal_obj: Mapping[str, Any], news_status: Mapping[str, Any]) -> Dict[str, bool]:
    reason = str(signal_obj.get("reason", ""))
    mask = {
        "confidence_gate": True,
        "deviation_gate": True,
        "consensus_gate": True,
        "kalman_gate": True,
        "news_gate": not bool(news_status.get("is_restricted")),
        "direction_gate": True,
        "variance_gate": True,
    }
    if "Low Confidence" in reason:
        mask["confidence_gate"] = False
    if "RLS deviation" in reason or "High Deviation" in reason or "RLS unstable" in reason:
        mask["deviation_gate"] = False
    if "Consensus/Kalman gate blocked" in reason:
        mask["consensus_gate"] = False
    if "Kalman structural break" in reason:
        mask["kalman_gate"] = False
    if "News Restriction" in reason:
        mask["news_gate"] = False
    if "RLS confirmation failed" in reason:
        dir_match = re.search(r"dir_ok=(\d)", reason)
        var_match = re.search(r"var_ok=(\d)", reason)
        if dir_match and dir_match.group(1) == "0":
            mask["direction_gate"] = False
        if var_match and var_match.group(1) == "0":
            mask["variance_gate"] = False
    return mask


def _blocked_by_from_mask(action_mask: Mapping[str, bool]) -> List[str]:
    return [gate for gate, passed in action_mask.items() if not bool(passed)]


def normalize_trade_signal_for_replay(
    symbol: str,
    signal_obj: Mapping[str, Any],
    monitor_cycle: Mapping[str, Any],
    symbol_to_group: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    signal_obj = dict(signal_obj or {})
    news_status = monitor_cycle.get("news_status", {}) or {}
    rls_health = _resolve_group_metric(symbol, monitor_cycle.get("rls_health", {}), symbol_to_group)
    parameter_deviation = _resolve_group_metric(symbol, monitor_cycle.get("parameter_deviations", {}), symbol_to_group)
    dcc_metric = _resolve_group_metric(symbol, monitor_cycle.get("dcc_metrics", {}), symbol_to_group)
    kalman_metric = _symbol_feature_lookup(symbol, monitor_cycle.get("kalman_metrics", {}))
    consensus_metric = _symbol_feature_lookup(symbol, monitor_cycle.get("consensus_metrics", {}))
    forecast_metric = _symbol_feature_lookup(symbol, monitor_cycle.get("rls_forecast", {}))
    deviation_metric = _symbol_feature_lookup(symbol, monitor_cycle.get("deviation_results", {}))

    preferred_action = _preferred_action_from_signal(signal_obj, consensus_metric, forecast_metric)
    action_mask = dict(signal_obj.get("action_mask") or _build_action_mask(signal_obj, news_status))
    blocked_by = list(signal_obj.get("blocked_by") or _blocked_by_from_mask(action_mask))

    predicted_return_pct = _safe_float(forecast_metric.get("rls_expected_return_pct"))
    predicted_return = None if predicted_return_pct is None else predicted_return_pct / 100.0

    feature_model = dict(signal_obj.get("feature_model") or {})
    feature_model.setdefault("rls_confidence", _safe_float(rls_health.get("confidence")))
    deviation_value = parameter_deviation.get("deviation") if isinstance(parameter_deviation, Mapping) else parameter_deviation
    feature_model.setdefault("deviation_score", _safe_float(deviation_value))
    feature_model.setdefault("kalman_zscore", _safe_float(kalman_metric.get("innovation_zscore") or kalman_metric.get("kalman_z")))
    feature_model.setdefault("dcc_correlation", _safe_float(dcc_metric.get("contagion_score") or dcc_metric.get("dcc_correlation")))
    feature_model.setdefault("predicted_return", predicted_return)
    feature_model.setdefault("pred_var", _safe_float(rls_health.get("pred_var")))
    feature_model.setdefault("spread", _safe_float(deviation_metric.get("spread") or signal_obj.get("spread")))
    feature_model.setdefault("regime_label", signal_obj.get("regime_label") or kalman_metric.get("trend") or consensus_metric.get("kalman_trend"))

    normalized = dict(signal_obj)
    normalized["signal"] = str(normalized.get("signal", "HOLD")).upper()
    normalized["preferred_action"] = preferred_action
    normalized["feature_model"] = feature_model
    normalized["action_mask"] = action_mask
    normalized["blocked_by"] = blocked_by
    return normalized


def normalize_monitor_cycle_for_replay(
    monitor_cycle: Mapping[str, Any],
    symbol_to_group: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    trade_signals = monitor_cycle.get("trade_signals", {}) or {}
    replay_trade_signals = {
        symbol: normalize_trade_signal_for_replay(symbol, signal_obj, monitor_cycle, symbol_to_group=symbol_to_group)
        for symbol, signal_obj in trade_signals.items()
    }
    normalized_cycle: Dict[str, Any] = {
        "timestamp": monitor_cycle.get("timestamp"),
        "latest_actual_prices": dict(monitor_cycle.get("latest_actual_prices", {}) or {}),
        "trade_signals": replay_trade_signals,
    }

    passthrough_fields = (
        "cycle_number",
        "rls_health",
        "deviation_results",
        "rls_forecast",
        "parameter_deviations",
        "dcc_metrics",
        "kalman_metrics",
        "consensus_metrics",
        "mean_reversion_candidates",
        "pipeline_run_id",
        "global_metrics",
        "news_status",
    )
    for field in passthrough_fields:
        if field in monitor_cycle:
            normalized_cycle[field] = monitor_cycle[field]
    return normalized_cycle


def normalize_monitor_cycles_for_replay(
    monitor_cycles: Iterable[Mapping[str, Any]],
    symbol_to_group: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    return [normalize_monitor_cycle_for_replay(cycle, symbol_to_group=symbol_to_group) for cycle in monitor_cycles]
