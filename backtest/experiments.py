"""Sprint 3 strategy-evaluation framework for replay backtests.

The helpers in this module turn replay decision ledgers into a systematic
experiment runner that can:
- enumerate parameter combinations for threshold/gate experiments;
- evaluate each configuration on deterministic one-bar labels;
- segment results by market regime;
- run walk-forward validation without leaking future rows into earlier windows;
- rank configurations using risk-adjusted return plus stability penalties.

The module is intentionally ledger-first: once ``build_replay_ledgers`` has
produced decision rows, experiments can be re-run cheaply without rebuilding
signal artefacts from scratch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from math import sqrt
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .replay import build_replay_ledgers

DEFAULT_HORIZON_FIELD = "pnl_1"
DEFAULT_REGIME_LABEL = "unclassified"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    rls_confidence_entry_threshold: Optional[float] = None
    rls_deviation_threshold: Optional[float] = None
    rls_deviation_close_all_threshold: Optional[float] = None
    consensus_threshold: Optional[float] = None
    kalman_flip_zscore: Optional[float] = None


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_start_timestamp: Optional[str]
    train_end_timestamp: Optional[str]
    test_start_timestamp: Optional[str]
    test_end_timestamp: Optional[str]


def build_experiment_grid(param_grid: Mapping[str, Sequence[Any]]) -> List[ExperimentConfig]:
    """Enumerate systematic parameter combinations for replay experiments."""
    ordered_keys = [
        "rls_confidence_entry_threshold",
        "rls_deviation_threshold",
        "rls_deviation_close_all_threshold",
        "consensus_threshold",
        "kalman_flip_zscore",
    ]
    value_lists = [list(param_grid.get(key, [None])) or [None] for key in ordered_keys]

    configs: List[ExperimentConfig] = []
    for index, values in enumerate(product(*value_lists), start=1):
        kwargs = dict(zip(ordered_keys, values))
        name = "cfg_{:03d}".format(index)
        configs.append(ExperimentConfig(name=name, **kwargs))
    return configs


def run_replay_experiment_grid(
    cycle_results: Sequence[Mapping[str, Any]],
    param_grid: Mapping[str, Sequence[Any]],
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    horizons: Sequence[int] = (1, 3, 5),
    horizon_field: str = DEFAULT_HORIZON_FIELD,
    train_size: int = 20,
    test_size: int = 10,
    step_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Convenience API: replay once, then evaluate a full Sprint 3 experiment grid."""
    replay_result = build_replay_ledgers(
        list(cycle_results),
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        horizons=horizons,
    )
    experiment_report = evaluate_walk_forward_grid(
        replay_result.decision_ledger,
        build_experiment_grid(param_grid),
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        horizon_field=horizon_field,
    )
    experiment_report["replay_metadata"] = replay_result.metadata
    return experiment_report


def evaluate_walk_forward_grid(
    decision_ledger: Sequence[Mapping[str, Any]],
    configs: Sequence[ExperimentConfig],
    *,
    train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
    horizon_field: str = DEFAULT_HORIZON_FIELD,
) -> Dict[str, Any]:
    ordered_rows = _sort_decision_rows(decision_ledger)
    windows = generate_walk_forward_windows(ordered_rows, train_size=train_size, test_size=test_size, step_size=step_size)

    config_reports: List[Dict[str, Any]] = []
    for config in configs:
        window_reports: List[Dict[str, Any]] = []
        for window in windows:
            train_rows = ordered_rows[window.train_start:window.train_end]
            test_rows = ordered_rows[window.test_start:window.test_end]
            train_summary = evaluate_strategy_config(train_rows, config, horizon_field=horizon_field)
            test_summary = evaluate_strategy_config(test_rows, config, horizon_field=horizon_field)
            window_reports.append(
                {
                    "window": asdict(window),
                    "train_summary": train_summary,
                    "test_summary": test_summary,
                }
            )

        aggregate = aggregate_walk_forward_results(config, window_reports)
        config_reports.append(
            {
                "config": asdict(config),
                "windows": window_reports,
                "aggregate": aggregate,
            }
        )

    ranking = rank_experiment_results(config_reports)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_field": horizon_field,
        "window_count": len(windows),
        "windows": [asdict(window) for window in windows],
        "configs": config_reports,
        "ranking": ranking,
    }


def generate_walk_forward_windows(
    decision_ledger: Sequence[Mapping[str, Any]],
    *,
    train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
) -> List[WalkForwardWindow]:
    """Build rolling walk-forward windows using decision order as chronology."""
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size dan test_size harus bernilai positif.")

    ordered_rows = _sort_decision_rows(decision_ledger)
    if len(ordered_rows) < train_size + test_size:
        return []

    step = step_size or test_size
    windows: List[WalkForwardWindow] = []
    cursor = 0
    window_index = 1
    while cursor + train_size + test_size <= len(ordered_rows):
        train_start = cursor
        train_end = cursor + train_size
        test_start = train_end
        test_end = test_start + test_size
        train_rows = ordered_rows[train_start:train_end]
        test_rows = ordered_rows[test_start:test_end]
        windows.append(
            WalkForwardWindow(
                window_id=f"wf_{window_index:03d}",
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_start_timestamp=_row_timestamp(train_rows[0]),
                train_end_timestamp=_row_timestamp(train_rows[-1]),
                test_start_timestamp=_row_timestamp(test_rows[0]),
                test_end_timestamp=_row_timestamp(test_rows[-1]),
            )
        )
        cursor += step
        window_index += 1
    return windows


def evaluate_strategy_config(
    decision_ledger: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
    *,
    horizon_field: str = DEFAULT_HORIZON_FIELD,
) -> Dict[str, Any]:
    """Evaluate one parameter configuration against replay decision rows."""
    executed_returns: List[float] = []
    blocked_counts: Dict[str, int] = {}
    regime_returns: Dict[str, List[float]] = {}
    evaluation_rows: List[Dict[str, Any]] = []

    for row in _sort_decision_rows(decision_ledger):
        preferred_direction = int(row.get("preferred_direction") or 0)
        candidate_return = _safe_float(row.get(horizon_field))
        regime_label = derive_regime_label(row)
        gate_results = _evaluate_config_gates(row, config)
        blocked_by = [gate for gate, passed in gate_results.items() if not passed]
        signal_ready = preferred_direction != 0 and candidate_return is not None
        actually_executed = signal_ready and not blocked_by

        if actually_executed:
            executed_returns.append(candidate_return)
            regime_returns.setdefault(regime_label, []).append(candidate_return)
        for gate_name in blocked_by:
            blocked_counts[gate_name] = blocked_counts.get(gate_name, 0) + 1

        evaluation_rows.append(
            {
                "decision_id": row.get("decision_id"),
                "timestamp": _row_timestamp(row),
                "symbol": row.get("symbol"),
                "preferred_action": row.get("preferred_action"),
                "preferred_direction": preferred_direction,
                "regime_label": regime_label,
                "candidate_return": candidate_return,
                "gate_results": gate_results,
                "blocked_by": blocked_by,
                "actually_executed": actually_executed,
            }
        )

    summary = _summarize_returns(executed_returns)
    summary.update(
        {
            "config": asdict(config),
            "evaluated_decisions": len(evaluation_rows),
            "executed_trades": len(executed_returns),
            "blocked_by_summary": dict(sorted(blocked_counts.items())),
            "regime_summary": _summarize_regimes(regime_returns),
            "evaluation_rows": evaluation_rows,
        }
    )
    return summary


def aggregate_walk_forward_results(config: ExperimentConfig, window_reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    test_net_returns: List[float] = []
    test_sharpes: List[float] = []
    drawdowns: List[float] = []
    cvars: List[float] = []
    trade_counts: List[int] = []
    blocked_counts: Dict[str, int] = {}
    regime_collector: Dict[str, List[float]] = {}

    for report in window_reports:
        test_summary = report["test_summary"]
        test_net_returns.append(float(test_summary["net_return"]))
        test_sharpes.append(float(test_summary["sharpe"]))
        drawdowns.append(float(test_summary["max_drawdown"]))
        cvars.append(float(test_summary["cvar_95"]))
        trade_counts.append(int(test_summary["executed_trades"]))
        for gate_name, count in test_summary["blocked_by_summary"].items():
            blocked_counts[gate_name] = blocked_counts.get(gate_name, 0) + int(count)
        for regime_label, regime_summary in test_summary["regime_summary"].items():
            regime_collector.setdefault(regime_label, []).append(float(regime_summary["net_return"]))

    stability = pstdev(test_net_returns) if len(test_net_returns) > 1 else 0.0
    avg_sharpe = mean(test_sharpes) if test_sharpes else 0.0
    total_net_return = sum(test_net_returns)
    max_drawdown = max(drawdowns) if drawdowns else 0.0
    mean_cvar = mean(cvars) if cvars else 0.0
    score = total_net_return + avg_sharpe - max_drawdown + mean_cvar - stability

    return {
        "config": asdict(config),
        "window_count": len(window_reports),
        "total_test_net_return": total_net_return,
        "average_test_sharpe": avg_sharpe,
        "average_test_trades": mean(trade_counts) if trade_counts else 0.0,
        "max_test_drawdown": max_drawdown,
        "mean_test_cvar_95": mean_cvar,
        "stability_variance": stability,
        "ranking_score": score,
        "blocked_by_summary": dict(sorted(blocked_counts.items())),
        "regime_summary": {
            regime_label: {
                "window_count": len(values),
                "average_net_return": mean(values),
                "best_net_return": max(values),
                "worst_net_return": min(values),
            }
            for regime_label, values in sorted(regime_collector.items())
        },
    }


def rank_experiment_results(config_reports: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ranking = []
    for report in config_reports:
        aggregate = dict(report["aggregate"])
        ranking.append(
            {
                "config": report["config"],
                "ranking_score": aggregate["ranking_score"],
                "total_test_net_return": aggregate["total_test_net_return"],
                "average_test_sharpe": aggregate["average_test_sharpe"],
                "max_test_drawdown": aggregate["max_test_drawdown"],
                "mean_test_cvar_95": aggregate["mean_test_cvar_95"],
                "stability_variance": aggregate["stability_variance"],
                "average_test_trades": aggregate["average_test_trades"],
            }
        )
    return sorted(
        ranking,
        key=lambda item: (
            item["ranking_score"],
            item["total_test_net_return"],
            item["average_test_sharpe"],
            -item["max_test_drawdown"],
            item["mean_test_cvar_95"],
        ),
        reverse=True,
    )


def derive_regime_label(row: Mapping[str, Any]) -> str:
    explicit = row.get("regime_label")
    if explicit:
        return str(explicit)

    predicted_return = _safe_float(row.get("predicted_return")) or 0.0
    pred_var = _safe_float(row.get("pred_var")) or 0.0
    correlation = _safe_float(row.get("dcc_correlation")) or 0.0
    spread = _safe_float(row.get("spread")) or 0.0

    volatility_state = "high_vol" if pred_var >= 0.01 else "low_vol"
    trend_state = "trend_up" if predicted_return >= 0.0 else "trend_down"
    correlation_state = "high_corr" if abs(correlation) >= 0.6 else "low_corr"
    spread_state = "wide_spread" if spread >= 0.001 else "tight_spread"
    label = f"{volatility_state}:{trend_state}:{correlation_state}:{spread_state}"
    return label or DEFAULT_REGIME_LABEL


def _evaluate_config_gates(row: Mapping[str, Any], config: ExperimentConfig) -> Dict[str, bool]:
    deviation_score = abs(_safe_float(row.get("deviation_score")) or 0.0)
    confidence = _safe_float(row.get("rls_confidence"))
    kalman_zscore = abs(_safe_float(row.get("kalman_zscore")) or 0.0)
    consensus_score = _compute_consensus_score(row)

    gate_results = {
        "confidence_gate": True if config.rls_confidence_entry_threshold is None or confidence is None else confidence >= config.rls_confidence_entry_threshold,
        "deviation_gate": True if config.rls_deviation_threshold is None else deviation_score <= config.rls_deviation_threshold,
        "deviation_close_all_gate": True if config.rls_deviation_close_all_threshold is None else deviation_score <= config.rls_deviation_close_all_threshold,
        "consensus_gate": True if config.consensus_threshold is None else consensus_score >= config.consensus_threshold,
        "kalman_flip_gate": True if config.kalman_flip_zscore is None else kalman_zscore <= config.kalman_flip_zscore,
    }
    return gate_results


def _compute_consensus_score(row: Mapping[str, Any]) -> float:
    predicted_return = abs(_safe_float(row.get("predicted_return")) or 0.0)
    pred_var = _safe_float(row.get("pred_var"))
    confidence = _safe_float(row.get("rls_confidence")) or 0.0

    if pred_var is not None and pred_var > 0.0:
        return predicted_return / sqrt(pred_var)
    return predicted_return * max(confidence, 1.0)


def _summarize_regimes(regime_returns: Mapping[str, Sequence[float]]) -> Dict[str, Dict[str, float]]:
    return {
        regime_label: {
            "trade_count": len(values),
            "net_return": sum(values),
            "avg_return": mean(values),
            "win_rate": sum(1 for value in values if value > 0.0) / len(values),
        }
        for regime_label, values in sorted(regime_returns.items())
        if values
    }


def _summarize_returns(returns: Sequence[float]) -> Dict[str, float]:
    returns_list = list(returns)
    total = len(returns_list)
    net_return = sum(returns_list)
    avg_return = mean(returns_list) if returns_list else 0.0
    volatility = pstdev(returns_list) if len(returns_list) > 1 else 0.0
    downside = [value for value in returns_list if value < 0.0]
    downside_vol = pstdev(downside) if len(downside) > 1 else (abs(downside[0]) if downside else 0.0)
    sharpe = 0.0 if volatility == 0.0 else (avg_return / volatility) * sqrt(total)
    sortino = 0.0 if downside_vol == 0.0 else (avg_return / downside_vol) * sqrt(total)
    cvar_95 = _compute_cvar(returns_list, alpha=0.95)
    equity_curve = _build_additive_curve(returns_list)
    max_drawdown = _compute_additive_drawdown(equity_curve)

    return {
        "total_trades": total,
        "win_rate": (sum(1 for value in returns_list if value > 0.0) / total) if total else 0.0,
        "gross_return": net_return,
        "net_return": net_return,
        "avg_return": avg_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "cvar_95": cvar_95,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
    }


def _compute_cvar(returns: Sequence[float], *, alpha: float) -> float:
    if not returns:
        return 0.0
    sorted_returns = sorted(returns)
    tail_count = max(1, int(round((1.0 - alpha) * len(sorted_returns))))
    tail = sorted_returns[:tail_count]
    return sum(tail) / len(tail)


def _build_additive_curve(returns: Sequence[float]) -> List[float]:
    equity = 0.0
    curve: List[float] = []
    for value in returns:
        equity += value
        curve.append(equity)
    return curve


def _compute_additive_drawdown(curve: Sequence[float]) -> float:
    peak = 0.0
    max_drawdown = 0.0
    for value in curve:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, peak - value)
    return max_drawdown


def _sort_decision_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: (_timestamp_sort_key(row.get("timestamp")), str(row.get("decision_id") or "")))


def _timestamp_sort_key(value: Any) -> Tuple[int, str]:
    normalized = _row_timestamp({"timestamp": value})
    if normalized is None:
        return (1, "")
    return (0, normalized)


def _row_timestamp(row: Mapping[str, Any]) -> Optional[str]:
    value = row.get("timestamp")
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
