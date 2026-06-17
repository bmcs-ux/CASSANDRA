"""Ledger-first replay backtest helpers for monitoring cycle artifacts.

Sprint 3 — intrabar simulation dengan fidelitas penuh terhadap live engine:

Fitur baru vs Sprint 2
----------------------
* ``mtf_base_dfs`` : Dict[symbol, List[Dict[OHLCV]]] — data M1 per simbol.
  Posisi tidak lagi selalu ditutup di bar t+1; simulator menelusuri bar M1
  satu per satu sampai SL/TP tersentuh, Kalman flip terjadi, atau batas
  ``max_holding_bars`` tercapai.

* ``PositionSimulator`` : kelas utama yang mereplikasi logika live engine:
    1. Update SL/TP dinamis setiap N bar menggunakan ATR + RLS deviation.
    2. Cek Kalman flip (``close_due_to_kalman_flip``) berdasarkan
       ``innovation_zscore`` dan ``trend`` dari bar M1.
    3. Cek SL/TP hit intrabar berdasarkan urutan OHLC.
    4. Force-close saat ``max_holding_bars`` tercapai.
    5. Tandai ``open_at_end=True`` jika data M1 habis.

* ``exit_reason`` baru: ``sl_hit``, ``tp_hit``, ``kalman_flip``,
  ``max_bars_reached``, ``open_at_end``, ``next_bar_close`` (legacy).

* Posisi ``open_at_end`` **tidak** masuk trade_ledger agar KPI bersih.

* Label ML multi-horizon (``pnl_1/3/5``, ``max_adverse``, ``max_favorable``)
  dihitung dari bar M1 aktual selama holding period.

* ``IntrabarDataAdapter``: abstraksi ringan di atas List[Dict], pandas DF,
  atau polars DF — caller tidak perlu konversi format.

Backward-compatibility
----------------------
``mtf_base_dfs=None`` (default) → perilaku sprint 2 dipertahankan penuh.
``run_one_bar_replay_backtest`` tetap ada sebagai compatibility wrapper.

Struktur ``cycle_results`` yang diharapkan
------------------------------------------
Setiap elemen adalah ``current_cycle_results_summary`` dari live engine:

    {
        "timestamp"           : str,
        "latest_actual_prices": {symbol: float},
        "trade_signals"       : {
            symbol: {
                "signal"         : "BUY"|"SELL"|"HOLD",
                "entry_price"    : float,
                "sl"             : float,
                "tp"             : float,
                "position_units" : float,
                "rls_confidence" : float,
                "kalman_zscore"  : float,
                "deviation_score": float,
                "dcc_correlation": float,
            }
        },
        "kalman_metrics"      : {symbol: {"trend": str, "innovation_zscore": float}},
        "parameter_deviations": {group: float},
        "dcc_metrics"         : {group: {"contagion_score": float}},
        "latest_hf_atrs"      : {symbol: float},   # opsional, untuk dynamic SL/TP
    }
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import (
    Any, Dict, Iterable, Iterator, List, Mapping,
    MutableMapping, Optional, Sequence, Tuple,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "sprint3.v1"
DEFAULT_EQUITY_CURVE_MODE = "additive"
SUPPORTED_EQUITY_CURVE_MODES = {"additive", "compounding"}
DEFAULT_HORIZONS: Tuple[int, ...] = (1, 3, 5)
DEFAULT_MAX_HOLDING_BARS: int = 500
DEFAULT_KALMAN_FLIP_ZSCORE: float = 3.0
DEFAULT_DCC_FLIP_EPS_MULTIPLIER: float = 0.5
DEFAULT_DYNAMIC_SLTP_UPDATE_INTERVAL: int = 0  # 0 = disabled

BASE_GATE_FIELDS: Tuple[str, ...] = (
    "signal_present",
    "entry_price_available",
    "exit_price_available",
    "execution_ready",
)

FEATURE_FIELD_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "rls_confidence":  ("rls_confidence", "confidence", "rls_health"),
    "deviation_score": ("deviation_score", "parameter_deviation", "parameter_deviations", "deviation"),
    "kalman_zscore":   ("kalman_zscore", "kalman_flip_zscore", "kalman_score", "innovation_zscore"),
    "dcc_correlation": ("dcc_correlation", "correlation", "dcc_corr", "contagion_score"),
    "predicted_return":("predicted_return", "forecast_return", "expected_return"),
    "pred_var":        ("pred_var", "prediction_variance", "forecast_variance"),
    "spread":          ("spread", "current_spread"),
    "regime_label":    ("regime_label", "market_regime", "regime"),
}

FEATURE_CONTAINER_KEYS: Tuple[str, ...] = (
    "features", "feature_model", "feature_snapshot", "model_features", "metrics",
)
GATE_CONTAINER_KEYS: Tuple[str, ...] = (
    "action_mask", "gate_status", "gate_results", "gates",
)
BLOCKED_BY_KEYS: Tuple[str, ...]       = ("blocked_by", "blocked_gates")
PREFERRED_ACTION_KEYS: Tuple[str, ...] = (
    "preferred_action", "raw_action", "intended_action", "suggested_action",
)
PRICE_SOURCE_KEYS: Tuple[str, ...] = ("latest_actual_prices", "actual_prices", "prices")

EXIT_SL          = "sl_hit"
EXIT_TP          = "tp_hit"
EXIT_KALMAN_FLIP = "kalman_flip"
EXIT_MAX_BARS    = "max_bars_reached"
EXIT_OPEN_AT_END = "open_at_end"
EXIT_NEXT_BAR    = "next_bar_close"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

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
    open_at_end_count: int = 0
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


@dataclass
class _PositionState:
    """State SL/TP yang dapat berubah selama posisi terbuka."""
    sl_price: Optional[float]
    tp_price: Optional[float]

    def copy(self) -> "_PositionState":
        return _PositionState(sl_price=self.sl_price, tp_price=self.tp_price)


@dataclass(frozen=True)
class _PositionResult:
    exit_price_raw: float
    exit_timestamp: Any
    exit_reason: str
    bars_held: int
    open_at_end: bool
    final_sl: Optional[float]
    final_tp: Optional[float]
    kalman_flip_bar: Optional[int]
    intrabar_path: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# IntrabarDataAdapter
# ---------------------------------------------------------------------------

class IntrabarDataAdapter:
    """Abstraksi format M1: List[Dict], pandas DataFrame, polars DataFrame."""

    def __init__(self, data: Any) -> None:
        self._data = data
        self._kind = self._detect(data)

    @staticmethod
    def _detect(data: Any) -> str:
        if isinstance(data, list):
            return "list"
        mod = getattr(type(data), "__module__", "") or ""
        if "pandas" in mod:
            return "pandas"
        if "polars" in mod:
            return "polars"
        return "list"

    def bars_after(self, entry_timestamp: Any) -> Iterator[Dict[str, Any]]:
        entry_ts = _normalize_timestamp(entry_timestamp)
        for bar in self._iter_all():
            bar_ts = _normalize_timestamp(bar.get("Timestamp"))
            if bar_ts is not None and entry_ts is not None and bar_ts <= entry_ts:
                continue
            yield bar

    def _iter_all(self) -> Iterator[Dict[str, Any]]:
        if self._kind == "list":
            for item in self._data:
                yield dict(item) if not isinstance(item, dict) else item
            return
        if self._kind == "pandas":
            for row in self._data.itertuples(index=False):
                yield row._asdict()
            return
        if self._kind == "polars":
            cols = self._data.columns
            for row in self._data.iter_rows():
                yield dict(zip(cols, row))
            return
        for item in self._data:
            yield dict(item)


# ---------------------------------------------------------------------------
# Internal context
# ---------------------------------------------------------------------------

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
    intrabar_adapter: Optional[IntrabarDataAdapter]
    max_holding_bars: int
    kalman_flip_zscore: float
    dcc_flip_eps_multiplier: float
    dynamic_sltp_update_interval: int


@dataclass(frozen=True)
class _HorizonOutcome:
    labels: Dict[str, Any]
    price_path: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
        r = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(r) else r


def _normalize_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _decision_id(cycle_index: int, signal_index: int, timestamp: Any, symbol: str) -> str:
    ts = _normalize_timestamp(timestamp) or "unknown-ts"
    return f"{ts}:{cycle_index}:{signal_index}:{symbol}"


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lo = value.strip().lower()
        if lo in {"true", "1", "yes", "y", "pass", "passed", "allow", "allowed"}:
            return True
        if lo in {"false", "0", "no", "n", "block", "blocked", "fail", "failed"}:
            return False
    return None


def _normalize_blocked_by(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        seen: List[str] = []
        for item in value:
            if item is None:
                continue
            g = str(item).strip()
            if g and g not in seen:
                seen.append(g)
        return seen
    return []


# ---------------------------------------------------------------------------
# Feature / gate extraction
# ---------------------------------------------------------------------------

def _iter_container_candidates(
    signal_obj: Mapping[str, Any],
    current_cycle: Mapping[str, Any],
    symbol: str,
) -> Iterator[Mapping[str, Any]]:
    yield signal_obj
    for key in FEATURE_CONTAINER_KEYS + GATE_CONTAINER_KEYS:
        c = _mapping_or_empty(signal_obj.get(key))
        if c:
            yield c
    for ck in ("symbol_features", "feature_models", "feature_snapshots",
               "signal_features", "symbol_metrics"):
        sm = _mapping_or_empty(current_cycle.get(ck))
        sc = _mapping_or_empty(sm.get(symbol))
        if sc:
            yield sc
            for key in FEATURE_CONTAINER_KEYS + GATE_CONTAINER_KEYS:
                n = _mapping_or_empty(sc.get(key))
                if n:
                    yield n


def _first_present(candidates: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> Any:
    for candidate in candidates:
        for key in keys:
            if key in candidate and candidate.get(key) is not None:
                return candidate.get(key)
    return None


def _extract_preferred_action(
    signal_obj: Mapping[str, Any],
    current_cycle: Mapping[str, Any],
    symbol: str,
    final_action: str,
) -> str:
    candidates = tuple(_iter_container_candidates(signal_obj, current_cycle, symbol))
    preferred  = _first_present(candidates, PREFERRED_ACTION_KEYS)
    return final_action if preferred is None else str(preferred).upper()


def _extract_feature_snapshot(
    signal_obj: Mapping[str, Any],
    current_cycle: Mapping[str, Any],
    symbol: str,
) -> Dict[str, Any]:
    candidates = tuple(_iter_container_candidates(signal_obj, current_cycle, symbol))
    features: Dict[str, Any] = {}
    for canonical, aliases in FEATURE_FIELD_ALIASES.items():
        value = _first_present(candidates, aliases)
        features[canonical] = (
            None if value is None
            else str(value) if canonical == "regime_label"
            else _safe_float(value)
        )
    return features


def _extract_gate_evidence(
    signal_obj: Mapping[str, Any],
    current_cycle: Mapping[str, Any],
    symbol: str,
) -> Tuple[Dict[str, bool], List[str], Optional[bool], Optional[bool]]:
    candidates = tuple(_iter_container_candidates(signal_obj, current_cycle, symbol))
    blocked_by = _normalize_blocked_by(_first_present(candidates, BLOCKED_BY_KEYS))
    can_buy    = _normalize_bool(_first_present(candidates, ("can_buy",)))
    can_sell   = _normalize_bool(_first_present(candidates, ("can_sell",)))

    non_gate = {
        "signal", "entry_price", "sl", "tp", "stop_loss", "take_profit",
        "position_units", "position_size", "timestamp", "symbol",
        "direction", "preferred_direction",
        "feature_model", "features", "feature_snapshot", "model_features", "metrics",
        "action_mask", "gate_status", "gate_results", "gates",
        "regime_label", "spread", "predicted_return", "pred_var",
        "rls_confidence", "deviation_score", "kalman_zscore", "dcc_correlation",
        "innovation_zscore", "trend", "contagion_score",
        *BLOCKED_BY_KEYS, *PREFERRED_ACTION_KEYS,
    }
    gate_results: Dict[str, bool] = {}
    for candidate in candidates:
        for k, v in candidate.items():
            if k in {"can_buy", "can_sell", *non_gate}:
                continue
            nb = _normalize_bool(v)
            if nb is not None:
                gate_results.setdefault(str(k), nb)
    for gn in blocked_by:
        gate_results[gn] = False
    return gate_results, blocked_by, can_buy, can_sell


# ---------------------------------------------------------------------------
# Signal payload helpers
# ---------------------------------------------------------------------------

def _extract_sl_tp(signal_obj: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    sl = _safe_float(signal_obj.get("sl") or signal_obj.get("stop_loss") or signal_obj.get("sl_price"))
    tp = _safe_float(signal_obj.get("tp") or signal_obj.get("take_profit") or signal_obj.get("tp_price"))
    return sl, tp


def _extract_position_units(signal_obj: Mapping[str, Any]) -> Optional[float]:
    return _safe_float(
        signal_obj.get("position_units")
        or signal_obj.get("position_size")
        or signal_obj.get("units")
        or signal_obj.get("qty")
        or signal_obj.get("quantity")
    )


def _extract_atr_for_symbol(current_cycle: Mapping[str, Any], symbol: str) -> Optional[float]:
    for key in ("latest_hf_atrs", "hf_atrs", "atrs", "atr_map"):
        atr_map = _mapping_or_empty(current_cycle.get(key))
        val = _safe_float(atr_map.get(symbol))
        if val is not None and val > 0:
            return val
    return None


# ---------------------------------------------------------------------------
# Kalman flip — replika logika live engine
# ---------------------------------------------------------------------------

def _compute_dcc_flip_multiplier(
    current_cycle: Mapping[str, Any],
    symbol: str,
    dcc_flip_eps_multiplier: float,
) -> float:
    """Replika: max(1.0, 1 + contagion_score * DCC_FLIP_EPS_MULTIPLIER)."""
    dcc_map = _mapping_or_empty(current_cycle.get("dcc_metrics"))
    contagion: Optional[float] = None

    if symbol in dcc_map:
        contagion = _safe_float(_mapping_or_empty(dcc_map[symbol]).get("contagion_score"))

    if contagion is None:
        for gdata in dcc_map.values():
            sc = _safe_float(_mapping_or_empty(gdata).get("contagion_score"))
            if sc is not None:
                contagion = sc
                break

    return max(1.0, 1.0 + (contagion or 0.0) * dcc_flip_eps_multiplier)


def _check_kalman_flip(
    bar_trend: Optional[str],
    bar_z: Optional[float],
    direction: int,
    flip_threshold: float,
) -> bool:
    """Replika: ((is_buy and trend=="DOWN") or (is_sell and trend=="UP")) and z >= threshold."""
    if bar_trend is None or bar_z is None:
        return False
    trend = str(bar_trend).upper()
    is_buy  = direction == 1
    is_sell = direction == -1
    return ((is_buy and trend == "DOWN") or (is_sell and trend == "UP")) and bar_z >= flip_threshold


# ---------------------------------------------------------------------------
# Dynamic SL/TP — replika logika live engine
# ---------------------------------------------------------------------------

def _update_dynamic_sl_tp(
    state: _PositionState,
    direction: int,
    current_price: float,
    atr: Optional[float],
    deviation_score: Optional[float],
    k_atr_stop: float = 1.5,
    rls_scaling_factor_sl: float = 0.5,
    rls_sl_max_multiplier: float = 2.0,
) -> _PositionState:
    """Trailing SL update berbasis ATR + RLS deviation — replika live engine."""
    if atr is None or atr <= 0.0:
        return state
    dev = deviation_score if deviation_score is not None else 0.0
    increase_factor = 1.0 + dev * rls_scaling_factor_sl
    k_adj   = min(k_atr_stop * increase_factor, k_atr_stop * rls_sl_max_multiplier)
    sl_dist = k_adj * atr

    new_state = state.copy()
    if direction == 1:
        candidate = current_price - sl_dist
        new_state.sl_price = max(candidate, state.sl_price) if state.sl_price is not None else candidate
    elif direction == -1:
        candidate = current_price + sl_dist
        new_state.sl_price = min(candidate, state.sl_price) if state.sl_price is not None else candidate
    return new_state


# ---------------------------------------------------------------------------
# PositionSimulator
# ---------------------------------------------------------------------------

class PositionSimulator:
    """Simulasi satu posisi terbuka melalui bar-bar M1.

    Urutan pemeriksaan per bar (mirror live engine):
    1. Update SL/TP dinamis (jika interval terpenuhi).
    2. Cek Kalman flip.
    3. Cek SL/TP hit (OHLC intrabar).
    4. Cek max_holding_bars.
    """

    def __init__(
        self,
        *,
        symbol: str,
        entry_price_raw: float,
        direction: int,
        sl_price: Optional[float],
        tp_price: Optional[float],
        adapter: IntrabarDataAdapter,
        entry_timestamp: Any,
        max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
        kalman_flip_zscore: float = DEFAULT_KALMAN_FLIP_ZSCORE,
        dcc_flip_multiplier: float = 1.0,
        dynamic_sltp_interval: int = DEFAULT_DYNAMIC_SLTP_UPDATE_INTERVAL,
        atr: Optional[float] = None,
        deviation_score: Optional[float] = None,
    ) -> None:
        self.symbol               = symbol
        self.entry_price_raw      = entry_price_raw
        self.direction            = direction
        self.adapter              = adapter
        self.entry_timestamp      = entry_timestamp
        self.max_holding_bars     = max_holding_bars
        self.flip_threshold       = kalman_flip_zscore * dcc_flip_multiplier
        self.dynamic_sltp_interval= dynamic_sltp_interval
        self.atr                  = atr
        self.deviation_score      = deviation_score
        self._state               = _PositionState(sl_price=sl_price, tp_price=tp_price)

    def run(self) -> _PositionResult:
        intrabar_path: List[Dict[str, Any]] = []
        bars_held       = 0
        last_bar: Optional[Dict[str, Any]] = None
        kalman_flip_bar: Optional[int]     = None

        for bar in self.adapter.bars_after(self.entry_timestamp):
            bars_held += 1
            bar_ts    = _normalize_timestamp(bar.get("Timestamp"))
            bar_open  = _safe_float(bar.get("Open"))
            bar_high  = _safe_float(bar.get("High"))
            bar_low   = _safe_float(bar.get("Low"))
            bar_close = _safe_float(bar.get("Close"))
            mid_price = bar_close or bar_open or self.entry_price_raw

            # 1. Dynamic SL/TP update
            if (
                self.dynamic_sltp_interval > 0
                and bars_held % self.dynamic_sltp_interval == 0
                and mid_price is not None
            ):
                self._state = _update_dynamic_sl_tp(
                    state           = self._state,
                    direction       = self.direction,
                    current_price   = mid_price,
                    atr             = self.atr,
                    deviation_score = self.deviation_score,
                )

            intrabar_path.append({
                "bar_index": bars_held,
                "timestamp": bar_ts,
                "open":      bar_open,
                "high":      bar_high,
                "low":       bar_low,
                "close":     bar_close,
                "sl_active": self._state.sl_price,
                "tp_active": self._state.tp_price,
            })

            # 2. Kalman flip
            bar_trend = bar.get("kalman_trend") or bar.get("trend")
            bar_kz    = _safe_float(
                bar.get("kalman_zscore")
                or bar.get("innovation_zscore")
                or bar.get("kalman_z")
            )
            if _check_kalman_flip(bar_trend, bar_kz, self.direction, self.flip_threshold):
                kalman_flip_bar = bars_held
                exit_px = bar_open if bar_open is not None else mid_price
                return _PositionResult(
                    exit_price_raw  = exit_px,
                    exit_timestamp  = bar_ts,
                    exit_reason     = EXIT_KALMAN_FLIP,
                    bars_held       = bars_held,
                    open_at_end     = False,
                    final_sl        = self._state.sl_price,
                    final_tp        = self._state.tp_price,
                    kalman_flip_bar = kalman_flip_bar,
                    intrabar_path   = intrabar_path,
                )

            # 3. SL/TP hit
            sl_hit, tp_hit = self._check_sl_tp(bar_open, bar_high, bar_low)
            if sl_hit and tp_hit:
                sl_px = self._state.sl_price
                tp_px = self._state.tp_price
                if sl_px is not None and tp_px is not None and bar_open is not None:
                    if abs(bar_open - sl_px) <= abs(bar_open - tp_px):
                        return self._make(sl_px, bar_ts, EXIT_SL, bars_held, intrabar_path)
                    return self._make(tp_px, bar_ts, EXIT_TP, bars_held, intrabar_path)
                return self._make(
                    sl_px if sl_px is not None else (bar_close or self.entry_price_raw),
                    bar_ts, EXIT_SL, bars_held, intrabar_path,
                )
            if sl_hit:
                return self._make(
                    self._state.sl_price if self._state.sl_price is not None
                    else (bar_close or self.entry_price_raw),
                    bar_ts, EXIT_SL, bars_held, intrabar_path,
                )
            if tp_hit:
                return self._make(
                    self._state.tp_price if self._state.tp_price is not None
                    else (bar_close or self.entry_price_raw),
                    bar_ts, EXIT_TP, bars_held, intrabar_path,
                )

            last_bar = bar

            # 4. Max holding bars
            if bars_held >= self.max_holding_bars:
                return _PositionResult(
                    exit_price_raw  = bar_close or self.entry_price_raw,
                    exit_timestamp  = bar_ts,
                    exit_reason     = EXIT_MAX_BARS,
                    bars_held       = bars_held,
                    open_at_end     = False,
                    final_sl        = self._state.sl_price,
                    final_tp        = self._state.tp_price,
                    kalman_flip_bar = None,
                    intrabar_path   = intrabar_path,
                )

        # Data M1 habis
        if last_bar is not None:
            lc = _safe_float(last_bar.get("close")) or self.entry_price_raw
            lt = _normalize_timestamp(last_bar.get("timestamp"))
        else:
            lc = self.entry_price_raw
            lt = _normalize_timestamp(self.entry_timestamp)

        return _PositionResult(
            exit_price_raw  = lc,
            exit_timestamp  = lt,
            exit_reason     = EXIT_OPEN_AT_END,
            bars_held       = bars_held,
            open_at_end     = True,
            final_sl        = self._state.sl_price,
            final_tp        = self._state.tp_price,
            kalman_flip_bar = None,
            intrabar_path   = intrabar_path,
        )

    def _check_sl_tp(
        self,
        bar_open: Optional[float],
        bar_high: Optional[float],
        bar_low: Optional[float],
    ) -> Tuple[bool, bool]:
        sl, tp = self._state.sl_price, self._state.tp_price
        sl_hit = tp_hit = False
        if self.direction == 1:
            if sl is not None and bar_low  is not None: sl_hit = bar_low  <= sl
            if tp is not None and bar_high is not None: tp_hit = bar_high >= tp
        else:
            if sl is not None and bar_high is not None: sl_hit = bar_high >= sl
            if tp is not None and bar_low  is not None: tp_hit = bar_low  <= tp
        return sl_hit, tp_hit

    def _make(
        self,
        exit_price_raw: float,
        exit_timestamp: Any,
        exit_reason: str,
        bars_held: int,
        intrabar_path: List[Dict[str, Any]],
    ) -> _PositionResult:
        return _PositionResult(
            exit_price_raw  = exit_price_raw,
            exit_timestamp  = exit_timestamp,
            exit_reason     = exit_reason,
            bars_held       = bars_held,
            open_at_end     = False,
            final_sl        = self._state.sl_price,
            final_tp        = self._state.tp_price,
            kalman_flip_bar = None,
            intrabar_path   = intrabar_path,
        )


# ---------------------------------------------------------------------------
# Price resolution
# ---------------------------------------------------------------------------

def _resolve_entry_price(
    signal_obj: Mapping[str, Any],
    current_prices: Mapping[str, Any],
    symbol: str,
) -> Tuple[Optional[float], Optional[str], bool]:
    explicit = _safe_float(signal_obj.get("entry_price"))
    fallback = _safe_float(current_prices.get(symbol))
    if explicit is not None:
        return explicit, "signal", False
    if fallback is not None:
        return fallback, "latest_actual_prices", True
    return None, None, False


def _compute_effective_prices(
    entry_price_raw: float,
    exit_price_raw: float,
    direction: int,
    side_cost_rate: float,
) -> _ExecutionPrices:
    return _ExecutionPrices(
        entry_price_raw       = entry_price_raw,
        entry_price_effective = entry_price_raw * (1.0 + direction * side_cost_rate),
        exit_price_raw        = exit_price_raw,
        exit_price_effective  = exit_price_raw  * (1.0 - direction * side_cost_rate),
    )


# ---------------------------------------------------------------------------
# ML labels — multi-horizon outcomes
# ---------------------------------------------------------------------------

def _compute_horizon_outcomes(
    context: "_CycleContext",
    entry_price_raw: Optional[float],
    evaluation_direction: int,
    intrabar_path: Optional[List[Dict[str, Any]]] = None,
) -> _HorizonOutcome:
    labels: Dict[str, Any] = {
        "pnl_1": None, "pnl_3": None, "pnl_5": None,
        "max_adverse": None, "max_favorable": None,
        "hit_1": None, "hit_3": None, "hit_5": None,
        "t_profit": None,
    }
    price_path: List[Dict[str, Any]] = []

    if evaluation_direction == 0 or entry_price_raw is None:
        return _HorizonOutcome(labels=labels, price_path=price_path)

    scr = (context.fee_bps + context.slippage_bps) / 10_000.0
    returns: Dict[int, float] = {}

    if intrabar_path:
        for horizon in context.horizons:
            idx = horizon - 1
            bar = intrabar_path[idx] if idx < len(intrabar_path) else intrabar_path[-1]
            fp  = _safe_float(bar.get("close"))
            if fp is None:
                continue
            ep  = _compute_effective_prices(entry_price_raw, fp, evaluation_direction, scr)
            pnl = evaluation_direction * (
                (ep.exit_price_effective - ep.entry_price_effective) / ep.entry_price_effective
            )
            returns[horizon] = pnl
            price_path.append({
                "horizon": horizon, "timestamp": bar.get("timestamp"),
                "exit_price_raw": fp, "pnl": pnl, "source": "intrabar_m1",
            })
    else:
        for horizon in context.horizons:
            fi = context.cycle_index + horizon
            if fi >= len(context.cycle_results):
                continue
            fc  = context.cycle_results[fi]
            fps = _first_present((_mapping_or_empty(fc),), PRICE_SOURCE_KEYS) or {}
            fp  = _safe_float(_mapping_or_empty(fps).get(context.symbol))
            if fp is None:
                continue
            ep  = _compute_effective_prices(entry_price_raw, fp, evaluation_direction, scr)
            pnl = evaluation_direction * (
                (ep.exit_price_effective - ep.entry_price_effective) / ep.entry_price_effective
            )
            returns[horizon] = pnl
            price_path.append({
                "horizon": horizon,
                "timestamp": _normalize_timestamp(fc.get("timestamp")),
                "exit_price_raw": fp, "pnl": pnl, "source": "cycle_results",
            })

    for h in context.horizons:
        fn = f"pnl_{h}"
        if fn in labels:
            labels[fn]       = returns.get(h)
            labels[f"hit_{h}"] = (returns[h] > 0.0) if h in returns else None

    obs = list(returns.values())
    if obs:
        labels["max_adverse"]   = min(obs)
        labels["max_favorable"] = max(obs)
        labels["t_profit"]      = any(v > 0.0 for v in obs)

    return _HorizonOutcome(labels=labels, price_path=price_path)


# ---------------------------------------------------------------------------
# Decision row template
# ---------------------------------------------------------------------------

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
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    position_units: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "schema_version":           SCHEMA_VERSION,
        "decision_id":              decision_id,
        "timestamp":                _normalize_timestamp(timestamp),
        "next_timestamp":           _normalize_timestamp(next_timestamp),
        "symbol":                   symbol,
        "action":                   action,
        "preferred_action":         preferred_action,
        "direction":                direction,
        "preferred_direction":      preferred_direction,
        "signal_generated":         preferred_direction != 0,
        "passed_all_gates":         False,
        "actually_executed":        False,
        "open_at_end":              False,
        "blocked_by":               [],
        "gate_results":             {},
        "gate_pass_mask":           [],
        "gate_fields":              [],
        "can_buy":                  None,
        "can_sell":                 None,
        "rls_confidence":           None,
        "deviation_score":          None,
        "kalman_zscore":            None,
        "dcc_correlation":          None,
        "predicted_return":         None,
        "pred_var":                 None,
        "spread":                   None,
        "regime_label":             None,
        "feature_model":            {},
        "sl_price":                 sl_price,
        "tp_price":                 tp_price,
        "position_units":           position_units,
        "entry_price":              None,
        "exit_price":               None,
        "entry_price_raw":          None,
        "entry_price_effective":    None,
        "exit_price_raw":           None,
        "exit_price_effective":     None,
        "entry_price_source":       None,
        "exit_timestamp":           None,
        "exit_reason":              None,
        "final_sl":                 None,
        "final_tp":                 None,
        "bars_held":                None,
        "kalman_flip_bar":          None,
        "gross_return":             None,
        "cost_return":              None,
        "net_return":               None,
        "fee_bps":                  float(fee_bps),
        "slippage_bps":             float(slippage_bps),
        "transaction_cost_bps":     float(2.0 * (fee_bps + slippage_bps)),
        "skip_reason":              None,
        "used_entry_fallback":      False,
        "price_path":               [],
        "intrabar_path":            [],
        "pnl_1":                    None,
        "pnl_3":                    None,
        "pnl_5":                    None,
        "max_adverse":              None,
        "max_favorable":            None,
        "hit_1":                    None,
        "hit_3":                    None,
        "hit_5":                    None,
        "t_profit":                 None,
    }


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def _apply_gate_result(
    decision_row: MutableMapping[str, Any],
    gate_name: str,
    passed: bool,
) -> None:
    decision_row.setdefault("gate_results", {})[str(gate_name)] = bool(passed)
    if not passed:
        bl = decision_row.setdefault("blocked_by", [])
        if gate_name not in bl:
            bl.append(str(gate_name))


# ---------------------------------------------------------------------------
# Trade row builder
# ---------------------------------------------------------------------------

def _build_trade_row(
    context: _CycleContext,
    decision_row: Mapping[str, Any],
    execution_prices: _ExecutionPrices,
    gross_return: float,
    cost_return: float,
    net_return: float,
) -> Dict[str, Any]:
    return {
        "schema_version":           SCHEMA_VERSION,
        "decision_id":              decision_row["decision_id"],
        "timestamp":                decision_row["timestamp"],
        "next_timestamp":           decision_row["next_timestamp"],
        "symbol":                   context.symbol,
        "action":                   decision_row["action"],
        "preferred_action":         decision_row["preferred_action"],
        "direction":                decision_row["direction"],
        "preferred_direction":      decision_row["preferred_direction"],
        "entry_price":              execution_prices.entry_price_effective,
        "exit_price":               execution_prices.exit_price_effective,
        "entry_price_raw":          execution_prices.entry_price_raw,
        "entry_price_effective":    execution_prices.entry_price_effective,
        "exit_price_raw":           execution_prices.exit_price_raw,
        "exit_price_effective":     execution_prices.exit_price_effective,
        "entry_price_source":       decision_row["entry_price_source"],
        "used_entry_fallback":      decision_row["used_entry_fallback"],
        "sl_price":                 decision_row["sl_price"],
        "tp_price":                 decision_row["tp_price"],
        "final_sl":                 decision_row["final_sl"],
        "final_tp":                 decision_row["final_tp"],
        "position_units":           decision_row["position_units"],
        "exit_timestamp":           decision_row["exit_timestamp"],
        "exit_reason":              decision_row["exit_reason"],
        "bars_held":                decision_row["bars_held"],
        "kalman_flip_bar":          decision_row["kalman_flip_bar"],
        "open_at_end":              decision_row["open_at_end"],
        "gross_return":             gross_return,
        "cost_return":              cost_return,
        "net_return":               net_return,
        "fee_bps":                  float(context.fee_bps),
        "slippage_bps":             float(context.slippage_bps),
        "transaction_cost_bps":     float(2.0 * (context.fee_bps + context.slippage_bps)),
        "blocked_by":               list(decision_row["blocked_by"]),
        "gate_results":             dict(decision_row["gate_results"]),
        "rls_confidence":           decision_row["rls_confidence"],
        "deviation_score":          decision_row["deviation_score"],
        "kalman_zscore":            decision_row["kalman_zscore"],
        "dcc_correlation":          decision_row["dcc_correlation"],
        "predicted_return":         decision_row["predicted_return"],
        "pred_var":                 decision_row["pred_var"],
        "spread":                   decision_row["spread"],
        "regime_label":             decision_row["regime_label"],
        "pnl_1":                    decision_row["pnl_1"],
        "pnl_3":                    decision_row["pnl_3"],
        "pnl_5":                    decision_row["pnl_5"],
        "max_adverse":              decision_row["max_adverse"],
        "max_favorable":            decision_row["max_favorable"],
        "hit_1":                    decision_row["hit_1"],
        "hit_3":                    decision_row["hit_3"],
        "hit_5":                    decision_row["hit_5"],
        "t_profit":                 decision_row["t_profit"],
    }


# ---------------------------------------------------------------------------
# Context iterator
# ---------------------------------------------------------------------------

def _iter_cycle_contexts(
    cycle_results: Sequence[Mapping[str, Any]],
    fee_bps: float,
    slippage_bps: float,
    horizons: Tuple[int, ...],
    intrabar_adapters: Optional[Mapping[str, IntrabarDataAdapter]],
    max_holding_bars: int,
    kalman_flip_zscore: float,
    dcc_flip_eps_multiplier: float,
    dynamic_sltp_update_interval: int,
) -> Iterator[_CycleContext]:
    for ci in range(len(cycle_results) - 1):
        cc   = cycle_results[ci]
        nc   = cycle_results[ci + 1]
        ts   = cc.get("trade_signals", {}) or {}
        np_  = _first_present((_mapping_or_empty(nc),), PRICE_SOURCE_KEYS) or {}
        cp_  = _first_present((_mapping_or_empty(cc),), PRICE_SOURCE_KEYS) or {}

        for si, (symbol, signal_obj) in enumerate(ts.items()):
            yield _CycleContext(
                cycle_index                  = ci,
                signal_index                 = si,
                timestamp                    = cc.get("timestamp"),
                next_timestamp               = nc.get("timestamp"),
                symbol                       = symbol,
                signal_obj                   = _mapping_or_empty(signal_obj),
                current_cycle                = cc,
                cycle_results                = cycle_results,
                current_prices               = _mapping_or_empty(cp_),
                next_prices                  = _mapping_or_empty(np_),
                fee_bps                      = fee_bps,
                slippage_bps                 = slippage_bps,
                horizons                     = horizons,
                intrabar_adapter             = (intrabar_adapters or {}).get(symbol),
                max_holding_bars             = max_holding_bars,
                kalman_flip_zscore           = kalman_flip_zscore,
                dcc_flip_eps_multiplier      = dcc_flip_eps_multiplier,
                dynamic_sltp_update_interval = dynamic_sltp_update_interval,
            )


# ---------------------------------------------------------------------------
# Core row builder
# ---------------------------------------------------------------------------

def _build_rows_for_context(
    context: _CycleContext,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    action              = str(context.signal_obj.get("signal", "HOLD")).upper()
    preferred_action    = _extract_preferred_action(
        context.signal_obj, context.current_cycle, context.symbol, action
    )
    direction           = _signal_direction(action)
    preferred_direction = _signal_direction(preferred_action)
    decision_id         = _decision_id(
        context.cycle_index, context.signal_index, context.timestamp, context.symbol
    )
    sl_price, tp_price  = _extract_sl_tp(context.signal_obj)
    position_units      = _extract_position_units(context.signal_obj)

    decision_row = _base_decision_row(
        decision_id=decision_id, timestamp=context.timestamp,
        next_timestamp=context.next_timestamp, symbol=context.symbol,
        action=action, preferred_action=preferred_action,
        direction=direction, preferred_direction=preferred_direction,
        fee_bps=context.fee_bps, slippage_bps=context.slippage_bps,
        sl_price=sl_price, tp_price=tp_price, position_units=position_units,
    )

    feature_snapshot = _extract_feature_snapshot(
        context.signal_obj, context.current_cycle, context.symbol
    )
    decision_row.update(feature_snapshot)
    decision_row["feature_model"] = feature_snapshot.copy()

    gate_results, blocked_by, can_buy, can_sell = _extract_gate_evidence(
        context.signal_obj, context.current_cycle, context.symbol
    )
    decision_row["blocked_by"] = list(blocked_by)
    decision_row["can_buy"]    = can_buy  if can_buy  is not None else (preferred_direction >= 0)
    decision_row["can_sell"]   = can_sell if can_sell is not None else (preferred_direction <= 0)
    for gn, passed in gate_results.items():
        _apply_gate_result(decision_row, gn, passed)
    _apply_gate_result(decision_row, "signal_present", preferred_direction != 0)

    if preferred_direction == 0:
        decision_row["skip_reason"] = "hold_signal"
        return decision_row, None

    entry_price_raw, entry_price_source, used_entry_fallback = _resolve_entry_price(
        context.signal_obj, context.current_prices, context.symbol
    )
    _apply_gate_result(decision_row, "entry_price_available", entry_price_raw is not None)
    decision_row["entry_price_raw"]     = entry_price_raw
    decision_row["entry_price_source"]  = entry_price_source
    decision_row["used_entry_fallback"] = used_entry_fallback

    # ── Intrabar simulation ─────────────────────────────────────────────────
    intrabar_path: List[Dict[str, Any]] = []

    if context.intrabar_adapter is not None and entry_price_raw is not None:
        atr             = _extract_atr_for_symbol(context.current_cycle, context.symbol)
        deviation_score = _safe_float(
            feature_snapshot.get("deviation_score")
            or _mapping_or_empty(context.current_cycle.get("parameter_deviations")).get(context.symbol)
        )
        dcc_mult = _compute_dcc_flip_multiplier(
            context.current_cycle, context.symbol, context.dcc_flip_eps_multiplier
        )
        sim = PositionSimulator(
            symbol                = context.symbol,
            entry_price_raw       = entry_price_raw,
            direction             = preferred_direction,
            sl_price              = sl_price,
            tp_price              = tp_price,
            adapter               = context.intrabar_adapter,
            entry_timestamp       = context.timestamp,
            max_holding_bars      = context.max_holding_bars,
            kalman_flip_zscore    = context.kalman_flip_zscore,
            dcc_flip_multiplier   = dcc_mult,
            dynamic_sltp_interval = context.dynamic_sltp_update_interval,
            atr                   = atr,
            deviation_score       = deviation_score,
        )
        pr             = sim.run()
        intrabar_path  = pr.intrabar_path
        exit_price_raw = pr.exit_price_raw
        exit_timestamp = pr.exit_timestamp
        exit_reason    = pr.exit_reason
        bars_held      = pr.bars_held
        open_at_end    = pr.open_at_end
        final_sl       = pr.final_sl
        final_tp       = pr.final_tp
        kfb            = pr.kalman_flip_bar
    else:
        exit_price_raw = _safe_float(context.next_prices.get(context.symbol))
        exit_timestamp = context.next_timestamp
        exit_reason    = EXIT_NEXT_BAR if exit_price_raw is not None else None
        bars_held      = 1 if exit_price_raw is not None else 0
        open_at_end    = False
        final_sl       = sl_price
        final_tp       = tp_price
        kfb            = None

    _apply_gate_result(decision_row, "exit_price_available", exit_price_raw is not None)
    decision_row.update({
        "exit_price_raw":   exit_price_raw,
        "exit_timestamp":   _normalize_timestamp(exit_timestamp),
        "exit_reason":      exit_reason,
        "bars_held":        bars_held,
        "open_at_end":      open_at_end,
        "final_sl":         final_sl,
        "final_tp":         final_tp,
        "kalman_flip_bar":  kfb,
        "intrabar_path":    intrabar_path,
    })

    outcome = _compute_horizon_outcomes(
        context, entry_price_raw, preferred_direction, intrabar_path or None
    )
    decision_row.update(outcome.labels)
    decision_row["price_path"] = outcome.price_path

    if entry_price_raw is None:
        decision_row["skip_reason"] = "missing_entry_price"
        return decision_row, None

    if blocked_by or any(
        not p for gn, p in decision_row["gate_results"].items()
        if gn not in BASE_GATE_FIELDS
    ):
        decision_row["skip_reason"] = "blocked_by_gate"
        return decision_row, None

    if exit_price_raw is None:
        decision_row["skip_reason"] = "missing_exit_price"
        return decision_row, None

    if direction == 0:
        decision_row["skip_reason"] = "hold_signal"
        return decision_row, None

    if open_at_end:
        decision_row["skip_reason"] = EXIT_OPEN_AT_END
        return decision_row, None

    scr              = (context.fee_bps + context.slippage_bps) / 10_000.0
    execution_prices = _compute_effective_prices(entry_price_raw, exit_price_raw, direction, scr)
    gross_return     = direction * (
        (execution_prices.exit_price_raw - execution_prices.entry_price_raw)
        / execution_prices.entry_price_raw
    )
    effective_return = direction * (
        (execution_prices.exit_price_effective - execution_prices.entry_price_effective)
        / execution_prices.entry_price_effective
    )
    cost_return = gross_return - effective_return
    net_return  = gross_return - cost_return

    _apply_gate_result(decision_row, "execution_ready", True)
    decision_row.update({
        "passed_all_gates":      not decision_row["blocked_by"] and all(decision_row["gate_results"].values()),
        "actually_executed":     True,
        "entry_price":           execution_prices.entry_price_effective,
        "exit_price":            execution_prices.exit_price_effective,
        "entry_price_effective": execution_prices.entry_price_effective,
        "exit_price_effective":  execution_prices.exit_price_effective,
        "skip_reason":           None,
        "gross_return":          gross_return,
        "cost_return":           cost_return,
        "net_return":            net_return,
    })
    return decision_row, _build_trade_row(
        context, decision_row, execution_prices, gross_return, cost_return, net_return
    )


# ---------------------------------------------------------------------------
# Gate schema finalization
# ---------------------------------------------------------------------------

def _finalize_gate_schema(decision_ledger: List[MutableMapping[str, Any]]) -> List[str]:
    ordered = list(BASE_GATE_FIELDS)
    for row in decision_ledger:
        for gn in row.get("gate_results", {}):
            if gn not in ordered:
                ordered.append(gn)
    for row in decision_ledger:
        gr = row.get("gate_results", {})
        row["gate_fields"]    = list(ordered)
        row["gate_pass_mask"] = [1 if gr.get(g, True) else 0 for g in ordered]
    return ordered


# ---------------------------------------------------------------------------
# Gate attribution
# ---------------------------------------------------------------------------

def summarize_gate_attribution(
    decision_ledger: Iterable[Mapping[str, Any]],
    horizon_field: str = "pnl_1",
) -> List[Dict[str, Any]]:
    per_gate: Dict[str, List[float]] = {}
    eval_cnt: Dict[str, int]         = {}
    unlock_cnt: Dict[str, int]       = {}

    for row in decision_ledger:
        pd_  = int(row.get("preferred_direction") or 0)
        bb   = list(row.get("blocked_by") or [])
        lv   = row.get(horizon_field)
        cand = float(lv) if lv is not None else 0.0
        act  = float(row["net_return"]) if row.get("actually_executed") else 0.0

        for gn in row.get("gate_fields", []):
            if gn in BASE_GATE_FIELDS:
                continue
            eval_cnt[gn] = eval_cnt.get(gn, 0) + 1
            wp = act
            if pd_ != 0 and gn in bb and len(bb) == 1:
                wp = cand
                unlock_cnt[gn] = unlock_cnt.get(gn, 0) + 1
            per_gate.setdefault(gn, []).append(wp - act)

    rows: List[Dict[str, Any]] = []
    for gn in sorted(per_gate):
        imps  = per_gate[gn]
        simps = sorted(imps)
        p95   = max(0, math.ceil(0.95 * len(simps)) - 1)
        rows.append({
            "gate":                  gn,
            "horizon_field":         horizon_field,
            "observations":          len(imps),
            "evaluated_decisions":   eval_cnt.get(gn, 0),
            "unblocked_trade_count": unlock_cnt.get(gn, 0),
            "mean_impact":           sum(imps) / len(imps),
            "median_impact":         simps[len(simps) // 2],
            "tail_impact_p95":       simps[p95],
            "positive_impact_rate":  sum(1 for i in imps if i > 0.0) / len(imps),
        })
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_replay_ledgers(
    cycle_results: List[Dict[str, Any]],
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    equity_curve_mode: str = DEFAULT_EQUITY_CURVE_MODE,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    mtf_base_dfs: Optional[Dict[str, Any]] = None,
    max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
    kalman_flip_zscore: float = DEFAULT_KALMAN_FLIP_ZSCORE,
    dcc_flip_eps_multiplier: float = DEFAULT_DCC_FLIP_EPS_MULTIPLIER,
    dynamic_sltp_update_interval: int = DEFAULT_DYNAMIC_SLTP_UPDATE_INTERVAL,
) -> ReplayResult:
    """Build row-level decision dan trade ledgers dengan simulasi intrabar penuh.

    Parameters
    ----------
    cycle_results                : output loop monitoring live engine
    fee_bps                      : biaya per sisi dalam basis poin
    slippage_bps                 : slippage per sisi dalam basis poin
    equity_curve_mode            : ``"additive"`` atau ``"compounding"``
    horizons                     : horizon label ML (default 1, 3, 5 bar M1)
    mtf_base_dfs                 : Dict[symbol, List[Dict]] data M1.
                                   None → legacy one-bar mode.
    max_holding_bars             : batas bar M1 sebelum force-close
    kalman_flip_zscore           : threshold z Kalman flip (live: KALMAN_FLIP_ZSCORE=3.0)
    dcc_flip_eps_multiplier      : DCC multiplier (live: DCC_FLIP_EPS_MULTIPLIER=0.5)
    dynamic_sltp_update_interval : update SL/TP setiap N bar M1 (0=disabled)
    """
    if equity_curve_mode not in SUPPORTED_EQUITY_CURVE_MODES:
        raise ValueError(
            f"equity_curve_mode harus salah satu dari: "
            f"{', '.join(sorted(SUPPORTED_EQUITY_CURVE_MODES))}."
        )
    norm_h = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    if not norm_h:
        raise ValueError("horizons harus berisi setidaknya satu horizon positif.")

    adapters: Optional[Dict[str, IntrabarDataAdapter]] = None
    if mtf_base_dfs:
        adapters = {sym: IntrabarDataAdapter(d) for sym, d in mtf_base_dfs.items()}

    sim_mode = "intrabar_kalman_sltp" if adapters else "one_bar_legacy"
    base_meta: Dict[str, Any] = {
        "schema_version":               SCHEMA_VERSION,
        "generated_at":                 datetime.now(timezone.utc).isoformat(),
        "cycle_count":                  len(cycle_results),
        "fee_bps":                      float(fee_bps),
        "slippage_bps":                 float(slippage_bps),
        "equity_curve_mode":            equity_curve_mode,
        "horizons":                     list(norm_h),
        "simulation_mode":              sim_mode,
        "max_holding_bars":             max_holding_bars,
        "kalman_flip_zscore":           kalman_flip_zscore,
        "dcc_flip_eps_multiplier":      dcc_flip_eps_multiplier,
        "dynamic_sltp_update_interval": dynamic_sltp_update_interval,
        "parallelizable_units":         ["symbol", "cycle_batch"],
    }

    if len(cycle_results) < 2:
        empty = BacktestSummary(
            total_trades=0, win_rate=0.0, gross_pnl=0.0,
            avg_pnl_per_trade=0.0, equity_curve_mode=equity_curve_mode,
        )
        return ReplayResult(
            summary=empty, decision_ledger=[], trade_ledger=[],
            metadata={
                **base_meta,
                "gate_fields": list(BASE_GATE_FIELDS),
                "gate_attribution_horizon": "pnl_1",
                "gate_attribution_summary": [],
                "hold_reason_summary": {},
                "blocked_by_summary": {},
                "exit_reason_summary": {},
            },
            gate_attribution=[],
        )

    decision_ledger: List[Dict[str, Any]] = []
    trade_ledger:    List[Dict[str, Any]] = []

    for ctx in _iter_cycle_contexts(
        cycle_results,
        fee_bps=fee_bps, slippage_bps=slippage_bps, horizons=norm_h,
        intrabar_adapters=adapters, max_holding_bars=max_holding_bars,
        kalman_flip_zscore=kalman_flip_zscore,
        dcc_flip_eps_multiplier=dcc_flip_eps_multiplier,
        dynamic_sltp_update_interval=dynamic_sltp_update_interval,
    ):
        dr, tr = _build_rows_for_context(ctx)
        decision_ledger.append(dr)
        if tr is not None:
            trade_ledger.append(tr)

    gate_fields = _finalize_gate_schema(decision_ledger)
    gate_attr   = summarize_gate_attribution(decision_ledger, horizon_field="pnl_1")

    hold_summary  = dict(sorted(Counter(
        r.get("skip_reason") for r in decision_ledger if r.get("skip_reason")
    ).items()))
    block_summary = dict(sorted(Counter(
        g for r in decision_ledger for g in r.get("blocked_by", [])
    ).items()))
    exit_summary  = dict(sorted(Counter(
        r.get("exit_reason") for r in trade_ledger if r.get("exit_reason")
    ).items()))

    summary = summarize_trade_ledger(trade_ledger, decision_ledger, equity_curve_mode)
    return ReplayResult(
        summary=summary,
        decision_ledger=decision_ledger,
        trade_ledger=trade_ledger,
        gate_attribution=gate_attr,
        metadata={
            **base_meta,
            "gate_fields":               gate_fields,
            "gate_attribution_horizon":  "pnl_1",
            "gate_attribution_summary":  gate_attr,
            "hold_reason_summary":       hold_summary,
            "blocked_by_summary":        block_summary,
            "exit_reason_summary":       exit_summary,
            "export_backend_candidates": ["pyarrow", "fastparquet", "polars"],
            "dataframe_api_note": (
                "Replay export tetap berbasis row records agar migrasi "
                "transformasi/parquet ke Polars bertahap."
            ),
        },
    )


# ---------------------------------------------------------------------------
# summarize_trade_ledger
# ---------------------------------------------------------------------------

def summarize_trade_ledger(
    trade_ledger: Iterable[Mapping[str, Any]],
    decision_ledger: Optional[Iterable[Mapping[str, Any]]] = None,
    equity_curve_mode: str = DEFAULT_EQUITY_CURVE_MODE,
) -> BacktestSummary:
    trows = list(trade_ledger)
    drows = list(decision_ledger or [])
    closed  = [r for r in trows if not r.get("open_at_end")]
    open_ae = len(trows) - len(closed)
    total   = len(closed)
    wins    = sum(1 for r in closed if float(r["net_return"]) > 0.0)
    gross   = sum(float(r["gross_return"]) for r in closed)
    net     = sum(float(r["net_return"])   for r in closed)
    curve   = _build_equity_curve(closed, mode=equity_curve_mode)
    mdd     = _compute_max_drawdown(curve, mode=equity_curve_mode)
    skipped = sum(
        1 for r in drows
        if r.get("signal_generated")
        and not r.get("actually_executed")
        and r.get("skip_reason") != EXIT_OPEN_AT_END
    )
    return BacktestSummary(
        total_trades=total, win_rate=(wins / total) if total else 0.0,
        gross_pnl=gross, avg_pnl_per_trade=(net / total) if total else 0.0,
        net_pnl=net, gross_return=gross, net_return=net,
        skipped_trades=skipped, open_at_end_count=open_ae,
        max_drawdown=mdd, equity_curve=curve, equity_curve_mode=equity_curve_mode,
    )


def _build_equity_curve(
    trade_ledger: Iterable[Mapping[str, Any]],
    mode: str = DEFAULT_EQUITY_CURVE_MODE,
) -> List[float]:
    curve: List[float] = []
    if mode == "additive":
        eq = 0.0
        for r in trade_ledger:
            eq += float(r["net_return"])
            curve.append(eq)
        return curve
    eq = 1.0
    for r in trade_ledger:
        eq *= 1.0 + float(r["net_return"])
        curve.append(eq)
    return curve


def _compute_max_drawdown(
    equity_curve: Iterable[float],
    mode: str = DEFAULT_EQUITY_CURVE_MODE,
) -> float:
    peak = 0.0 if mode == "additive" else 1.0
    mdd  = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) if mode == "additive" else (0.0 if peak == 0 else (peak - eq) / peak)
        mdd = max(mdd, dd)
    return mdd


# ---------------------------------------------------------------------------
# Backward-compatibility wrapper
# ---------------------------------------------------------------------------

def run_one_bar_replay_backtest(
    cycle_results: List[Dict[str, Any]],
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    equity_curve_mode: str = DEFAULT_EQUITY_CURVE_MODE,
) -> BacktestSummary:
    """Compatibility wrapper — no mtf_base_dfs, returns summary only."""
    return build_replay_ledgers(
        cycle_results, fee_bps=fee_bps,
        slippage_bps=slippage_bps, equity_curve_mode=equity_curve_mode,
    ).summary


# ---------------------------------------------------------------------------
# Parquet export
# ---------------------------------------------------------------------------

def export_replay_ledgers_to_parquet(
    replay_result: ReplayResult,
    output_dir: str,
    run_metadata: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, str]:
    """Persist decision_ledger + trade_ledger sebagai parquet + metadata JSON."""
    meta = {
        **replay_result.metadata,
        "schema_version": SCHEMA_VERSION,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        **(dict(run_metadata or {})),
    }
    td = Path(output_dir)
    td.mkdir(parents=True, exist_ok=True)
    dp = td / "decision_ledger.parquet"
    tp = td / "trade_ledger.parquet"
    mp = td / "metadata.json"
    _write_rows_to_parquet(_attach_export_metadata(replay_result.decision_ledger, meta), dp)
    _write_rows_to_parquet(_attach_export_metadata(replay_result.trade_ledger,    meta), tp)
    mp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {"decision_ledger": str(dp), "trade_ledger": str(tp), "metadata": str(mp)}


def _write_rows_to_parquet(rows: List[Mapping[str, Any]], path: Path) -> None:
    if find_spec("pyarrow"):
        pa = import_module("pyarrow")
        pq = import_module("pyarrow.parquet")
        pq.write_table(pa.Table.from_pylist(list(rows)), path)
        return
    if find_spec("fastparquet"):
        pd = import_module("pandas")
        pd.DataFrame(list(rows)).to_parquet(path, engine="fastparquet", index=False)
        return
    raise RuntimeError("Parquet export membutuhkan dependency 'pyarrow' atau 'fastparquet'.")


def _attach_export_metadata(
    rows: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    return [{**row, **metadata} for row in rows]
