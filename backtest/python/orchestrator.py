"""
python/orchestrator.py
──────────────────────
Python-side orchestrator that:
1. Converts ``cycle_results`` (live-engine format) → ``List[SignalInput dict]``
2. Loads per-symbol M1 DataFrames into ``PyFastEngine`` instances
3. Calls ``backtest.run_backtest`` (Rust core)
4. Returns a ``ReplayResult``-compatible dict

This file is the *only* Python code that runs at backtest time.
All hot-path logic (bar loop, SL/TP detection, label calculation) lives in Rust.

BOTTLENECK NOTES
────────────────
* Signal extraction is a single Python list-comprehension — O(N*S) but pure
  dict lookups; no pandas involved.
* Polars DataFrames are passed to PyFastEngine via Arrow IPC zero-copy path.
* The Rust layer returns plain Python dicts (via serde_json); the orchestrator
  can wrap them in polars/pandas for further analysis if needed.

Backward compatibility
──────────────────────
``build_replay_ledgers_fast`` mirrors the Python ``build_replay_ledgers``
signature exactly.  Pass ``use_rust=False`` to fall back to the original
Python implementation.
"""

from __future__ import annotations

import importlib
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence
import re

# ---------------------------------------------------------------------------
# Feature aliases — must stay in sync with Rust FEATURE_FIELD_ALIASES
# ---------------------------------------------------------------------------

_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "rls_confidence":   ("rls_confidence",  "confidence",       "rls_health"),
    "deviation_score":  ("deviation_score", "parameter_deviation", "deviation"),
    "kalman_zscore":    ("kalman_zscore",   "kalman_flip_zscore", "innovation_zscore"),
    "dcc_correlation":  ("dcc_correlation", "correlation",        "dcc_corr"),
    "predicted_return": ("predicted_return","forecast_return",    "expected_return"),
    "pred_var":         ("pred_var",        "prediction_variance","forecast_variance"),
    "spread":           ("spread",          "current_spread"),
    "regime_label":     ("regime_label",    "market_regime",      "regime"),
}

_PRICE_KEYS  = ("latest_actual_prices", "actual_prices", "prices")
_GATE_SKIP   = frozenset({
    "signal", "entry_price", "sl", "tp", "stop_loss", "take_profit",
    "position_units", "timestamp", "symbol", "direction",
    "feature_model", "features", "feature_snapshot", "model_features", "metrics",
    "action_mask", "gate_status", "gate_results", "gates",
    "blocked_by", "blocked_gates", "preferred_action", "raw_action",
    "intended_action", "suggested_action",
    "regime_label", "spread", "predicted_return", "pred_var",
    "rls_confidence", "deviation_score", "kalman_zscore", "dcc_correlation",
    "innovation_zscore", "trend", "contagion_score", "can_buy", "can_sell",
})

_VALID_GATE_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first(d: Mapping, keys: tuple) -> Any:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None

def _prices(cycle: Mapping) -> Mapping:
    for k in _PRICE_KEYS:
        v = cycle.get(k)
        if isinstance(v, Mapping):
            return v
    return {}

def _extract_gates(signal_obj: Mapping, current_cycle: Mapping, symbol: str) -> tuple[dict, list]:
    gate_results: dict[str, bool] = {}
    blocked_by: list[str] = []
    candidates: list[Mapping] = [signal_obj]
    for ck in ("features", "feature_model", "gate_status", "gate_results"):
        c = signal_obj.get(ck)
        if isinstance(c, Mapping):
            candidates.append(c)
    for ck in ("symbol_features", "symbol_metrics"):
        sc = current_cycle.get(ck, {}).get(symbol)
        if isinstance(sc, Mapping):
            candidates.append(sc)

    # blocked_by
    for cand in candidates:
        bb = cand.get("blocked_by") or cand.get("blocked_gates")
        if bb:
            if isinstance(bb, str):
                blocked_by = [bb] if bb else []
            elif isinstance(bb, (list, tuple)):
                blocked_by = [str(g) for g in bb if g]
            break

    # gate bool fields
    for cand in candidates:
        for k, v in cand.items():
            if k in _GATE_SKIP or k.startswith("_"):
                continue
            if not _VALID_GATE_NAME.match(k):   # ← TAMBAH: tolak key non-identifier
                continue
            if isinstance(v, bool):
                gate_results.setdefault(k, v)
            elif isinstance(v, int) and not isinstance(v, bool):
                # int 0/1 boleh; float dilarang (bisa NaN / junk values)
                gate_results.setdefault(k, bool(v))
    for gn in blocked_by:
        gate_results[gn] = False
    return gate_results, blocked_by


def _ts_to_epoch_ms(ts: Any) -> int:
    """Convert various timestamp formats to milliseconds since epoch (i64)."""
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1000)
    if isinstance(ts, str):
        try:
            import dateutil.parser
            dt = dateutil.parser.parse(ts)
            return int(dt.timestamp() * 1000)
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# cycle_results → List[SignalInput dict]
# ---------------------------------------------------------------------------

def extract_signals(
    cycle_results: Sequence[Mapping],
) -> list[dict]:
    """Convert cycle_results to a flat list of SignalInput dicts."""
    out: list[dict] = []
    for ci in range(len(cycle_results) - 1):
        cc = cycle_results[ci]
        nc = cycle_results[ci + 1]
        cp = _prices(cc)
        np_ = _prices(nc)
        ts_raw  = cc.get("timestamp")
        nts_raw = nc.get("timestamp")

        kalman_metrics     = cc.get("kalman_metrics", {})
        param_deviations   = cc.get("parameter_deviations", {})
        dcc_metrics        = cc.get("dcc_metrics", {})
        hf_atrs            = cc.get("latest_hf_atrs", cc.get("hf_atrs", {}))

        for si, (symbol, signal_obj) in enumerate(
            (cc.get("trade_signals") or {}).items()
        ):
            if not isinstance(signal_obj, Mapping):
                continue

            action = str(signal_obj.get("signal", "HOLD")).upper()
            km     = kalman_metrics.get(symbol, {})
            dcc_sym= dcc_metrics.get(symbol, {})
            dev_score = (
                signal_obj.get("deviation_score")
                or (param_deviations.get(symbol) if isinstance(param_deviations, Mapping)
                    else param_deviations if isinstance(param_deviations, (int, float)) else None)
            )

            gate_results, blocked_by = _extract_gates(signal_obj, cc, symbol)

            sig: dict = {
                "timestamp":       _ts_to_epoch_ms(ts_raw),
                "timestamp_str":   str(ts_raw) if ts_raw is not None else "",
                "cycle_index":     ci,
                "signal_index":    si,
                "next_timestamp":  str(nts_raw) if nts_raw is not None else None,
                "symbol":          symbol,
                "action":          action,
                "preferred_action":signal_obj.get("preferred_action"),
                "entry_price":     (
                    signal_obj.get("entry_price")
                    or cp.get(symbol)
                ),
                "exit_price_next": np_.get(symbol),
                "sl":              (signal_obj.get("sl") or signal_obj.get("stop_loss")),
                "tp":              (signal_obj.get("tp") or signal_obj.get("take_profit")),
                "position_units":  (
                    signal_obj.get("position_units")
                    or signal_obj.get("position_size")
                    or signal_obj.get("units")
                ),
                # Features
                "rls_confidence":   signal_obj.get("rls_confidence"),
                "deviation_score":  dev_score,
                "kalman_zscore":    signal_obj.get("kalman_zscore") or km.get("innovation_zscore"),
                "dcc_correlation":  signal_obj.get("dcc_correlation"),
                "dcc_contagion":    dcc_sym.get("contagion_score"),
                "predicted_return": signal_obj.get("predicted_return"),
                "pred_var":         signal_obj.get("pred_var"),
                "spread":           signal_obj.get("spread"),
                "regime_label":     signal_obj.get("regime_label"),
                "atr":              hf_atrs.get(symbol) if isinstance(hf_atrs, Mapping) else None,
                # Gates
                "gate_results":    gate_results,
                "blocked_by":      blocked_by,
                "can_buy":         signal_obj.get("can_buy"),
                "can_sell":        signal_obj.get("can_sell"),
            }
            out.append(sig)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_replay_ledgers_fast(
    cycle_results: list[dict],
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    equity_curve_mode: str = "additive",
    horizons: Sequence[int] = (1, 3, 5),
    mtf_base_dfs: Optional[Dict[str, Any]] = None,
    max_holding_bars: int = 500,
    kalman_flip_zscore: float = 3.0,
    dcc_flip_eps_multiplier: float = 0.5,
    dynamic_sltp_update_interval: int = 0,
    parallel: bool = True,
    use_rust: bool = True,
) -> dict:
    """
    High-performance backtest using the Rust core.

    Parameters mirror ``build_replay_ledgers`` in the original Python module.
    ``use_rust=False`` falls back to the pure-Python implementation.
    """
    if not use_rust:
        from backtest_helpers import build_replay_ledgers  # original Python module
        import dataclasses
        result = build_replay_ledgers(
            cycle_results,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            equity_curve_mode=equity_curve_mode, horizons=horizons,
            mtf_base_dfs=mtf_base_dfs, max_holding_bars=max_holding_bars,
            kalman_flip_zscore=kalman_flip_zscore,
            dcc_flip_eps_multiplier=dcc_flip_eps_multiplier,
            dynamic_sltp_update_interval=dynamic_sltp_update_interval,
        )
        return dataclasses.asdict(result)

    try:
        import backtest  # compiled Rust extension
    except ImportError as exc:
        raise ImportError(
            "backtest Rust extension not found.  "
            "Run `maturin develop --release` in the backtest directory, "
            "or pass use_rust=False to fall back to the Python implementation."
        ) from exc

    # ── Build Rust config ───────────────────────────────────────────────────
    cfg = backtest.BacktestConfig(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        horizons=list(horizons),
        max_holding_bars=max_holding_bars,
        kalman_flip_zscore=kalman_flip_zscore,
        dcc_flip_eps_multiplier=dcc_flip_eps_multiplier,
        dynamic_sltp_update_interval=dynamic_sltp_update_interval,
        equity_curve_mode=equity_curve_mode,
    )

    # ── Build per-symbol engines ────────────────────────────────────────────
    engines: dict[str, backtest_rs.PyFastEngine] = {}
    engine_errors: list[str] = []

    if mtf_base_dfs:
        for sym, raw in mtf_base_dfs.items():
            df = _to_polars(raw, sym)
            if df is None:
                engine_errors.append(f"{sym}: _to_polars gagal (input type={type(raw).__name__})")
                continue
            if df.height() == 0:
                engine_errors.append(f"{sym}: DataFrame kosong, skip engine")
                continue
            try:
                ipc_bytes = _df_to_ipc_bytes(df, sym)
                engines[sym] = backtest_rs.PyFastEngine(ipc_bytes, sym)
            except Exception as e:
                engine_errors.append(f"{sym}: {e}")

    if engine_errors:
        # Raise kalau SEMUA engine gagal — ini indikasi masalah sistemik
        if mtf_base_dfs and not engines:
            raise RuntimeError(
                "Semua M1 engine gagal dibangun:\n" + "\n".join(engine_errors)
            )
        # Sebagian gagal — warn tapi lanjut
        warnings.warn(
            f"{len(engine_errors)} engine gagal (dari {len(mtf_base_dfs)}):\n"
            + "\n".join(engine_errors),
            stacklevel=2,
        )

    # Log ringkasan engine yang berhasil
    if engines:
        print(f"[backtest_rs] {len(engines)}/{len(mtf_base_dfs or {})} engine M1 aktif: "
              f"{sorted(engines.keys())}")
    else:
        print("[backtest_rs] Tidak ada engine M1 — mode legacy one-bar aktif")


    # ── Extract signals ────────────────────────────────────────────────────
    signals = extract_signals(cycle_results)

    if not signals:
        return _empty_result(fee_bps, slippage_bps, equity_curve_mode, list(horizons))

    # ── Run ────────────────────────────────────────────────────────────────
    result = backtest.run_backtest(
        signals,
        engines,
        cfg,
        parallel=parallel,
        horizon_field="pnl_1",
    )

    result["metadata"] = {
        "schema_version":               "sprint3.v1",
        "generated_at":                 datetime.now(timezone.utc).isoformat(),
        "cycle_count":                  len(cycle_results),
        "fee_bps":                      fee_bps,
        "slippage_bps":                 slippage_bps,
        "equity_curve_mode":            equity_curve_mode,
        "horizons":                     sorted(set(int(h) for h in horizons if h > 0)),
        "simulation_mode":              "intrabar_kalman_sltp" if engines else "one_bar_legacy",
        "max_holding_bars":             max_holding_bars,
        "kalman_flip_zscore":           kalman_flip_zscore,
        "dcc_flip_eps_multiplier":      dcc_flip_eps_multiplier,
        "dynamic_sltp_update_interval": dynamic_sltp_update_interval,
        "rust_engine":                  True,
        "parallel":                     parallel,
        "gate_attribution_horizon":     "pnl_1",
        "gate_attribution_summary":     result.get("gate_attribution", []),
    }
    return result


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------

def _df_to_ipc_bytes(df: Any, symbol: str) -> bytes:
    """Serialize polars DataFrame → Arrow IPC bytes.

    Raises ValueError dengan pesan jelas jika df bukan polars DataFrame
    atau jika write_ipc gagal — agar caller bisa log dengan benar.
    """
    import io
    try:
        buf = io.BytesIO()
        df.write_ipc(buf)           # polars: write_ipc(file) → None
        result = buf.getvalue()
        if not result:
            raise ValueError(f"[{symbol}] write_ipc produced 0 bytes — DataFrame mungkin kosong")
        return result
    except AttributeError:
        raise ValueError(
            f"[{symbol}] Object bukan polars DataFrame (type={type(df).__name__}). "
            "Pastikan mtf_base_dfs berisi polars.DataFrame."
        )
    except Exception as exc:
        raise ValueError(f"[{symbol}] Gagal serialize ke IPC: {exc}") from exc


def _to_polars(data: Any, symbol: str):
    """Convert List[dict], pandas DataFrame, or polars DataFrame to polars DataFrame."""
    try:
        import polars as pl
    except ImportError:
        warnings.warn(f"polars not installed; skipping M1 engine for {symbol}")
        return None

    if isinstance(data, pl.DataFrame):
        return data

    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
    except ImportError:
        pass

    if isinstance(data, list) and data:
        return pl.DataFrame(data)

    return None


def _empty_result(fee_bps, slippage_bps, mode, horizons):
    return {
        "trade_ledger":    [],
        "decision_ledger": [],
        "summary": {
            "total_trades":      0, "win_rate": 0.0,
            "gross_pnl":         0.0, "avg_pnl_per_trade": 0.0,
            "net_pnl":           0.0, "gross_return": 0.0,
            "net_return":        0.0, "skipped_trades": 0,
            "open_at_end_count": 0,  "max_drawdown": 0.0,
            "equity_curve":      [], "equity_curve_mode": mode,
        },
        "gate_attribution": [],
        "metadata": {},
    }
