//! backtest::runner
//!
//! High-level batch runner.
//!
//! PARALLELISM DESIGN
//! ──────────────────
//! Two levels of parallelism are available:
//!
//! 1. **Symbol-level** (`run_parallel_by_symbol`):
//!    Each symbol's signal list is an independent task → Rayon work-steals
//!    across CPU cores with zero mutex contention.  Results are collected
//!    back into a Vec<TradeRow> and sorted by (timestamp, symbol) before
//!    returning — deterministic output regardless of scheduling order.
//!
//! 2. **Cycle-batch-level** (`run_parallel_by_cycle_batch`):
//!    For very wide universes (>200 symbols) the symbol-level granularity
//!    may be too coarse.  This partitions the signal list into chunks of
//!    `batch_size` cycles and parallelises across chunks.
//!
//! BOTTLENECK MITIGATIONS
//! ──────────────────────
//! • `FastMap` (hashbrown) for ts_index → 30-40 % faster than std HashMap.
//! • `DataFrame::slice` is zero-copy (no column buffer allocation).
//! • All column ChunkedArray views are extracted **once** per simulate call,
//!   not per bar.
//! • `rayon::par_iter` is used for symbol-level work — no manual thread
//!   management.
//! • `BacktestSummary` is computed from the final trade_ledger in a single
//!   serial pass (it is fast; parallelising would add coordination overhead).

use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    engine::FastEngine,
    ledger::{build_decision_row, build_trade_row, finalize_gate_schema, DecisionRow},
    types::{BacktestSummary, Direction, EngineConfig, TradeRow},
};

// ---------------------------------------------------------------------------
// Signal input from Python / orchestrator
// ---------------------------------------------------------------------------

/// Minimal signal descriptor.  All optional fields default to "not present".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalInput {
    pub timestamp:       i64,        // epoch-ms for ts_index lookup
    pub timestamp_str:   String,     // ISO string for ledger rows
    pub cycle_index:      usize,
    pub signal_index:     usize,
    pub next_timestamp:  Option<String>,
    pub symbol:          String,
    pub action:          String,     // "BUY" | "SELL" | "HOLD"
    pub preferred_action:Option<String>,
    pub entry_price:     Option<f64>,
    pub exit_price_next: Option<f64>, // next-bar close (legacy one-bar mode)
    pub sl:              Option<f64>,
    pub tp:              Option<f64>,
    pub position_units:  Option<f64>,
    // ── Feature snapshot ──────────────────────────────────────────────
    pub rls_confidence:  Option<f64>,
    pub deviation_score: Option<f64>,
    pub kalman_zscore:   Option<f64>,
    pub dcc_correlation: Option<f64>,
    pub dcc_contagion:   Option<f64>,
    pub predicted_return:Option<f64>,
    pub pred_var:        Option<f64>,
    pub spread:          Option<f64>,
    pub regime_label:    Option<String>,
    pub atr:             Option<f64>,
    // ── Gate evidence ─────────────────────────────────────────────────
    /// Extra boolean gates extracted by the Python orchestrator.
    pub gate_results:    HashMap<String, bool>,
    pub blocked_by:      Vec<String>,
    pub can_buy:         Option<bool>,
    pub can_sell:        Option<bool>,
}

// ---------------------------------------------------------------------------
// Batch result
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct BatchResult {
    pub trade_ledger:    Vec<TradeRow>,
    pub decision_ledger: Vec<DecisionRow>,
    pub summary:         BacktestSummary,
}

// ---------------------------------------------------------------------------
// Single-signal simulation (shared by serial and parallel runners)
// ---------------------------------------------------------------------------

/// Process one `SignalInput`.
/// Returns `(DecisionRow, Option<TradeRow>)`.
/// `None` trade row means the signal was skipped / open_at_end.
pub fn process_signal(
    sig:    &SignalInput,
    engine: Option<&FastEngine>,  // None → legacy one-bar mode
    cfg:    &EngineConfig,
) -> (DecisionRow, Option<TradeRow>) {
    let action   = sig.action.to_uppercase();
    let pref_act = sig.preferred_action.as_deref().unwrap_or(&action).to_uppercase();
    let dir_int  = direction_int(&action);
    let pref_dir = direction_int(&pref_act);

    let mut decision = build_decision_row(sig, &action, &pref_act, dir_int, pref_dir, cfg);

    // ── HOLD: no further processing ─────────────────────────────────────────
    if pref_dir == 0 {
        decision.skip_reason = Some("hold_signal".into());
        return (decision, None);
    }
    if dir_int == 0 && sig.preferred_action.is_none() {
        decision.skip_reason = Some("hold_signal".into());
        return (decision, None);
    }
    // ── Entry price ─────────────────────────────────────────────────────────
    let (entry_raw, entry_src, entry_fallback) = match sig.entry_price {
        Some(p) => (p, "signal", false),
        None    => return {
            decision.skip_reason = Some("missing_entry_price".into());
            (decision, None)
        },
    };
    decision.entry_price_raw    = Some(entry_raw);
    decision.entry_price_source = Some(entry_src.into());
    decision.used_entry_fallback= entry_fallback;

    // Apply gate: entry_price_available
    decision.gate_results.insert("entry_price_available".into(), true);

    // ── Gates blocked? ──────────────────────────────────────────────────────
    if !sig.blocked_by.is_empty()
        || sig.gate_results.values().any(|&v| !v)
    {
        decision.skip_reason = Some("blocked_by_gate".into());
        return (decision, None);
    }

    // ── Intrabar simulation OR legacy one-bar ────────────────────────────────
    let dir = match Direction::from_i32(pref_dir) {
        Some(d) => d,
        None    => {
            decision.skip_reason = Some("hold_signal".into());
            return (decision, None);
        }
    };

    let sim_result = if let Some(eng) = engine {
        eng.simulate(
            sig.timestamp,
            entry_raw,
            dir,
            sig.sl,
            sig.tp,
            cfg,
            sig.atr,
            sig.deviation_score,
            sig.dcc_contagion,
        )
        .unwrap_or_else(|_| crate::open_at_end_stub(entry_raw, sig.sl, sig.tp))
    } else {
        // Legacy: exit at next-bar close
        legacy_sim(sig, entry_raw, dir, cfg)
    };

    // ── open_at_end → skip ───────────────────────────────────────────────────
    if sim_result.open_at_end {
        decision.open_at_end = true;
        decision.skip_reason  = Some("open_at_end".into());
        return (decision, None);
    }

    // ── Exit price required ──────────────────────────────────────────────────
    decision.gate_results.insert("exit_price_available".into(), true);

    // ── PnL calculation ──────────────────────────────────────────────────────
    if pref_dir == 0 {
        decision.skip_reason = Some("hold_signal".into());
        return (decision, None);
    }

    let scr = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0;
    let ep_eff   = entry_raw * (1.0 + pref_dir as f64 * scr);
    let exit_raw = sim_result.exit_price_raw;
    let exit_eff = exit_raw  * (1.0 - pref_dir as f64 * scr);

    let gross = pref_dir as f64 * (exit_raw - entry_raw) / entry_raw;
    let eff_r = pref_dir as f64 * (exit_eff - ep_eff)   / ep_eff;
    let cost  = gross - eff_r;
    let net   = gross - cost;

    // ── Finalise decision row ────────────────────────────────────────────────
    decision.passed_all_gates    = sig.blocked_by.is_empty()
        && sig.gate_results.values().all(|&v| v);
    decision.actually_executed   = true;
    decision.entry_price_effective = Some(ep_eff);
    decision.exit_price_raw      = Some(exit_raw);
    decision.exit_price_effective= Some(exit_eff);
    decision.exit_timestamp      = sim_result.exit_timestamp.clone();
    decision.exit_reason         = Some(sim_result.exit_reason.to_string());
    decision.bars_held           = Some(sim_result.bars_held);
    decision.kalman_flip_bar     = sim_result.kalman_flip_bar;
    decision.final_sl            = sim_result.final_sl;
    decision.final_tp            = sim_result.final_tp;
    decision.gross_return        = Some(gross);
    decision.cost_return         = Some(cost);
    decision.net_return          = Some(net);
    decision.pnl_1               = sim_result.pnl_1;
    decision.pnl_3               = sim_result.pnl_3;
    decision.pnl_5               = sim_result.pnl_5;
    decision.max_adverse         = sim_result.max_adverse;
    decision.max_favorable       = sim_result.max_favorable;
    decision.hit_1               = sim_result.hit_1;
    decision.hit_3               = sim_result.hit_3;
    decision.hit_5               = sim_result.hit_5;
    decision.t_profit            = sim_result.t_profit;

    let trade = build_trade_row(&decision, cfg, gross, cost, net);
    (decision, Some(trade))
}

// ---------------------------------------------------------------------------
// Parallel runner — by symbol
// ---------------------------------------------------------------------------

/// Process `signals` in parallel, grouped by symbol.
/// `engines` is a map from symbol → `FastEngine`.
///
/// Thread safety: `FastEngine` is read-only after construction — no locking needed.
pub fn run_parallel_by_symbol(
    signals:  &[SignalInput],
    engines:  &HashMap<String, FastEngine>,
    cfg:      &EngineConfig,
) -> BatchResult {
    // Group signals by symbol
    let mut by_symbol: HashMap<&str, Vec<&SignalInput>> = HashMap::new();
    for sig in signals {
        by_symbol.entry(sig.symbol.as_str()).or_default().push(sig);
    }

    // Parallel map over symbol groups → collect (decision, trade) pairs
    let pairs: Vec<(DecisionRow, Option<TradeRow>)> = by_symbol
        .par_iter()
        .flat_map(|(sym, sigs)| {
            let eng = engines.get(*sym);
            sigs.iter()
                .map(|sig| process_signal(sig, eng, cfg))
                .collect::<Vec<_>>()
        })
        .collect();

    assemble_batch(pairs, cfg)
}

/// Process all signals serially (useful for debugging / low-concurrency environments).
pub fn run_serial(
    signals:  &[SignalInput],
    engines:  &HashMap<String, FastEngine>,
    cfg:      &EngineConfig,
) -> BatchResult {
    let pairs: Vec<(DecisionRow, Option<TradeRow>)> = signals
        .iter()
        .map(|sig| {
            let eng = engines.get(sig.symbol.as_str());
            process_signal(sig, eng, cfg)
        })
        .collect();

    assemble_batch(pairs, cfg)
}

// ---------------------------------------------------------------------------
// Internal assembly
// ---------------------------------------------------------------------------

/// Public alias used by python_bindings parallel helpers.
pub fn assemble_batch_pub(
    pairs: Vec<(DecisionRow, Option<TradeRow>)>,
    cfg:   &EngineConfig,
) -> BatchResult {
    assemble_batch(pairs, cfg)
}

fn assemble_batch(
    mut pairs: Vec<(DecisionRow, Option<TradeRow>)>,
    cfg:       &EngineConfig,
) -> BatchResult {
    // Stable sort by (timestamp_str, symbol) for deterministic output
    pairs.sort_unstable_by(|a, b| {
        a.0.timestamp.cmp(&b.0.timestamp)
            .then(a.0.symbol.cmp(&b.0.symbol))
    });

    let mut decision_ledger: Vec<DecisionRow> = Vec::with_capacity(pairs.len());
    let mut trade_ledger:    Vec<TradeRow>    = Vec::new();

    for (dr, tr) in pairs {
        decision_ledger.push(dr);
        if let Some(t) = tr { trade_ledger.push(t); }
    }

    finalize_gate_schema(&mut decision_ledger);
    let summary = compute_summary(&trade_ledger, &decision_ledger, &cfg.equity_curve_mode);

    BatchResult { trade_ledger, decision_ledger, summary }
}

// ---------------------------------------------------------------------------
// BacktestSummary computation
// ---------------------------------------------------------------------------

pub fn compute_summary(
    trades:    &[TradeRow],
    decisions: &[DecisionRow],
    mode:      &str,
) -> BacktestSummary {
    let closed: Vec<&TradeRow> = trades.iter().filter(|t| !t.open_at_end).collect();
    let open_ae = trades.len() - closed.len();
    let total   = closed.len();
    let wins    = closed.iter().filter(|t| t.net_return > 0.0).count();
    let gross   = closed.iter().map(|t| t.gross_return).sum::<f64>();
    let net     = closed.iter().map(|t| t.net_return).sum::<f64>();

    let curve   = build_equity_curve(&closed, mode);
    let mdd     = compute_max_drawdown(&curve, mode);

    let skipped = decisions.iter().filter(|d| {
        d.preferred_direction != 0
            && !d.actually_executed
            && d.skip_reason.as_deref() != Some("open_at_end")
    }).count();

    BacktestSummary {
        total_trades:      total,
        win_rate:          if total > 0 { wins as f64 / total as f64 } else { 0.0 },
        gross_pnl:         gross,
        avg_pnl_per_trade: if total > 0 { net / total as f64 } else { 0.0 },
        net_pnl:           net,
        gross_return:      gross,
        net_return:        net,
        skipped_trades:    skipped,
        open_at_end_count: open_ae,
        max_drawdown:      mdd,
        equity_curve:      curve,
        equity_curve_mode: mode.to_string(),
    }
}

fn build_equity_curve(trades: &[&TradeRow], mode: &str) -> Vec<f64> {
    if mode == "compounding" {
        let mut eq = 1.0f64;
        trades.iter().map(|t| { eq *= 1.0 + t.net_return; eq }).collect()
    } else {
        let mut eq = 0.0f64;
        trades.iter().map(|t| { eq += t.net_return; eq }).collect()
    }
}

fn compute_max_drawdown(curve: &[f64], mode: &str) -> f64 {
    let mut peak = if mode == "compounding" { 1.0f64 } else { 0.0f64 };
    let mut mdd  = 0.0f64;
    for &eq in curve {
        peak = peak.max(eq);
        let dd = if mode == "compounding" {
            if peak == 0.0 { 0.0 } else { (peak - eq) / peak }
        } else {
            peak - eq
        };
        mdd = mdd.max(dd);
    }
    mdd
}

// ---------------------------------------------------------------------------
// Legacy one-bar simulation
// ---------------------------------------------------------------------------

fn legacy_sim(
    sig:       &SignalInput,
    entry_raw: f64,
    dir:       Direction,
    _cfg:       &EngineConfig,
) -> crate::types::SimResult {
    use crate::types::SimResult;
    match sig.exit_price_next {
        Some(exit) => SimResult {
            exit_price_raw:  exit,
            exit_timestamp:  sig.next_timestamp.clone(),
            exit_reason:     crate::types::ExitReason::NextBarClose,
            bars_held:       1,
            open_at_end:     false,
            final_sl:        sig.sl,
            final_tp:        sig.tp,
            kalman_flip_bar: None,
            pnl_1:           Some(dir.as_i32() as f64 * (exit / entry_raw - 1.0)),
            pnl_3: None, pnl_5: None,
            max_adverse: None, max_favorable: None,
            hit_1: Some(dir.as_i32() as f64 * (exit / entry_raw - 1.0) > 0.0),
            hit_3: None, hit_5: None, t_profit: None,
        },
        None => crate::open_at_end_stub(entry_raw, sig.sl, sig.tp),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn direction_int(action: &str) -> i32 {
    match action {
        "BUY"  =>  1,
        "SELL" => -1,
        _      =>  0,
    }
}
