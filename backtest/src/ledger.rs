//! backtest_rs::ledger
//!
//! Row-building helpers that mirror the Python `_base_decision_row` and
//! `_build_trade_row` functions exactly — field names preserved for
//! pandas/Polars DF schema compatibility on the Python side.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    runner::SignalInput,
    types::{EngineConfig, TradeRow},
};

pub const SCHEMA_VERSION: &str = "sprint3.v1";

// ---------------------------------------------------------------------------
// DecisionRow — mirrors Python decision_ledger schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DecisionRow {
    pub schema_version:         &'static str,
    pub decision_id:            String,
    pub timestamp:              Option<String>,
    pub next_timestamp:         Option<String>,
    pub symbol:                 String,
    pub action:                 String,
    pub preferred_action:       String,
    pub direction:              i32,
    pub preferred_direction:    i32,
    pub signal_generated:       bool,
    pub passed_all_gates:       bool,
    pub actually_executed:      bool,
    pub open_at_end:            bool,
    pub blocked_by:             Vec<String>,
    pub gate_results:           HashMap<String, bool>,
    pub gate_pass_mask:         Vec<u8>,
    pub gate_fields:            Vec<String>,
    pub can_buy:                Option<bool>,
    pub can_sell:               Option<bool>,
    // Feature snapshot
    pub rls_confidence:         Option<f64>,
    pub deviation_score:        Option<f64>,
    pub kalman_zscore:          Option<f64>,
    pub dcc_correlation:        Option<f64>,
    pub predicted_return:       Option<f64>,
    pub pred_var:               Option<f64>,
    pub spread:                 Option<f64>,
    pub regime_label:           Option<String>,
    // Prices
    pub sl_price:               Option<f64>,
    pub tp_price:               Option<f64>,
    pub position_units:         Option<f64>,
    pub entry_price_raw:        Option<f64>,
    pub entry_price_effective:  Option<f64>,
    pub exit_price_raw:         Option<f64>,
    pub exit_price_effective:   Option<f64>,
    pub entry_price_source:     Option<String>,
    pub used_entry_fallback:    bool,
    // Exit info
    pub exit_timestamp:         Option<String>,
    pub exit_reason:            Option<String>,
    pub final_sl:               Option<f64>,
    pub final_tp:               Option<f64>,
    pub bars_held:              Option<usize>,
    pub kalman_flip_bar:        Option<usize>,
    // Returns
    pub gross_return:           Option<f64>,
    pub cost_return:            Option<f64>,
    pub net_return:             Option<f64>,
    pub fee_bps:                f64,
    pub slippage_bps:           f64,
    pub transaction_cost_bps:   f64,
    pub skip_reason:            Option<String>,
    // ML labels
    pub pnl_1:                  Option<f64>,
    pub pnl_3:                  Option<f64>,
    pub pnl_5:                  Option<f64>,
    pub max_adverse:            Option<f64>,
    pub max_favorable:          Option<f64>,
    pub hit_1:                  Option<bool>,
    pub hit_3:                  Option<bool>,
    pub hit_5:                  Option<bool>,
    pub t_profit:               Option<bool>,
}

// ---------------------------------------------------------------------------
// build_decision_row
// ---------------------------------------------------------------------------

pub fn build_decision_row(
    sig:      &SignalInput,
    action:   &str,
    pref_act: &str,
    dir:      i32,
    pref_dir: i32,
    cfg:      &EngineConfig,
) -> DecisionRow {
    let tcost = 2.0 * (cfg.fee_bps + cfg.slippage_bps);

    // Mirror Python _decision_id: "{ts}:{cycle_index}:{signal_index}:{symbol}"
    let did = format!(
        "{}:{}:{}:{}",
        sig.timestamp_str, sig.cycle_index, sig.signal_index, sig.symbol
    );

    let mut gr = sig.gate_results.clone();
    gr.insert("signal_present".into(), pref_dir != 0);

    DecisionRow {
        schema_version:       SCHEMA_VERSION,
        decision_id:          did,
        timestamp:            Some(sig.timestamp_str.clone()),
        next_timestamp:       sig.next_timestamp.clone(),
        symbol:               sig.symbol.clone(),
        action:               action.to_string(),
        preferred_action:     pref_act.to_string(),
        direction:            dir,
        preferred_direction:  pref_dir,
        signal_generated:     pref_dir != 0,
        blocked_by:           sig.blocked_by.clone(),
        gate_results:         gr,
        can_buy:              sig.can_buy.or(Some(pref_dir >= 0)),
        can_sell:             sig.can_sell.or(Some(pref_dir <= 0)),
        rls_confidence:       sig.rls_confidence,
        deviation_score:      sig.deviation_score,
        kalman_zscore:        sig.kalman_zscore,
        dcc_correlation:      sig.dcc_correlation,
        predicted_return:     sig.predicted_return,
        pred_var:             sig.pred_var,
        spread:               sig.spread,
        regime_label:         sig.regime_label.clone(),
        sl_price:             sig.sl,
        tp_price:             sig.tp,
        position_units:       sig.position_units,
        fee_bps:              cfg.fee_bps,
        slippage_bps:         cfg.slippage_bps,
        transaction_cost_bps: tcost,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// build_trade_row
// ---------------------------------------------------------------------------

pub fn build_trade_row(
    dr:    &DecisionRow,
    cfg:   &EngineConfig,
    gross: f64,
    cost:  f64,
    net:   f64,
) -> TradeRow {
    TradeRow {
        schema_version:         SCHEMA_VERSION,
        decision_id:            dr.decision_id.clone(),
        timestamp:              dr.timestamp.clone(),
        next_timestamp:         dr.next_timestamp.clone(),
        symbol:                 dr.symbol.clone(),
        action:                 dr.action.clone(),
        preferred_action:       dr.preferred_action.clone(),
        direction:              dr.direction,
        entry_price_raw:        dr.entry_price_raw.unwrap_or(0.0),
        entry_price_effective:  dr.entry_price_effective.unwrap_or(0.0),
        exit_price_raw:         dr.exit_price_raw.unwrap_or(0.0),
        exit_price_effective:   dr.exit_price_effective.unwrap_or(0.0),
        sl_price:               dr.sl_price,
        tp_price:               dr.tp_price,
        final_sl:               dr.final_sl,
        final_tp:               dr.final_tp,
        position_units:         dr.position_units,
        exit_timestamp:         dr.exit_timestamp.clone(),
        exit_reason:            dr.exit_reason.clone().unwrap_or_default(),
        bars_held:              dr.bars_held.unwrap_or(0),
        kalman_flip_bar:        dr.kalman_flip_bar,
        open_at_end:            dr.open_at_end,
        gross_return:           gross,
        cost_return:            cost,
        net_return:             net,
        fee_bps:                cfg.fee_bps,
        slippage_bps:           cfg.slippage_bps,
        transaction_cost_bps:   2.0 * (cfg.fee_bps + cfg.slippage_bps),
        pnl_1:                  dr.pnl_1,
        pnl_3:                  dr.pnl_3,
        pnl_5:                  dr.pnl_5,
        max_adverse:            dr.max_adverse,
        max_favorable:          dr.max_favorable,
        hit_1:                  dr.hit_1,
        hit_3:                  dr.hit_3,
        hit_5:                  dr.hit_5,
        t_profit:               dr.t_profit,
    }
}

// ---------------------------------------------------------------------------
// Gate schema finalization (mirrors Python _finalize_gate_schema)
// ---------------------------------------------------------------------------

const BASE_GATE_FIELDS: &[&str] = &[
    "signal_present",
    "entry_price_available",
    "exit_price_available",
    "execution_ready",
];

pub fn finalize_gate_schema(ledger: &mut Vec<DecisionRow>) {
    // Collect ordered unique gate names
    let mut ordered: Vec<String> = BASE_GATE_FIELDS.iter().map(|s| s.to_string()).collect();
    for row in ledger.iter() {
        for gn in row.gate_results.keys() {
            if !ordered.contains(gn) {
                ordered.push(gn.clone());
            }
        }
    }
    // Apply mask to each row
    for row in ledger.iter_mut() {
        row.gate_fields = ordered.clone();
        row.gate_pass_mask = ordered
            .iter()
            .map(|g| if row.gate_results.get(g).copied().unwrap_or(true) { 1 } else { 0 })
            .collect();
    }
}
