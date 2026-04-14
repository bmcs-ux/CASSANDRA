//! backtest::attribution
//!
//! Gate attribution summarisation — exact port of Python
//! `summarize_gate_attribution`.

use serde::{Deserialize, Serialize};

use crate::ledger::DecisionRow;

const BASE_GATE_FIELDS: &[&str] = &[
    "signal_present",
    "entry_price_available",
    "exit_price_available",
    "execution_ready",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateAttribution {
    pub gate:                   String,
    pub horizon_field:          String,
    pub observations:           usize,
    pub evaluated_decisions:    usize,
    pub unblocked_trade_count:  usize,
    pub mean_impact:            f64,
    pub median_impact:          f64,
    pub tail_impact_p95:        f64,
    pub positive_impact_rate:   f64,
}

pub fn summarize_gate_attribution(
    ledger:        &[DecisionRow],
    horizon_field: &str,
) -> Vec<GateAttribution> {
    use std::collections::HashMap;

    let mut impacts:    HashMap<String, Vec<f64>> = HashMap::new();
    let mut eval_cnt:   HashMap<String, usize>    = HashMap::new();
    let mut unlock_cnt: HashMap<String, usize>    = HashMap::new();

    for row in ledger {
        let pd  = row.preferred_direction;
        let bb  = &row.blocked_by;
        let act = if row.actually_executed {
            row.net_return.unwrap_or(0.0)
        } else {
            0.0
        };
        // "candidate" PnL: the horizon label we would have captured
        let cand = horizon_pnl(row, horizon_field).unwrap_or(0.0);

        for gn in &row.gate_fields {
            if BASE_GATE_FIELDS.contains(&gn.as_str()) {
                continue;
            }
            *eval_cnt.entry(gn.clone()).or_insert(0) += 1;

            let wp = if pd != 0 && bb.len() == 1 && bb[0] == *gn {
                *unlock_cnt.entry(gn.clone()).or_insert(0) += 1;
                cand
            } else {
                act
            };
            impacts.entry(gn.clone()).or_default().push(wp - act);
        }
    }

    let mut result: Vec<GateAttribution> = impacts
        .iter()
        .map(|(gn, imps)| {
            let mut sorted = imps.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n    = sorted.len();
            let p95i = ((0.95 * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
            let mean = imps.iter().sum::<f64>() / n as f64;
            let pos_rate = imps.iter().filter(|&&v| v > 0.0).count() as f64 / n as f64;

            GateAttribution {
                gate:                  gn.clone(),
                horizon_field:         horizon_field.to_string(),
                observations:          n,
                evaluated_decisions:   eval_cnt.get(gn).copied().unwrap_or(0),
                unblocked_trade_count: unlock_cnt.get(gn).copied().unwrap_or(0),
                mean_impact:           mean,
                median_impact:         sorted[n / 2],
                tail_impact_p95:       sorted[p95i],
                positive_impact_rate:  pos_rate,
            }
        })
        .collect();

    result.sort_by(|a, b| a.gate.cmp(&b.gate));
    result
}

fn horizon_pnl(row: &DecisionRow, field: &str) -> Option<f64> {
    match field {
        "pnl_1" => row.pnl_1,
        "pnl_3" => row.pnl_3,
        "pnl_5" => row.pnl_5,
        _       => None,
    }
}
