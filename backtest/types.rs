//! backtest_rs::types
//!
//! Shared value types used across the entire crate.
//! All structs derive Serde so they can be round-tripped to JSON in the Python
//! layer without a separate serialisation step.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Exit reason
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitReason {
    SlHit,
    TpHit,
    KalmanFlip,
    MaxBarsReached,
    OpenAtEnd,
    NextBarClose,
}

impl ExitReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SlHit         => "sl_hit",
            Self::TpHit         => "tp_hit",
            Self::KalmanFlip    => "kalman_flip",
            Self::MaxBarsReached=> "max_bars_reached",
            Self::OpenAtEnd     => "open_at_end",
            Self::NextBarClose  => "next_bar_close",
        }
    }
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Trade direction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i8)]
pub enum Direction {
    Long  =  1,
    Short = -1,
}

impl Direction {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            1  => Some(Self::Long),
            -1 => Some(Self::Short),
            _  => None,
        }
    }
    pub fn as_i32(self) -> i32 { self as i32 }
}

// ---------------------------------------------------------------------------
// SimResult — output of a single position simulation
// ---------------------------------------------------------------------------

/// Full simulation result for one position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimResult {
    pub exit_price_raw: f64,
    /// Timestamp as ISO-8601 string (or raw i64 ms since epoch as string).
    pub exit_timestamp: Option<String>,
    pub exit_reason:    ExitReason,
    pub bars_held:      usize,
    pub open_at_end:    bool,
    pub final_sl:       Option<f64>,
    pub final_tp:       Option<f64>,
    pub kalman_flip_bar: Option<usize>,
    // ── ML horizon labels ──────────────────────────────────────────────
    pub pnl_1:          Option<f64>,
    pub pnl_3:          Option<f64>,
    pub pnl_5:          Option<f64>,
    pub max_adverse:    Option<f64>,
    pub max_favorable:  Option<f64>,
    pub hit_1:          Option<bool>,
    pub hit_3:          Option<bool>,
    pub hit_5:          Option<bool>,
    pub t_profit:       Option<bool>,
}

// ---------------------------------------------------------------------------
// TradeRow — final row written to the ledger
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRow {
    pub schema_version:         &'static str,
    pub decision_id:            String,
    pub timestamp:              Option<String>,
    pub next_timestamp:         Option<String>,
    pub symbol:                 String,
    pub action:                 String,
    pub direction:              i32,
    pub entry_price_raw:        f64,
    pub entry_price_effective:  f64,
    pub exit_price_raw:         f64,
    pub exit_price_effective:   f64,
    pub sl_price:               Option<f64>,
    pub tp_price:               Option<f64>,
    pub final_sl:               Option<f64>,
    pub final_tp:               Option<f64>,
    pub position_units:         Option<f64>,
    pub exit_timestamp:         Option<String>,
    pub exit_reason:            String,
    pub bars_held:              usize,
    pub kalman_flip_bar:        Option<usize>,
    pub open_at_end:            bool,
    pub gross_return:           f64,
    pub cost_return:            f64,
    pub net_return:             f64,
    pub fee_bps:                f64,
    pub slippage_bps:           f64,
    pub transaction_cost_bps:   f64,
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
// BacktestSummary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestSummary {
    pub total_trades:       usize,
    pub win_rate:           f64,
    pub gross_pnl:          f64,
    pub avg_pnl_per_trade:  f64,
    pub net_pnl:            f64,
    pub gross_return:       f64,
    pub net_return:         f64,
    pub skipped_trades:     usize,
    pub open_at_end_count:  usize,
    pub max_drawdown:       f64,
    pub equity_curve:       Vec<f64>,
    pub equity_curve_mode:  String,
}

// ---------------------------------------------------------------------------
// EngineConfig — all tuning parameters in one place
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub fee_bps:                      f64,
    pub slippage_bps:                 f64,
    pub horizons:                     Vec<usize>,
    pub max_holding_bars:             usize,
    pub kalman_flip_zscore:           f64,
    pub dcc_flip_eps_multiplier:      f64,
    pub dynamic_sltp_update_interval: usize,  // 0 = disabled
    pub equity_curve_mode:            String, // "additive" | "compounding"
    // ATR-based trailing-stop knobs (mirror Python defaults)
    pub k_atr_stop:                   f64,
    pub rls_scaling_factor_sl:        f64,
    pub rls_sl_max_multiplier:        f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            fee_bps:                      0.0,
            slippage_bps:                 0.0,
            horizons:                     vec![1, 3, 5],
            max_holding_bars:             500,
            kalman_flip_zscore:           3.0,
            dcc_flip_eps_multiplier:      0.5,
            dynamic_sltp_update_interval: 0,
            equity_curve_mode:            "additive".to_string(),
            k_atr_stop:                   1.5,
            rls_scaling_factor_sl:        0.5,
            rls_sl_max_multiplier:        2.0,
        }
    }
}
