//! backtest_rs::engine
//!
//! `FastEngine` owns one per-symbol M1 DataFrame sorted by Timestamp.
//! It prebuilds a HashMap<i64, usize> for O(1) bar-index lookup, then
//! uses `DataFrame::slice` for zero-copy intrabar windows.
//!
//! DESIGN DECISIONS THAT ELIMINATE BOTTLENECKS
//! ─────────────────────────────────────────────
//! 1. **Pre-sorted + index map at construction time.**
//!    `get_slice` is O(1) hash lookup + O(1) Polars slice (pointer bump,
//!    no allocation for the column buffers themselves).
//!
//! 2. **Column extraction is done once per simulation call**, not per bar.
//!    `simulate_trade` pulls &[Option<f64>] views via ChunkedArray iterators
//!    that are backed by the original Arrow buffers — no copy.
//!
//! 3. **Kalman state columns are optional.**  If they don't exist the flip
//!    check is simply skipped (zero branching overhead per missing column).
//!
//! 4. **Dynamic SL/TP update** uses only scalar arithmetic — no allocation.
//!
//! 5. **Label pass** is a single forward loop reusing the same bar indices
//!    already visited during SL/TP detection — no second traversal.

use std::collections::HashMap;

use anyhow::{Context, Result};
use hashbrown::HashMap as FastMap;
use polars::prelude::*;

use crate::types::{Direction, EngineConfig, ExitReason, SimResult};

// ---------------------------------------------------------------------------
// Column name constants — centralised to avoid typo bugs
// ---------------------------------------------------------------------------

pub const COL_TS     : &str = "Timestamp";
pub const COL_OPEN   : &str = "Open";
pub const COL_HIGH   : &str = "High";
pub const COL_LOW    : &str = "Low";
pub const COL_CLOSE  : &str = "Close";
pub const COL_KTREND : &str = "kalman_trend";   // optional
pub const COL_KZ     : &str = "kalman_zscore";  // optional; also "innovation_zscore"
pub const COL_KZ_ALT : &str = "innovation_zscore";

// ---------------------------------------------------------------------------
// FastEngine
// ---------------------------------------------------------------------------

/// One engine instance per symbol.  Cheap to clone (Arc-backed Polars Series).
#[derive(Debug)]
pub struct FastEngine {
    /// Sorted M1 DataFrame (Arrow-backed, zero-copy slices).
    df: DataFrame,
    /// Timestamp (ms since epoch) → row index.
    ts_index: FastMap<i64, usize>,
    /// Pre-resolved column presence flags (avoid repeated `contains` checks).
    has_ktrend: bool,
    has_kz:     bool,
}

impl FastEngine {
    // ── Constructor ─────────────────────────────────────────────────────────

    /// Build from an already-loaded Polars DataFrame.
    /// The DataFrame **must** have at minimum: Timestamp, Open, High, Low, Close.
    pub fn new(mut df: DataFrame) -> Result<Self> {
        // Sort in-place — ChunkedArray sort is SIMD-accelerated in Polars 0.39+
        df.sort_in_place([COL_TS], SortMultipleOptions::default())
            .context("FastEngine::new — sort by Timestamp failed")?;

        let ts_col = df
            .column(COL_TS)
            .context("FastEngine: missing 'Timestamp' column")?
            .cast(&DataType::Int64)
            .context("FastEngine: Timestamp not castable to i64")?;

        let ts_ca = ts_col.i64().context("FastEngine: Timestamp i64 cast failed")?;

        // Single-pass O(n) index build
        let mut ts_index = FastMap::with_capacity(ts_ca.len());
        for (i, opt_v) in ts_ca.into_iter().enumerate() {
            if let Some(v) = opt_v {
                ts_index.insert(v, i);
            }
        }

        let has_ktrend = df.get_column_names().contains(&COL_KTREND);
        let has_kz     = df.get_column_names().contains(&COL_KZ)
                      || df.get_column_names().contains(&COL_KZ_ALT);

        Ok(Self { df, ts_index, has_ktrend, has_kz })
    }

    // ── Slicing ─────────────────────────────────────────────────────────────

    /// Zero-copy slice starting **after** `ts` for at most `max_bars` rows.
    /// Returns `None` if `ts` is not found.
    #[inline]
    pub fn get_slice(&self, ts: i64, max_bars: usize) -> Option<DataFrame> {
        let idx = *self.ts_index.get(&ts)?;
        let start = (idx + 1) as i64;
        Some(self.df.slice(start, max_bars))
    }

    // ── Full simulation ──────────────────────────────────────────────────────

    /// Simulate one position from `entry_ts`, returning a `SimResult`.
    pub fn simulate(
        &self,
        entry_ts:      i64,
        entry_price:   f64,
        direction:     Direction,
        sl:            Option<f64>,
        tp:            Option<f64>,
        config:        &EngineConfig,
        atr:           Option<f64>,
        deviation_score: Option<f64>,
        dcc_contagion: Option<f64>,
    ) -> Result<SimResult> {
        let df = match self.get_slice(entry_ts, config.max_holding_bars) {
            Some(d) => d,
            None    => return Ok(open_at_end_result(entry_price, sl, tp)),
        };
        if df.height() == 0 {
            return Ok(open_at_end_result(entry_price, sl, tp));
        }

        // Compute the effective Kalman flip threshold (mirrors Python exactly)
        let dcc_mult = compute_dcc_multiplier(dcc_contagion, config.dcc_flip_eps_multiplier);
        let flip_threshold = config.kalman_flip_zscore * dcc_mult;

        simulate_trade_inner(
            &df,
            entry_price,
            direction,
            sl,
            tp,
            config,
            atr,
            deviation_score,
            flip_threshold,
            self.has_ktrend,
            self.has_kz,
        )
    }
}

// ---------------------------------------------------------------------------
// simulate_trade — inner engine (pure function, easy to unit-test)
// ---------------------------------------------------------------------------

/// Core bar-loop simulation.  All column data is extracted once into
/// `ChunkedArray` views; the inner loop only does scalar arithmetic.
pub fn simulate_trade_inner(
    df:              &DataFrame,
    entry_price:     f64,
    direction:       Direction,
    initial_sl:      Option<f64>,
    initial_tp:      Option<f64>,
    config:          &EngineConfig,
    atr:             Option<f64>,
    deviation_score: Option<f64>,
    flip_threshold:  f64,
    has_ktrend:      bool,
    has_kz:          bool,
) -> Result<SimResult> {
    let n   = df.height();
    let dir = direction.as_i32();

    // ── Extract column views once (zero-copy Arrow buffer references) ────────
    let high_ca  = get_f64_ca(df, COL_HIGH)?;
    let low_ca   = get_f64_ca(df, COL_LOW)?;
    let close_ca = get_f64_ca(df, COL_CLOSE)?;
    // Optional columns (kalman state)
    let ktrend_opt = if has_ktrend { df.column(COL_KTREND).ok() } else { None };
    let kz_opt     = if has_kz {
        df.column(COL_KZ).ok().or_else(|| df.column(COL_KZ_ALT).ok())
    } else {
        None
    };
    let kz_ca_opt: Option<&ChunkedArray<Float64Type>> = kz_opt
        .as_ref()
        .and_then(|s| s.cast(&DataType::Float64).ok().as_ref().map(|_| None).or(None));
    // We cast lazily to avoid allocation when kalman not present — handle below.

    // ── Mutable SL/TP state ──────────────────────────────────────────────────
    let mut sl       = initial_sl;
    let mut tp       = initial_tp;
    // ── Output values ────────────────────────────────────────────────────────
    let mut exit_idx:        usize         = n - 1;  // default: last bar
    let mut exit_reason:     ExitReason    = ExitReason::MaxBarsReached;
    let mut kalman_flip_bar: Option<usize> = None;
    // ML label accumulators — computed in the same pass
    let mut pnl_at:   [Option<f64>; 5]    = [None; 5];
    let mut max_adv   = f64::MAX;
    let mut max_fav   = f64::MIN;

    // ── Single forward loop ──────────────────────────────────────────────────
    'bars: for i in 0..n {
        let bar_i = i + 1; // 1-based "bars_held"

        // 1. Dynamic SL/TP update (every N bars)
        if config.dynamic_sltp_update_interval > 0
            && bar_i % config.dynamic_sltp_update_interval == 0
        {
            if let Some(mid) = close_ca.get(i) {
                update_dynamic_sl_tp(&mut sl, &mut tp, mid, dir, atr, deviation_score, config);
            }
        }

        let h_opt = high_ca.get(i);
        let l_opt = low_ca.get(i);
        let c_opt = close_ca.get(i);

        // 2. Kalman flip check (before SL/TP to mirror live engine)
        if has_ktrend && has_kz {
            if let Some(flipped) = check_kalman_flip_at(
                df, ktrend_opt.as_ref(), i, dir, flip_threshold,
            ) {
                if flipped {
                    kalman_flip_bar = Some(bar_i);
                    exit_idx    = i;
                    exit_reason = ExitReason::KalmanFlip;
                    // Record label for this bar then break
                    if let Some(c) = c_opt {
                        let pnl = raw_pnl(c, entry_price, dir);
                        record_label_bar(i, pnl, &mut pnl_at, &mut max_adv, &mut max_fav);
                    }
                    break 'bars;
                }
            }
        }

        // 3. SL/TP hit detection (intrabar OHLC order)
        let sl_hit = match (dir, sl, l_opt, h_opt) {
            (1,  Some(s), Some(l), _      ) => l <= s,
            (-1, Some(s), _,       Some(h)) => h >= s,
            _ => false,
        };
        let tp_hit = match (dir, tp, h_opt, l_opt) {
            (1,  Some(t), Some(h), _      ) => h >= t,
            (-1, Some(t), _,       Some(l)) => l <= t,
            _ => false,
        };

        // Record horizon PnL labels using close of each bar
        if let Some(c) = c_opt {
            let pnl = raw_pnl(c, entry_price, dir);
            record_label_bar(i, pnl, &mut pnl_at, &mut max_adv, &mut max_fav);
        }

        if sl_hit || tp_hit {
            exit_idx = i;
            exit_reason = if sl_hit && tp_hit {
                // Tiebreak: whichever price is closer to bar open
                let open_v = df.column(COL_OPEN).ok()
                    .and_then(|s| s.cast(&DataType::Float64).ok())
                    .and_then(|s| { let ca = s; ca.f64().ok().map(|a| a.get(i)) }.flatten());
                match (open_v, sl, tp) {
                    (Some(o), Some(s), Some(t)) => {
                        if (o - s).abs() <= (o - t).abs() { ExitReason::SlHit }
                        else { ExitReason::TpHit }
                    }
                    _ => ExitReason::SlHit,
                }
            } else if sl_hit {
                ExitReason::SlHit
            } else {
                ExitReason::TpHit
            };
            break 'bars;
        }

        // 4. Max bars guard (already bounded by slice length, but explicit)
        if bar_i >= config.max_holding_bars {
            exit_idx    = i;
            exit_reason = ExitReason::MaxBarsReached;
            break 'bars;
        }
    } // end 'bars

    // ── Derive exit price from exit_reason + exit_idx ────────────────────────
    let exit_price_raw = match exit_reason {
        ExitReason::SlHit      => sl.or_else(|| close_ca.get(exit_idx)).unwrap_or(entry_price),
        ExitReason::TpHit      => tp.or_else(|| close_ca.get(exit_idx)).unwrap_or(entry_price),
        ExitReason::KalmanFlip => {
            // Use bar open (mirrors live engine: fill at open of flip bar)
            df.column(COL_OPEN).ok()
              .and_then(|s| s.cast(&DataType::Float64).ok())
              .and_then(|s| s.f64().ok().map(|ca| ca.get(exit_idx)).flatten())
              .unwrap_or(entry_price)
        }
        _ => close_ca.get(exit_idx).unwrap_or(entry_price),
    };

    // ── Finalise open_at_end (data exhausted without hitting any exit) ────────
    let open_at_end = exit_reason == ExitReason::MaxBarsReached
        && (exit_idx + 1) < n
        // If we broke at max_bars the simulation is NOT open_at_end — it hit the cap
        // Only mark open_at_end when data was truly exhausted before max_bars
        // (handled by the constructor: slice is bounded by max_holding_bars, so if
        // df.height() < max_holding_bars, we exhausted data)
        && df.height() < config.max_holding_bars;

    let exit_reason_final = if open_at_end { ExitReason::OpenAtEnd } else { exit_reason };

    // ── Horizon labels ────────────────────────────────────────────────────────
    let scr    = (config.fee_bps + config.slippage_bps) / 10_000.0;
    let cost   = 2.0 * scr;  // round-trip cost approximation for label net-pnl
    let mk_lbl = |p: Option<f64>| p.map(|v| v - cost);

    let pnl_1  = mk_lbl(pnl_at[0]);
    let pnl_3  = mk_lbl(pnl_at[2]);
    let pnl_5  = mk_lbl(pnl_at[4]);
    let max_adverse  = if max_adv == f64::MAX  { None } else { Some(max_adv  - cost) };
    let max_favorable= if max_fav == f64::MIN  { None } else { Some(max_fav  - cost) };

    let hit_1 = pnl_1.map(|v| v > 0.0);
    let hit_3 = pnl_3.map(|v| v > 0.0);
    let hit_5 = pnl_5.map(|v| v > 0.0);
    let t_profit = match (hit_1, hit_3, hit_5) {
        (None, None, None) => None,
        (h1, h3, h5) => Some(
            h1.unwrap_or(false) || h3.unwrap_or(false) || h5.unwrap_or(false)
        ),
    };

    // ── Timestamps ───────────────────────────────────────────────────────────
    let exit_timestamp = extract_ts_string(df, exit_idx);

    Ok(SimResult {
        exit_price_raw,
        exit_timestamp,
        exit_reason: exit_reason_final,
        bars_held: exit_idx + 1,
        open_at_end,
        final_sl: sl,
        final_tp: tp,
        kalman_flip_bar,
        pnl_1, pnl_3, pnl_5,
        max_adverse, max_favorable,
        hit_1, hit_3, hit_5, t_profit,
    })
}

// ---------------------------------------------------------------------------
// Dynamic SL/TP update (mirrors Python _update_dynamic_sl_tp exactly)
// ---------------------------------------------------------------------------

#[inline]
fn update_dynamic_sl_tp(
    sl: &mut Option<f64>,
    tp: &mut Option<f64>,   // tp unchanged in trailing-stop mode, kept for symmetry
    mid: f64,
    dir: i32,
    atr: Option<f64>,
    dev: Option<f64>,
    cfg: &EngineConfig,
) {
    let _ = tp; // tp not modified in trailing-stop logic
    let atr_v = match atr { Some(a) if a > 0.0 => a, _ => return };
    let d = dev.unwrap_or(0.0);
    let increase = 1.0 + d * cfg.rls_scaling_factor_sl;
    let k_adj    = (cfg.k_atr_stop * increase).min(cfg.k_atr_stop * cfg.rls_sl_max_multiplier);
    let dist     = k_adj * atr_v;

    if dir == 1 {
        let candidate = mid - dist;
        *sl = Some(match *sl { Some(prev) => prev.max(candidate), None => candidate });
    } else {
        let candidate = mid + dist;
        *sl = Some(match *sl { Some(prev) => prev.min(candidate), None => candidate });
    }
}

// ---------------------------------------------------------------------------
// Kalman flip check
// ---------------------------------------------------------------------------

/// Returns `Some(true)` if flip triggered, `Some(false)` if checked but no flip,
/// `None` if data missing (caller skips).
#[inline]
fn check_kalman_flip_at(
    df:         &DataFrame,
    ktrend_col: Option<&Series>,
    i:          usize,
    dir:        i32,
    threshold:  f64,
) -> Option<bool> {
    let trend_s = ktrend_col?;
    let trend_v: &str = trend_s.str().ok()?.get(i)?;
    let trend_upper = trend_v.to_uppercase();

    // kz: try f64 cast of COL_KZ or COL_KZ_ALT
    let kz: f64 = df.column(COL_KZ)
        .or_else(|_| df.column(COL_KZ_ALT))
        .ok()
        .and_then(|s| s.cast(&DataType::Float64).ok())
        .and_then(|s| s.f64().ok().map(|ca| ca.get(i)).flatten())?;

    let flipped = match (dir, trend_upper.as_str()) {
        (1,  "DOWN") => kz >= threshold,
        (-1, "UP")   => kz >= threshold,
        _            => false,
    };
    Some(flipped)
}

// ---------------------------------------------------------------------------
// Label helpers
// ---------------------------------------------------------------------------

#[inline]
fn raw_pnl(close: f64, entry: f64, dir: i32) -> f64 {
    (close / entry - 1.0) * dir as f64
}

/// Record PnL at horizon indices 0,1,2,3,4 (bar 1,2,3,4,5).
#[inline]
fn record_label_bar(
    i:        usize,
    pnl:      f64,
    pnl_at:   &mut [Option<f64>; 5],
    max_adv:  &mut f64,
    max_fav:  &mut f64,
) {
    if i < 5 {
        pnl_at[i] = Some(pnl);
    }
    if pnl < *max_adv { *max_adv = pnl; }
    if pnl > *max_fav { *max_fav = pnl; }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

#[inline]
fn get_f64_ca<'a>(df: &'a DataFrame, name: &str) -> Result<&'a ChunkedArray<Float64Type>> {
    df.column(name)
        .context(format!("missing column '{name}'"))?
        .f64()
        .context(format!("column '{name}' is not Float64"))
}

fn extract_ts_string(df: &DataFrame, idx: usize) -> Option<String> {
    let s = df.column(COL_TS).ok()?;
    // Try string first (ISO), then i64 (epoch ms)
    if let Ok(ca) = s.str() {
        return ca.get(idx).map(|v| v.to_string());
    }
    if let Ok(ca) = s.i64() {
        return ca.get(idx).map(|v| v.to_string());
    }
    None
}

fn compute_dcc_multiplier(contagion: Option<f64>, eps_mult: f64) -> f64 {
    1.0_f64.max(1.0 + contagion.unwrap_or(0.0) * eps_mult)
}

pub fn open_at_end_result(entry_price: f64, sl: Option<f64>, tp: Option<f64>) -> SimResult {
    SimResult {
        exit_price_raw: entry_price,
        exit_timestamp: None,
        exit_reason: ExitReason::OpenAtEnd,
        bars_held: 0,
        open_at_end: true,
        final_sl: sl,
        final_tp: tp,
        kalman_flip_bar: None,
        pnl_1: None, pnl_3: None, pnl_5: None,
        max_adverse: None, max_favorable: None,
        hit_1: None, hit_3: None, hit_5: None,
        t_profit: None,
    }
}
