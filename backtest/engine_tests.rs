//! tests/engine_tests.rs
//!
//! Regression tests that verify the Rust engine produces the same results
//! as the Python reference implementation.

use backtest_rs::{
    engine::{simulate_trade_inner, COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TS},
    types::{Direction, EngineConfig, ExitReason},
};
use polars::prelude::*;

fn df(opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64]) -> DataFrame {
    let n = opens.len();
    DataFrame::new(vec![
        Series::new(COL_TS,    (0i64..(n as i64)).collect::<Vec<_>>()),
        Series::new(COL_OPEN,  opens.to_vec()),
        Series::new(COL_HIGH,  highs.to_vec()),
        Series::new(COL_LOW,   lows.to_vec()),
        Series::new(COL_CLOSE, closes.to_vec()),
    ]).unwrap()
}

fn cfg() -> EngineConfig { EngineConfig::default() }

// ── SL hit tests ─────────────────────────────────────────────────────────────

#[test]
fn long_sl_hit_on_bar2() {
    // Long, SL=98, bar 1 is fine (low=99), bar 2 hits SL (low=97)
    let df = df(
        &[100.0, 100.0, 100.0],
        &[101.0, 101.0, 101.0],
        &[99.0,  97.0,  99.0 ],
        &[100.5, 99.0,  100.5],
    );
    let res = simulate_trade_inner(
        &df, 100.0, Direction::Long, Some(98.0), Some(105.0),
        &cfg(), None, None, 3.0, false, false,
    ).unwrap();
    assert_eq!(res.exit_reason, ExitReason::SlHit);
    assert_eq!(res.bars_held,   2);
    assert!((res.exit_price_raw - 98.0).abs() < 1e-9);
}

#[test]
fn short_tp_hit_on_bar1() {
    // Short, TP=96, bar 1 low touches 95
    let df = df(
        &[100.0],
        &[101.0],
        &[95.0 ],
        &[99.0 ],
    );
    let res = simulate_trade_inner(
        &df, 100.0, Direction::Short, Some(103.0), Some(96.0),
        &cfg(), None, None, 3.0, false, false,
    ).unwrap();
    assert_eq!(res.exit_reason, ExitReason::TpHit);
    assert_eq!(res.bars_held,   1);
    assert!((res.exit_price_raw - 96.0).abs() < 1e-9);
}

// ── Max bars ─────────────────────────────────────────────────────────────────

#[test]
fn max_bars_reached() {
    let mut opens  = vec![100.0f64; 10];
    let mut highs  = vec![101.0f64; 10];
    let mut lows   = vec![99.5f64;  10]; // never hits SL=99 or TP=110
    let mut closes = vec![100.0f64; 10];

    let mut cfg = cfg();
    cfg.max_holding_bars = 5;  // cap at 5

    let df = df(&opens, &highs, &lows, &closes);
    let res = simulate_trade_inner(
        &df, 100.0, Direction::Long, Some(99.0), Some(110.0),
        &cfg, None, None, 3.0, false, false,
    ).unwrap();
    assert_eq!(res.exit_reason, ExitReason::MaxBarsReached);
    assert_eq!(res.bars_held,   5);
}

// ── ML labels ────────────────────────────────────────────────────────────────

#[test]
fn horizon_labels_correct() {
    // Long, 5 bars, close prices: 101, 102, 103, 104, 105
    // pnl_1 = (101/100 - 1) = 1%,  pnl_3 = 3%, pnl_5 = 5%
    let df = df(
        &[100.0; 5],
        &[106.0; 5], // high never hits TP
        &[99.5; 5],  // low never hits SL
        &[101.0, 102.0, 103.0, 104.0, 105.0],
    );
    let cfg = cfg();
    let res = simulate_trade_inner(
        &df, 100.0, Direction::Long, Some(95.0), Some(110.0),
        &cfg, None, None, 3.0, false, false,
    ).unwrap();
    let pnl_1 = res.pnl_1.unwrap();
    let pnl_3 = res.pnl_3.unwrap();
    let pnl_5 = res.pnl_5.unwrap();
    // approximate (costs=0 in default cfg)
    assert!((pnl_1 - 0.01).abs() < 1e-6, "pnl_1={pnl_1}");
    assert!((pnl_3 - 0.03).abs() < 1e-6, "pnl_3={pnl_3}");
    assert!((pnl_5 - 0.05).abs() < 1e-6, "pnl_5={pnl_5}");
    assert_eq!(res.hit_1, Some(true));
    assert_eq!(res.t_profit, Some(true));
}

// ── Trailing SL update ───────────────────────────────────────────────────────

#[test]
fn trailing_sl_tightens_correctly() {
    // Long, update every 2 bars, ATR=1.0, k=1.5 → SL tightens as price rises
    let mut cfg = cfg();
    cfg.dynamic_sltp_update_interval = 2;

    // 4 bars, close steadily rising: 101, 102, 103, 104
    let df = df(
        &[100.0; 4],
        &[105.0; 4],  // never hits initial TP=200
        &[99.0; 4],   // never hits initial SL=95
        &[101.0, 102.0, 103.0, 104.0],
    );
    let res = simulate_trade_inner(
        &df, 100.0, Direction::Long, Some(95.0), Some(200.0),
        &cfg, Some(1.0), Some(0.0), 3.0, false, false,
    ).unwrap();
    // After 2 bars (close=102): new SL candidate = 102 - 1.5*1.0 = 100.5 > 95 → updated
    // After 4 bars (close=104): candidate = 104 - 1.5 = 102.5 > 100.5 → updated again
    let final_sl = res.final_sl.unwrap();
    assert!(final_sl > 95.0, "SL should have been trailed up from 95.0, got {final_sl}");
}
