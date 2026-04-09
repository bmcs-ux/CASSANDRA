//! benches/simulate_bench.rs
//!
//! cargo bench --bench simulate_bench
//!
//! Measures throughput of `simulate_trade_inner` on synthetic M1 data.

use backtest_rs::{
    engine::{simulate_trade_inner, COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TS},
    types::{Direction, EngineConfig},
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars::prelude::*;

fn make_synthetic_df(n: usize) -> DataFrame {
    let ts:    Series = Series::new(COL_TS,    (0i64..(n as i64)).collect::<Vec<_>>());
    let open:  Series = Series::new(COL_OPEN,  vec![100.0f64; n]);
    let high:  Series = Series::new(COL_HIGH,  vec![101.0f64; n]);
    let low:   Series = Series::new(COL_LOW,   vec![99.0f64;  n]);
    let close: Series = Series::new(COL_CLOSE, vec![100.5f64; n]);
    DataFrame::new(vec![ts, open, high, low, close]).unwrap()
}

fn bench_simulate(c: &mut Criterion) {
    let cfg = EngineConfig::default();
    let mut group = c.benchmark_group("simulate_trade_inner");

    for &n in &[100usize, 500, 2000] {
        let df = make_synthetic_df(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("bars", n),
            &df,
            |b, df| {
                b.iter(|| {
                    simulate_trade_inner(
                        black_box(df),
                        100.0,
                        Direction::Long,
                        Some(98.0),
                        Some(103.0),
                        &cfg,
                        Some(0.5),   // atr
                        Some(0.1),   // deviation_score
                        3.0,         // flip_threshold
                        false,       // has_ktrend
                        false,       // has_kz
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_simulate);
criterion_main!(benches);
