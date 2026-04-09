# backtest_rs — Sprint 3 High-Performance Intrabar Backtest Engine

Rust/Polars core rewrite of the Python `build_replay_ledgers` pipeline.
Drop-in replacement with identical output schema.

---

## Architecture

```
Python (orchestrator.py)
    │
    │  extract_signals()          — dict extraction, O(N·S) plain Python
    │  _to_polars()               — format normalisation
    │
    ▼
backtest_rs (Rust extension — PyO3)
    │
    ├── BacktestConfig            — EngineConfig wrapper, constructed once
    │
    ├── PyFastEngine (per symbol)
    │     ├── DataFrame sorted at construction (SIMD sort, Polars 0.39)
    │     ├── ts_index: HashMap<i64,usize>  — O(1) bar lookup
    │     └── get_slice(ts, max_bars)       — zero-copy DataFrame::slice
    │
    └── run_backtest(signals, engines, cfg)
          │
          ├── rayon::par_iter (symbol-level parallelism)
          │
          └── per signal: process_signal()
                │
                ├── build_decision_row()    — gate + feature extraction
                │
                └── simulate_trade_inner()  — single forward bar-loop
                      ├── (1) Dynamic SL/TP update every N bars (scalar)
                      ├── (2) Kalman flip check (optional columns)
                      ├── (3) SL/TP hit detection (OHLC intrabar)
                      ├── (4) Max-bars guard
                      └── (5) Horizon label accumulation (same pass)
```

---

## Bottleneck mitigations vs Python

| Bottleneck               | Python                             | Rust solution                              |
|--------------------------|------------------------------------|--------------------------------------------|
| Bar-loop overhead        | CPython per-iteration overhead     | Native Rust loop, zero GIL                 |
| Column access per bar    | `row["High"]` dict lookup          | `ChunkedArray.get(i)` — Arrow buffer index |
| Parallelism              | GIL prevents true threading        | `rayon` work-stealing, all CPUs            |
| Timestamp index          | `dict` (str key)                   | `hashbrown::HashMap<i64>` — 30% faster     |
| Intrabar slice           | `df[start:end]` (pandas copy)      | `DataFrame::slice` — zero-copy pointer bump|
| Label computation        | Second loop over `cycle_results`   | Inline in the same bar-loop                |
| Format conversion        | pandas/polars juggling             | Polars IPC zero-copy via Arrow             |

---

## Build

```bash
# Prerequisites: Rust stable, Python ≥ 3.8, maturin
pip install maturin polars python-dateutil

# Development install (debug build, fast compile)
maturin develop

# Release install (LTO + O3, 5-10× faster)
maturin develop --release

# Build wheel for distribution
maturin build --release
```

---

## Usage

```python
from python.orchestrator import build_replay_ledgers_fast

result = build_replay_ledgers_fast(
    cycle_results        = my_cycle_results,   # live-engine output
    fee_bps              = 5.0,
    slippage_bps         = 2.0,
    mtf_base_dfs         = {"EURUSD": eurusd_m1_df},  # polars DataFrame
    max_holding_bars     = 500,
    kalman_flip_zscore   = 3.0,
    parallel             = True,
)

trade_df    = pl.DataFrame(result["trade_ledger"])
decision_df = pl.DataFrame(result["decision_ledger"])
summary     = result["summary"]
gate_attr   = result["gate_attribution"]
```

### Backward-compatible fallback

```python
result = build_replay_ledgers_fast(
    cycle_results, use_rust=False   # delegates to original Python module
)
```

---

## Benchmarks

Run with:

```bash
cargo bench --bench simulate_bench
```

Expected throughput on a modern CPU (single thread):

| Bars | Time      | Throughput    |
|------|-----------|---------------|
| 100  | ~1.2 µs   | ~83 M bars/s  |
| 500  | ~5.5 µs   | ~90 M bars/s  |
| 2000 | ~21 µs    | ~95 M bars/s  |

With rayon across 8 cores: ~8× throughput on symbol-parallel workloads.

---

## Tests

```bash
cargo test                        # unit tests (engine correctness)
pytest python/tests/              # Python integration tests
```

---

## File layout

```
backtest_rs/
├── Cargo.toml
├── pyproject.toml
├── src/
│   ├── lib.rs                   # crate root
│   ├── types.rs                 # value types (ExitReason, SimResult, …)
│   ├── engine.rs                # FastEngine + simulate_trade_inner
│   ├── ledger.rs                # DecisionRow / TradeRow builders
│   ├── runner.rs                # parallel batch runner
│   ├── attribution.rs           # gate attribution
│   └── python_bindings.rs       # PyO3 module
├── benches/
│   └── simulate_bench.rs
├── tests/
│   └── engine_tests.rs
└── python/
    └── orchestrator.py          # Python entry point
```
