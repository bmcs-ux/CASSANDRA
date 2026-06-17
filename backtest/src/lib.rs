//! backtest — Sprint 3 high-performance intrabar backtest engine
//!
//! Module layout:
//!   types           — shared value types (ExitReason, Direction, SimResult, …)
//!   engine          — FastEngine (zero-copy Polars slicing + bar-loop simulation)
//!   ledger          — DecisionRow / TradeRow builders + gate schema finalisation
//!   runner          — parallel batch runner (rayon)
//!   attribution     — gate attribution summarisation
//!   python_bindings — PyO3 extension module (feature = "python")

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]  // many-param simulation calls are intentional

pub mod attribution;
pub mod engine;
pub mod ledger;
pub mod runner;
pub mod types;

#[cfg(feature = "python")]
pub mod python_bindings;

// Re-export the PyO3 module init function under the crate name so that
// `maturin develop` can find it.
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn backtest(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    python_bindings::backtest(py, m)
}

// Expose open_at_end_stub publicly so runner.rs can call it
impl engine::FastEngine {
    // already pub via module
}

/// Convenience re-export: stub used when M1 data is exhausted before entry.
pub use engine::open_at_end_result as open_at_end_stub;
