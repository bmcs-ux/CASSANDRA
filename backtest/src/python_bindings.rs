//! backtest_rs::python_bindings
//!
//! PyO3 extension module.
//!
//! EXPOSED API (Python-visible)
//! ─────────────────────────────
//!   PyFastEngine(ipc_bytes, symbol) → per-symbol intrabar engine
//!   BacktestConfig                  → EngineConfig wrapper
//!   run_backtest(signals, engines, cfg) → BatchResult as Python dicts
//!
//! DATA FLOW
//! ─────────
//! Python orchestrator:
//!   1. Serialises each per-symbol polars DataFrame to IPC bytes on the
//!      Python side (`buf = io.BytesIO(); df.write_ipc(buf)`).
//!   2. Passes IPC bytes to `PyFastEngine(buf.getvalue(), sym)`.
//!   3. Calls `run_backtest(signals, engines, cfg)`.
//!
//! WHY IPC BYTES (NOT PyAny DataFrame)
//! ─────────────────────────────────────
//! `polars.DataFrame.write_ipc(path_or_buf)` writes to a file/buffer and
//! returns None — it does NOT return bytes.  Extracting `Vec<u8>` from None
//! raises a PyO3 `NoneType` conversion error.  Passing pre-serialised bytes
//! from Python is the correct, version-stable contract.

use std::collections::HashMap;
use polars::prelude::SerReader;

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
use serde_json::Value;

use crate::{
    attribution::summarize_gate_attribution,
    engine::FastEngine,
    runner::{run_parallel_by_symbol, run_serial, SignalInput},
    types::EngineConfig,
};

// ---------------------------------------------------------------------------
// Error mapping helper
// ---------------------------------------------------------------------------

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// BacktestConfig — Python-visible EngineConfig
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct BacktestConfig {
    inner: EngineConfig,
}

#[pymethods]
impl BacktestConfig {
    #[new]
    #[pyo3(signature = (
        fee_bps                      = 0.0,
        slippage_bps                 = 0.0,
        horizons                     = None,
        max_holding_bars             = 500,
        kalman_flip_zscore           = 3.0,
        dcc_flip_eps_multiplier      = 0.5,
        dynamic_sltp_update_interval = 0,
        equity_curve_mode            = "additive",
        k_atr_stop                   = 1.5,
        rls_scaling_factor_sl        = 0.5,
        rls_sl_max_multiplier        = 2.0,
    ))]
    fn new(
        fee_bps:                      f64,
        slippage_bps:                 f64,
        horizons:                     Option<Vec<usize>>,
        max_holding_bars:             usize,
        kalman_flip_zscore:           f64,
        dcc_flip_eps_multiplier:      f64,
        dynamic_sltp_update_interval: usize,
        equity_curve_mode:            &str,
        k_atr_stop:                   f64,
        rls_scaling_factor_sl:        f64,
        rls_sl_max_multiplier:        f64,
    ) -> Self {
        Self {
            inner: EngineConfig {
                fee_bps,
                slippage_bps,
                horizons:                     horizons.unwrap_or_else(|| vec![1, 3, 5]),
                max_holding_bars,
                kalman_flip_zscore,
                dcc_flip_eps_multiplier,
                dynamic_sltp_update_interval,
                equity_curve_mode:            equity_curve_mode.to_string(),
                k_atr_stop,
                rls_scaling_factor_sl,
                rls_sl_max_multiplier,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!("BacktestConfig({:?})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// PyFastEngine
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PyFastEngine {
    engine: FastEngine,
    symbol: String,
}

#[pymethods]
impl PyFastEngine {
    /// Construct from Arrow IPC bytes produced on the Python side.
    ///
    /// Python caller:
    ///     import io
    ///     buf = io.BytesIO()
    ///     df.write_ipc(buf)
    ///     engine = PyFastEngine(buf.getvalue(), "EURUSD")
    ///
    /// The DataFrame must have columns: Timestamp, Open, High, Low, Close.
    /// Optional: kalman_trend, kalman_zscore / innovation_zscore.
    #[new]
    fn new(ipc_bytes: Vec<u8>, symbol: &str) -> PyResult<Self> {
        use polars::io::ipc::IpcReader;
        use std::io::Cursor;

        let df = IpcReader::new(Cursor::new(ipc_bytes))
            .finish()
            .map_err(|e| PyValueError::new_err(format!("IPC parse error: {e}")))?;

        let engine = FastEngine::new(df).map_err(to_py_err)?;
        Ok(Self { engine, symbol: symbol.to_string() })
    }

    /// Simulate a single trade.  Returns a dict matching SimResult schema.
    #[pyo3(signature = (
        entry_ts,
        entry_price,
        direction,
        sl       = None,
        tp       = None,
        cfg      = None,
        atr      = None,
        deviation_score = None,
        dcc_contagion   = None,
    ))]
    fn simulate(
        &self,
        py:              Python<'_>,
        entry_ts:        i64,
        entry_price:     f64,
        direction:       i32,
        sl:              Option<f64>,
        tp:              Option<f64>,
        cfg:             Option<&BacktestConfig>,
        atr:             Option<f64>,
        deviation_score: Option<f64>,
        dcc_contagion:   Option<f64>,
    ) -> PyResult<PyObject> {
        let dir = crate::types::Direction::from_i32(direction)
            .ok_or_else(|| PyValueError::new_err("direction must be 1 or -1"))?;
        let default_cfg = EngineConfig::default();
        let cfg_ref = cfg.map(|c| &c.inner).unwrap_or(&default_cfg);

        let result = self.engine
            .simulate(entry_ts, entry_price, dir, sl, tp, cfg_ref, atr, deviation_score, dcc_contagion)
            .map_err(to_py_err)?;

        let v = serde_json::to_value(&result).map_err(to_py_err)?;
        Ok(json_to_py(py, &v))
    }

    #[getter]
    fn symbol(&self) -> &str { &self.symbol }
}

// ---------------------------------------------------------------------------
// run_backtest — main entry point
// ---------------------------------------------------------------------------

/// Run the full backtest.
///
/// Parameters
/// ----------
/// signals : list[dict]
///     Each dict matches the `SignalInput` schema.
/// engines : dict[str, PyFastEngine]
///     Symbol → engine.  Pass empty dict for legacy one-bar mode.
/// config  : BacktestConfig
/// parallel: bool  (default True)
///
/// Returns
/// -------
/// dict with keys: trade_ledger, decision_ledger, summary, gate_attribution
#[pyfunction]
#[pyo3(signature = (signals, engines, config, parallel=true, horizon_field="pnl_1"))]
pub fn run_backtest(
    py:            Python<'_>,
    signals:       &PyList,
    engines:       &PyDict,
    config:        &BacktestConfig,
    parallel:      bool,
    horizon_field: &str,
) -> PyResult<PyObject> {
    // ── Deserialise signals ────────────────────────────────────────────────
    let sig_inputs: Vec<SignalInput> = signals
        .iter()
        .map(|item| py_dict_to_signal_input(item))
        .collect::<PyResult<Vec<_>>>()?;

    // ── Build engine map (safe: collect owned references while holding GIL) ──
    //
    // We extract each PyFastEngine as PyRef (GIL-guarded) and immediately grab
    // a raw pointer that we'll wrap in a newtype implementing Send+Sync.
    //
    // SAFETY CONTRACT:
    //   • `py` (the GIL token) is held for the entire duration of run_backtest.
    //   • PyFastEngine objects are owned by the Python dict `engines`, which
    //     itself lives on the Python heap and cannot be freed while we hold `py`.
    //   • We only read through the pointers (FastEngine is immutable after new()).
    //   • The pointers are only used inside this function's scope.
    struct EnginePtr(*const FastEngine);
    // SAFETY: FastEngine is read-only (no interior mutability), so sharing
    // raw pointers across rayon threads is safe for the duration of this call.
    unsafe impl Send for EnginePtr {}
    unsafe impl Sync for EnginePtr {}

    let eng_ptrs: HashMap<String, EnginePtr> = engines
        .iter()
        .map(|(k, v)| {
            let sym: String = k.extract()?;
            let eng: PyRef<PyFastEngine> = v.extract()?;
            let ptr = EnginePtr(&eng.engine as *const FastEngine);
            Ok((sym, ptr))
        })
        .collect::<PyResult<HashMap<_, _>>>()?;

    // Build a reference map that the runner helpers can use
    let eng_ref_map: HashMap<String, &FastEngine> = eng_ptrs
        .iter()
        .map(|(sym, ptr)| {
            // SAFETY: pointer is valid (see contract above)
            (sym.clone(), unsafe { &*ptr.0 })
        })
        .collect();

    let batch = if parallel {
        run_parallel_signals(&sig_inputs, &eng_ref_map, &config.inner)
    } else {
        run_serial_signals(&sig_inputs, &eng_ref_map, &config.inner)
    };

    let attribution = summarize_gate_attribution(&batch.decision_ledger, horizon_field);

    // ── Serialise to Python dicts ─────────────────────────────────────────
    let out = PyDict::new(py);
    out.set_item("trade_ledger",    rows_to_py(py, &batch.trade_ledger))?;
    out.set_item("decision_ledger", rows_to_py(py, &batch.decision_ledger))?;
    out.set_item("summary",         json_to_py(py, &serde_json::to_value(&batch.summary).map_err(to_py_err)?))?;
    out.set_item("gate_attribution",rows_to_py(py, &attribution))?;

    Ok(out.into())
}

// ---------------------------------------------------------------------------
// Parallel helpers (bypass borrow-checker generics issue)
// ---------------------------------------------------------------------------

fn run_parallel_signals(
    signals:  &[SignalInput],
    engines:  &HashMap<String, &FastEngine>,
    cfg:      &EngineConfig,
) -> crate::runner::BatchResult {
    use rayon::prelude::*;
    use std::collections::HashMap as HM;

    // Build by-symbol groups
    let mut by_sym: HM<&str, Vec<&SignalInput>> = HM::new();
    for sig in signals {
        by_sym.entry(sig.symbol.as_str()).or_default().push(sig);
    }

    let pairs: Vec<_> = by_sym
        .par_iter()
        .flat_map(|(sym, sigs)| {
            let eng = engines.get(*sym).copied();
            sigs.iter()
                .map(|sig| crate::runner::process_signal(sig, eng, cfg))
                .collect::<Vec<_>>()
        })
        .collect();

    crate::runner::assemble_batch_pub(pairs, cfg)
}

fn run_serial_signals(
    signals:  &[SignalInput],
    engines:  &HashMap<String, &FastEngine>,
    cfg:      &EngineConfig,
) -> crate::runner::BatchResult {
    let pairs: Vec<_> = signals
        .iter()
        .map(|sig| {
            let eng = engines.get(sig.symbol.as_str()).copied();
            crate::runner::process_signal(sig, eng, cfg)
        })
        .collect();

    crate::runner::assemble_batch_pub(pairs, cfg)
}

// ---------------------------------------------------------------------------
// Serialisation helpers
// ---------------------------------------------------------------------------

fn rows_to_py<T: serde::Serialize>(py: Python<'_>, rows: &[T]) -> PyObject {
    let list = PyList::empty(py);
    for row in rows {
        let v = serde_json::to_value(row).unwrap_or(Value::Null);
        list.append(json_to_py(py, &v)).unwrap();
    }
    list.into()
}

fn json_to_py(py: Python<'_>, v: &Value) -> PyObject {
    match v {
        Value::Null          => py.None(),
        Value::Bool(b)       => b.into_py(py),
        Value::Number(n)     => {
            if let Some(i) = n.as_i64() { i.into_py(py) }
            else { n.as_f64().unwrap_or(f64::NAN).into_py(py) }
        }
        Value::String(s)     => s.into_py(py),
        Value::Array(arr)    => {
            let list = PyList::empty(py);
            for item in arr { list.append(json_to_py(py, item)).unwrap(); }
            list.into()
        }
        Value::Object(map)   => {
            let dict = PyDict::new(py);
            for (k, val) in map { dict.set_item(k, json_to_py(py, val)).unwrap(); }
            dict.into()
        }
    }
}

// ---------------------------------------------------------------------------
// Python dict → SignalInput
// ---------------------------------------------------------------------------

fn py_dict_to_signal_input(obj: &PyAny) -> PyResult<SignalInput> {
    let d: &PyDict = obj.downcast()?;

    macro_rules! get_opt_f64 {
        ($key:expr) => {
            d.get_item($key)?.and_then(|v| v.extract::<f64>().ok())
        };
    }
    macro_rules! get_opt_str {
        ($key:expr) => {
            d.get_item($key)?.and_then(|v| v.extract::<String>().ok())
        };
    }
    macro_rules! get_opt_bool {
        ($key:expr) => {
            d.get_item($key)?.and_then(|v| v.extract::<bool>().ok())
        };
    }

    let gate_results: HashMap<String, bool> = d
        .get_item("gate_results")?
        .and_then(|v| v.downcast::<PyDict>().ok().map(|dd| {
            dd.iter()
              .filter_map(|(k, v)| Some((k.extract::<String>().ok()?, v.extract::<bool>().ok()?)))
              .collect()
        }))
        .unwrap_or_default();

    let blocked_by: Vec<String> = d
        .get_item("blocked_by")?
        .and_then(|v| v.extract::<Vec<String>>().ok())
        .unwrap_or_default();

    Ok(SignalInput {
        timestamp:       d.get_item("timestamp")?.map(|v| v.extract::<i64>()).transpose()?.unwrap_or(0),
        timestamp_str:   d.get_item("timestamp_str")?.map(|v| v.extract::<String>()).transpose()?.unwrap_or_default(),
        next_timestamp:  get_opt_str!("next_timestamp"),
        symbol:          d.get_item("symbol")?.map(|v| v.extract::<String>()).transpose()?.unwrap_or_default(),
        action:          d.get_item("action")?.map(|v| v.extract::<String>()).transpose()?.unwrap_or_else(|| "HOLD".into()),
        preferred_action: get_opt_str!("preferred_action"),
        entry_price:     get_opt_f64!("entry_price"),
        exit_price_next: get_opt_f64!("exit_price_next"),
        sl:              get_opt_f64!("sl"),
        tp:              get_opt_f64!("tp"),
        position_units:  get_opt_f64!("position_units"),
        rls_confidence:  get_opt_f64!("rls_confidence"),
        deviation_score: get_opt_f64!("deviation_score"),
        kalman_zscore:   get_opt_f64!("kalman_zscore"),
        dcc_correlation: get_opt_f64!("dcc_correlation"),
        dcc_contagion:   get_opt_f64!("dcc_contagion"),
        predicted_return:get_opt_f64!("predicted_return"),
        pred_var:        get_opt_f64!("pred_var"),
        spread:          get_opt_f64!("spread"),
        regime_label:    get_opt_str!("regime_label"),
        atr:             get_opt_f64!("atr"),
        gate_results,
        blocked_by,
        can_buy:         get_opt_bool!("can_buy"),
        can_sell:        get_opt_bool!("can_sell"),
    })
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

//#[pymodule]
pub fn backtest(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFastEngine>()?;
    m.add_class::<BacktestConfig>()?;
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    Ok(())
}
