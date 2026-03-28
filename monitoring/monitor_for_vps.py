import pandas as pd
import numbers
from datetime import datetime, timedelta, timezone
import warnings
import numpy as np
import time
from io import StringIO
import re
import pickle
from typing import Optional, Dict, Any, Tuple
from scipy.stats import norm
import sys
import os
import requests

# NEW: Using MT5Adapter
from adapters.mt5_adapter import MT5Adapter
from adapters.mt5_adapter import MT5_TIMEFRAME_MAP

# Global helper function to convert NumPy floats to native Python types
def convert_numpy_floats(obj):
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, numbers.Number):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    elif isinstance(obj, (str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_numpy_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_floats(elem) for elem in obj]
    try:
        return float(obj)
    except (ValueError, TypeError):
        return obj


def _compute_rls_confidence(maturity: float, pred_variance: float, deviation_norm: float, variance_ref: float) -> float:
    """Compute smooth confidence score for RLS health, including warm-up cycles."""
    maturity = float(np.clip(maturity, 0.0, 1.0))
    variance_ref = max(float(variance_ref), 1e-12)
    pred_variance = max(float(pred_variance), 1e-12)
    deviation_norm = max(float(deviation_norm), 0.0)

    normalized_uncertainty = pred_variance / variance_ref
    warmup_factor = 0.25 + (0.75 * maturity)
    dev_penalty = np.exp(-0.2 * deviation_norm)

    return float(
        np.clip(
            warmup_factor * dev_penalty * np.exp(-parameter.RLS_CONFIDENCE_ALPHA * normalized_uncertainty),
            0.0,
            1.0,
        )
    )


def _summarize_rls_global_metrics(confidence_map: Dict[str, float], maturity_map: Dict[str, float], deviation_map: Dict[str, float]) -> Tuple[float, float]:
    """Aggregate global confidence/deviation with maturity-aware weighting."""
    weighted_conf_sum = 0.0
    weight_sum = 0.0

    for key, confidence in confidence_map.items():
        maturity = float(np.clip(maturity_map.get(key, 0.0), 0.0, 1.0))
        if not np.isfinite(confidence) or maturity <= 0:
            continue
        weighted_conf_sum += float(confidence) * maturity
        weight_sum += maturity

    if weight_sum > 0:
        global_confidence = weighted_conf_sum / weight_sum
    else:
        global_confidence = 0.0

    valid_deviations = [float(v) for v in deviation_map.values() if np.isfinite(v)]
    global_deviation = float(np.mean(valid_deviations)) if valid_deviations else 0.0
    return float(global_confidence), float(global_deviation)


def _resolve_pair_pred_variance(rls_metrics: Dict[str, Dict[str, float]], pair_group: str, timeframe: str = "H1") -> float:
    """Resolve pair/group prediction variance with timeframe fallback."""
    tf_group_key = f"{str(timeframe).upper()}::{pair_group}"

    for key in (pair_group, tf_group_key):
        pred_var = float(rls_metrics.get(key, {}).get("pred_var", float("nan")))
        if np.isfinite(pred_var):
            return pred_var

    return float("inf")


def _stabilize_expected_return(raw_expected_return: float, previous_expected_return: float) -> float:
    """Reduce cycle-to-cycle noise in expected return using EMA + deadband."""
    raw = float(raw_expected_return)
    prev = float(previous_expected_return)
    alpha = float(np.clip(getattr(parameter, "RLS_RETURN_EMA_ALPHA", 0.35), 0.0, 1.0))
    deadband = max(float(getattr(parameter, "RLS_RETURN_DEADBAND", 5e-5)), 0.0)

    smoothed = (alpha * raw) + ((1.0 - alpha) * prev)
    if abs(smoothed) < deadband:
        return 0.0
    return float(smoothed)


def _passes_rls_directional_confirmation(signal_side: str, expected_return: float) -> bool:
    """Direction-aware confirmation for BUY/SELL signal validation."""
    eps = max(float(getattr(parameter, "RLS_RETURN_DIRECTION_EPSILON", 0.0)), 0.0)
    expected_return = float(expected_return)

    if signal_side == "BUY":
        return expected_return > eps
    if signal_side == "SELL":
        return expected_return < -eps
    return True


def _compute_dynamic_position_tp(
    is_buy: bool,
    latest_actual_price: float,
    entry_price: float,
    sl_dist: float,
    kalman_result: Dict[str, Any],
    pair_rls_deviation: float,
) -> float:
    """Hitung TP modifikasi posisi dengan guard RR minimum + arah entry.

    Tujuan:
    - TP tidak terlalu dekat dengan harga saat ini (noise stop-out).
    - TP tidak berada di sisi merugikan terhadap harga entry (BUY di bawah entry / SELL di atas entry).
    """
    kalman_projected_tp = float(kalman_result.get("filtered_price", latest_actual_price)) + float(
        kalman_result.get("velocity", 0.0)
    )
    kalman_projected_tp = max(kalman_projected_tp, 1e-6)

    raw_tp = max(kalman_projected_tp, latest_actual_price) if is_buy else min(kalman_projected_tp, latest_actual_price)

    tp_rr_ratio = float(getattr(parameter, "TP_RR_RATIO", 1.0))
    tp_red_factor = 1 - (float(pair_rls_deviation) * float(getattr(parameter, "RLS_SCALING_FACTOR_TP", 0.0)))
    tp_rr_floor = float(getattr(parameter, "RLS_TP_RR_MIN", 0.3))
    tp_rr_adj = max(tp_rr_floor, tp_rr_ratio * tp_red_factor)
    min_target_dist = max(abs(float(sl_dist)) * tp_rr_adj, 1e-6)

    if is_buy:
        tp_floor = max(float(latest_actual_price), float(entry_price)) + min_target_dist
        return max(raw_tp, tp_floor)

    tp_ceiling = min(float(latest_actual_price), float(entry_price)) - min_target_dist
    return max(min(raw_tp, tp_ceiling), 1e-6)


import parameter

current_script_dir = parameter.ROOT_DIR
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)
import monitoring.vps_colab_connector

VPS_PARAM_DIR = parameter.VPS_PARAM_DIR
VPS_DATA_DIR = parameter.VPS_DATA_DIR

if VPS_PARAM_DIR not in sys.path:
    sys.path.insert(0, VPS_PARAM_DIR)

MT5_LOGIN = parameter.MT5_LOGIN
MT5_PASSWORD = parameter.MT5_PASSWORD
MT5_SERVER = parameter.MT5_SERVER

preprocessing_path = os.path.join(current_script_dir, 'preprocessing')
if preprocessing_path not in sys.path:
    sys.path.insert(0, preprocessing_path)

from preprocessing.log_return import apply_log_return_to_price as _apply_log_return_to_price
from preprocessing.combine_data import combine_log_returns as _combine_log_returns
from preprocessing.stationarity_test import test_and_stationarize_data as _test_and_stationarize_data

warnings.filterwarnings("ignore")

COLAB_API_KEY_FOR_MONITOR = parameter.COLAB_API_KEY_FOR_MONITOR
COLAB_URL_FILE_PATH = parameter.COLAB_URL_FILE_PATH

TRADE_ENGINE_API_URL = parameter.TRADE_ENGINE_API_URL

# Runtime cache untuk stabilisasi metrik antar-siklus
PAIR_REALIZED_STD_CACHE = {}
_MISSING_EXOG_WARNED = set()
KALMAN_STATE_CACHE = {}
KALMAN_MODEL_PARAMS = {}


def _extract_ensemble_payload(loaded_payload: dict) -> dict:
    """Normalisasi payload model yang dimuat dari pickle.

    Mendukung beberapa format payload model:
    - {"data": {...}} (legacy)
    - {"ensemble": {...}, "dcc_garch": {...}} (fitted_ensemble.pkl)
    - mapping grup langsung (legacy flat)
    """
    if not isinstance(loaded_payload, dict):
        return {}

    if "data" in loaded_payload and isinstance(loaded_payload["data"], dict):
        data_payload = loaded_payload["data"]
        if "ensemble" in data_payload or "dcc_garch" in data_payload:
            return data_payload
        return {"ensemble": {"H1": data_payload}}

    if "ensemble" in loaded_payload or "dcc_garch" in loaded_payload:
        return loaded_payload

    return {"ensemble": {"H1": loaded_payload}}


def _resolve_timeframe_seconds(tf_name: str) -> Optional[int]:
    mapping = {
        "M1": 60,
        "M5": 300,
        "M15": 900,
        "H1": 3600,
        "H4": 14400,
        "D1": 86400,
    }
    return mapping.get(str(tf_name).upper())


def _compute_contagion_score_from_covariance(
    covariance_matrix: np.ndarray,
    covariance_series_names: Optional[list],
    target_series_names: Optional[list],
) -> float:
    """Hitung contagion score dari covariance forecast DCC.

    Jika nama seri tersedia, metrik dihitung hanya untuk sub-matriks `target_series_names`
    agar skor per-group tidak tercampur dengan seri dari group lain.
    """
    if covariance_matrix is None:
        return 0.0

    matrix = np.asarray(covariance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        return 0.0

    if covariance_series_names and target_series_names:
        idx_map = {name: idx for idx, name in enumerate(covariance_series_names)}
        selected_idx = [idx_map[name] for name in target_series_names if name in idx_map]
        if len(selected_idx) >= 2:
            matrix = matrix[np.ix_(selected_idx, selected_idx)]

    std = np.sqrt(np.diag(matrix))
    denom = np.outer(std, std)
    corr_matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0)
    upper = np.triu(np.abs(corr_matrix), k=1)
    non_zero = upper[upper > 0]
    if non_zero.size == 0:
        return 0.0
    return float(np.clip(float(np.mean(non_zero)), 0.0, 1.0))


def _timeframe_close_marker(tf_name: str, current_ts):
    if current_ts is None:
        return None
    ts = pd.Timestamp(current_ts)
    tf = str(tf_name).upper()
    if tf == "D1":
        return ts.floor("D")
    sec = _resolve_timeframe_seconds(tf)
    if sec is None:
        return ts
    return pd.Timestamp(int(ts.timestamp() // sec * sec), unit="s", tz=ts.tz)


def _is_new_timeframe_close(
    tf_name: str,
    current_ts,
    last_seen_map: Dict[str, Any],
    marker_key: Optional[str] = None,
) -> bool:
    """True jika marker candle baru untuk key tertentu sudah close dibanding siklus sebelumnya."""
    marker = _timeframe_close_marker(tf_name, current_ts)
    if marker is None:
        return False
    key = marker_key or tf_name
    previous_ts = last_seen_map.get(key)
    if previous_ts is None:
        last_seen_map[key] = marker
        return True
    if marker > previous_ts:
        last_seen_map[key] = marker
        return True
    return False

def format_for_dashboard(rls_forecasts, latest_prices):
    """
    Menyederhanakan data RLS agar langsung bisa dibaca oleh JavaScript Dashboard.
    Mengubah format nested menjadi flat (angka murni).
    """
    formatted_forecast = {}
    for sym, data in rls_forecasts.items():
        # Ambil hanya harga prediksi (float)
        if isinstance(data, dict) and 'rls_predicted_price' in data:
            formatted_forecast[sym] = data['rls_predicted_price']
        else:
            formatted_forecast[sym] = data # Fallback jika sudah berupa angka

    return formatted_forecast

def _build_regressor_matrix(log_stream, current_hf_combined_log_returns_df, latest_hf_fred_exog_df, lagged_hf_log_returns_df, maxlags, endog_names_group, exog_names_group):
    phi_list = [1.0]

    # 1. Endogenous Lags
    for lag in range(1, maxlags + 1):
        for endog_name in endog_names_group:
            col_name = f'Lag{lag}_{endog_name}'
            val = 0.0
            if col_name in lagged_hf_log_returns_df.columns:
                # Gunakan .iloc[0] untuk mengambil baris pertama tanpa peduli label indeks
                val = lagged_hf_log_returns_df[col_name].iloc[0]
            else:
                #log_stream.write(f"    [WARN] Lagged column {col_name} not found. Using 0.\n")
                val = 0.0
            phi_list.append(val if pd.notnull(val) else 0.0)

    # 2. Exogenous Variables
    for exog_name in exog_names_group:
        val = 0.0
        normalized_exog_name = str(exog_name).replace('_Transformed', '').replace('_FinalTransformed', '')
        # Cek di FRED Exog
        if exog_name in latest_hf_fred_exog_df.columns:
            # PERBAIKAN DI SINI: Gunakan .iloc[0] untuk menghindari KeyError pada DatetimeIndex
            val = latest_hf_fred_exog_df[exog_name].iloc[0]

        elif exog_name in current_hf_combined_log_returns_df.columns:
            val = current_hf_combined_log_returns_df[exog_name].iloc[0]
        elif normalized_exog_name in latest_hf_fred_exog_df.columns:
            val = latest_hf_fred_exog_df[normalized_exog_name].iloc[0]
        elif normalized_exog_name in current_hf_combined_log_returns_df.columns:
            val = current_hf_combined_log_returns_df[normalized_exog_name].iloc[0]
        else:
            warn_key = (exog_name, normalized_exog_name)
            if warn_key not in _MISSING_EXOG_WARNED:
                log_stream.write(
                    f"    [WARN] Exogenous column {exog_name} (alias: {normalized_exog_name}) not found. Using 0.\n"
                )
                _MISSING_EXOG_WARNED.add(warn_key)

        phi_list.append(val if pd.notnull(val) else 0.0)

    return np.array([phi_list])

def _perform_rls_update(log_stream, theta: np.ndarray, P: np.ndarray, Phi: np.ndarray, Y_t: np.ndarray, forgetting_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    if Y_t.ndim == 1:
        Y_t = Y_t.reshape(1, -1)

    predicted_Y = Phi @ theta
    e = Y_t - predicted_Y

    K_scalar_denom = P @ Phi.T / (forgetting_factor + Phi @ P @ Phi.T)

    updated_theta = theta + K_scalar_denom @ e

    updated_P = (1.0 / forgetting_factor) * (P - K_scalar_denom @ Phi @ P)

    return updated_theta, updated_P

def _extract_baseline_varx_params(log_stream, fitted_model_obj, n_endog_group, k_regressors, endog_names_group, exog_names_group, maxlags) -> Optional[np.ndarray]:
    try:
        model_params = fitted_model_obj.params
        param_names = fitted_model_obj.param_names

        log_stream.write(f"    [INFO] Expected k_regressors for RLS theta: {k_regressors}\n")

        theta_ref = np.zeros((k_regressors, n_endog_group))

        for i, target_endog_name in enumerate(endog_names_group):
            regressor_idx_in_phi = 0

            const_param_name = 'intercept' if n_endog_group == 1 else f'intercept.{target_endog_name}'
            if const_param_name in param_names:
                theta_ref[regressor_idx_in_phi, i] = model_params[param_names.index(const_param_name)]
            else:
                log_stream.write(f"    [WARN] Constant '{const_param_name}' not found for {target_endog_name}. Assuming 0.\n")
            regressor_idx_in_phi += 1

            for lag in range(1, maxlags + 1):
                for source_endog_name in endog_names_group:
                    if n_endog_group > 1:
                        lagged_param_name = f'L{lag}.{source_endog_name}.{target_endog_name}'
                    else:
                        lagged_param_name = f'L{lag}.{source_endog_name}'

                    if lagged_param_name in param_names:
                        theta_ref[regressor_idx_in_phi, i] = model_params[param_names.index(lagged_param_name)]
                    else:
                        log_stream.write(f"    [WARN] Lagged param '{lagged_param_name}' not found for {target_endog_name}. Assuming 0.\n")
                    regressor_idx_in_phi += 1

            for exog_name in exog_names_group:
                if n_endog_group > 1:
                    exog_param_name = f'beta.{exog_name}.{target_endog_name}'
                else:
                    exog_param_name = exog_name

                if exog_param_name in param_names:
                    theta_ref[regressor_idx_in_phi, i] = model_params[param_names.index(exog_param_name)]
                else:
                    log_stream.write(f"    [WARN] Exog param '{exog_param_name}' not found for {target_endog_name}. Assuming 0.\n")
                regressor_idx_in_phi += 1

        log_stream.write(f"    [OK] Baseline VARX parameters extracted successfully. Shape: {theta_ref.shape}\n")
        return theta_ref

    except Exception as e:
        log_stream.write(f"    [ERROR] Failed to extract baseline VARX parameters: {e}\n")
        return None

def rls_forecast_step(
    log_stream,
    theta_rls: np.ndarray,
    current_hf_combined_log_returns_df,
    latest_hf_fred_exog_df,
    lagged_hf_log_returns_df,
    maxlags: int,
    endog_names_group: list,
    exog_names_group: list
) -> np.ndarray:
    """
    Perform 1-step-ahead forecast using RLS-adapted parameters.

    Returns
    -------
    Y_hat : np.ndarray
        Shape (1, n_endog_group)
    """

    # Build regressor Φ_t
    Phi_t = _build_regressor_matrix(
        log_stream=log_stream,
        current_hf_combined_log_returns_df=current_hf_combined_log_returns_df,
        latest_hf_fred_exog_df=latest_hf_fred_exog_df,
        lagged_hf_log_returns_df=lagged_hf_log_returns_df,
        maxlags=maxlags,
        endog_names_group=endog_names_group,
        exog_names_group=exog_names_group
    )

    # Sanity checks
    if Phi_t.shape[1] != theta_rls.shape[0]:
        raise ValueError(
            f"Regressor size mismatch: Phi_t has {Phi_t.shape[1]} cols, "
            f"theta has {theta_rls.shape[0]}"
        )

    # Forecast
    Y_hat = Phi_t @ theta_rls  # (1 × k) @ (k × n_endog)

    log_stream.write(
        f"    [FORECAST] RLS-based forecast computed | Shape: {Y_hat.shape}\n"
    )

    return Y_hat

def infer_rls_expected_return(
    log_stream,
    pair_name,
    rls_estimators,
    current_hf_combined_log_returns_df,
    latest_hf_fred_exog_df,
    lagged_hf_log_returns_df
):
    """
    Mencari grup RLS secara otomatis berdasarkan pair_name dan menghitung expected return.
    """
    target_group = None
    target_idx = -1

    # 1. Proses Pencarian Grup dan Indeks Otomatis
    for g_name, estimator in rls_estimators.items():
        for idx, full_col_name in enumerate(estimator['endog_names']):
            # Mencocokkan "XAU/USD" ke "XAU/USD_Close_Log_Return"
            if pair_name in full_col_name:
                target_group = g_name
                target_idx = idx
                break
        if target_group:
            break

    # 2. Validasi Keberadaan Data
    if not target_group:
        log_stream.write(f"    [WARN] {pair_name} tidak ditemukan di grup RLS manapun. Skipping.\n")
        return None

    # 3. Eksekusi Forecast menggunakan Data dari Estimator yang ditemukan
    try:
        est = rls_estimators[target_group]

        Y_hat = rls_forecast_step(
            log_stream=log_stream,
            theta_rls=est["theta"],
            current_hf_combined_log_returns_df=current_hf_combined_log_returns_df,
            latest_hf_fred_exog_df=latest_hf_fred_exog_df,
            lagged_hf_log_returns_df=lagged_hf_log_returns_df,
            maxlags=est["maxlags"],
            endog_names_group=est["endog_names"],
            exog_names_group=est["exog_names"]
        )

        expected_return = float(Y_hat[0, target_idx])

        log_stream.write(
            f"    [INFER] Group: {target_group} | Asset: {pair_name} | "
            f"Exp.Return: {expected_return:+.6f}\n"
        )
        return expected_return

    except Exception as e:
        log_stream.write(f"    [ERROR] Gagal menghitung RLS forecast untuk {pair_name}: {e}\n")
        return None

def fetch_high_frequency_data(log_stream, mt5_adapter_instance, mt5_timeframe_map, PAIRS, HF_LOOKBACK_DAYS, HF_BASE_INTERVAL):
    log_stream.write(f"\n[INFO] Mulai pengunduhan data high-frequency untuk {len(PAIRS)} pasangan dari MetaTrader5...\n")
    hf_data_dfs = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HF_LOOKBACK_DAYS)

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    # Get MT5 timeframe constant
    mt5_timeframe = MT5_TIMEFRAME_MAP.get(HF_BASE_INTERVAL)
    if mt5_timeframe is None:
        log_stream.write(f"  [ERROR] Interval '{HF_BASE_INTERVAL}' tidak didukung oleh MetaTrader5. Silakan periksa MT5_TIMEFRAME_MAP.\n")
        return {}

    for pair_name, symbol in PAIRS.items():
        log_stream.write(f"  [INFO] Mengunduh {pair_name} ({symbol}) dengan interval {HF_BASE_INTERVAL} dari {start_date.date()} hingga {end_date.date()}\n")
        try:
            rates = mt5_adapter_instance.copy_rates_range(
                symbol,
                mt5_timeframe,
                datetime.fromtimestamp(start_ts, tz=timezone.utc),
                datetime.fromtimestamp(end_ts, tz=timezone.utc),
            )

            if rates is not None and len(rates) > 0:
                data = pd.DataFrame(rates)
                data['time'] = pd.to_datetime(data['time'], unit='s')
                data = data.set_index('time')
                data.index = data.index.tz_localize('UTC')
                data = data[['open', 'high', 'low', 'close', 'tick_volume']]
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if 'Volume' not in data.columns:
                    data['Volume'] = 0

                hf_data_dfs[pair_name] = data
                log_stream.write(f"    [OK] Berhasil mengunduh {len(data)} bar untuk {pair_name}. Shape: {data.shape}. Latest Close: {data['Close'].iloc[-1]:.4f} (at {data.index[-1]}).\n")
            else:
                log_stream.write(f"    [WARN] Tidak ada data yang diunduh untuk {pair_name} ({symbol}) dengan interval {HF_BASE_INTERVAL} dalam rentang waktu yang diminta.\n")

        except Exception as e:
            log_stream.write(f"    [ERROR] Gagal mengunduh data untuk {pair_name} ({symbol}) dari MetaTrader5: {e}\n")
            import traceback
            log_stream.write(traceback.format_exc())

    log_stream.write(f"\n[OK] Pengunduhan data high-frequency selesai.\n")
    return hf_data_dfs

def prepare_high_frequency_exogenous_data(log_stream, final_stationarized_fred_data, hf_index):
    log_stream.write(f"\n[INFO] Menyiapkan data eksogen FRED high-frequency...\n")

    if isinstance(final_stationarized_fred_data, dict) and "data" in final_stationarized_fred_data:
        fred_data_to_use = final_stationarized_fred_data["data"]
    else:
        fred_data_to_use = final_stationarized_fred_data

    if not fred_data_to_use:
        log_stream.write(f"  [WARN] Data FRED stasioner kosong. Tidak dapat menyiapkan eksogen high-frequency.\n")
        return pd.DataFrame()

    fred_df_list = []
    for name, df in fred_data_to_use.items():
        if not df.empty:
            value_col = [col for col in df.columns if col.endswith('_FinalTransformed') or col not in ['effective_until_next_release']][0]
            fred_df_list.append(df[[value_col]].rename(columns={value_col: name}))

    if not fred_df_list:
        log_stream.write(f"  [WARN] Tidak ada data nilai yang valid di `final_stationarized_fred_data`. Mengembalikan DataFrame kosong.\n")
        return pd.DataFrame()

    combined_fred_daily = pd.concat(fred_df_list, axis=1, join='outer').sort_index()
    combined_fred_daily = combined_fred_daily.ffill().dropna(how='all')

    if combined_fred_daily.empty:
        log_stream.write(f"  [WARN] Data FRED gabungan kosong setelah pembersihan. Mengembalikan DataFrame kosong.\n")
        return pd.DataFrame()

    if hf_index.tz is None and combined_fred_daily.index.tz is not None:
        combined_fred_daily.index = combined_fred_daily.index.tz_convert(None)
    elif hf_index.tz is not None and combined_fred_daily.index.tz is None:
        combined_fred_daily.index = combined_fred_daily.index.tz_localize(hf_index.tz)
    elif hf_index.tz is not None and combined_fred_daily.index.tz is not None and hf_index.tz != combined_fred_daily.index.tz:
        combined_fred_daily.index = combined_fred_daily.index.tz_convert(hf_index.tz)

    hf_fred_exog_aligned = combined_fred_daily.reindex(hf_index, method='ffill')

    hf_fred_exog_aligned = hf_fred_exog_aligned.dropna(how='all')

    log_stream.write(f"  [OK] Data FRED eksogen high-frequency berhasil disiapkan. Shape: {hf_fred_exog_aligned.shape}\n")
    return hf_fred_exog_aligned

def preprocess_high_frequency_data(log_stream, hf_raw_data_dfs, apply_log_return_to_price_func, combine_log_returns_func, test_and_stationarize_data_func, prepare_high_frequency_exogenous_data_func, final_stationarized_fred_data, alpha):
    log_stream.write(f"\n[INFO] Mulai preprocessing data high-frequency...\n")

    hf_log_returns_raw = apply_log_return_to_price_func(log_stream, hf_raw_data_dfs)
    if hf_log_returns_raw is None:
        hf_log_returns_raw = {}
        log_stream.write(f"[WARN] Gagal menerapkan transformasi log-return pada data high-frequency.\n")

    hf_log_returns_dict = combine_log_returns_func(log_stream, hf_log_returns_raw, return_type='dict')
    hf_combined_log_returns_df = combine_log_returns_func(log_stream, hf_log_returns_raw, return_type='df')

    # Proteksi RAM: stationarity test pada HF sangat mahal untuk ratusan ribu bar.
    # Batasi observasi terbaru agar proses VPS tidak terhenti karena OOM/interupsi paksa.
    max_stationarity_rows = int(getattr(parameter, "HF_MAX_STATIONARITY_ROWS", 20000))
    if max_stationarity_rows > 0:
        trimmed_dict = {}
        for pair_name, pair_df in (hf_log_returns_dict or {}).items():
            if isinstance(pair_df, pd.DataFrame) and len(pair_df) > max_stationarity_rows:
                trimmed_dict[pair_name] = pair_df.tail(max_stationarity_rows).copy()
                log_stream.write(
                    f"  [INFO] Trim stationarity input {pair_name}: {len(pair_df)} -> {len(trimmed_dict[pair_name])} rows.\n"
                )
            else:
                trimmed_dict[pair_name] = pair_df
        hf_log_returns_dict = trimmed_dict

    if isinstance(hf_combined_log_returns_df, pd.DataFrame) and not hf_combined_log_returns_df.empty:
        float_cols = hf_combined_log_returns_df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            hf_combined_log_returns_df[float_cols] = hf_combined_log_returns_df[float_cols].astype('float32')

    log_stream.write(f"\n[INFO] Memeriksa stasioneritas data log-return high-frequency...\n")
    hf_stationarity_results, _ = test_and_stationarize_data_func(log_stream, hf_log_returns_dict, {}, alpha)

    hf_fred_exog_aligned = pd.DataFrame()
    if hf_combined_log_returns_df is not None and not hf_combined_log_returns_df.empty:
        hf_fred_exog_aligned = prepare_high_frequency_exogenous_data_func(log_stream, final_stationarized_fred_data, hf_combined_log_returns_df.index)
    else:
        log_stream.write(f"  [WARN] Tidak ada data log-return high-frequency untuk menentukan indeks, melewati penyelarasan FRED.\n")

    log_stream.write(f"\n[OK] Preprocessing data high-frequency selesai.\n")
    return hf_log_returns_dict, hf_combined_log_returns_df, hf_stationarity_results, hf_fred_exog_aligned

def calculate_atr(log_stream, df_ohlc, atr_period=14):
    log_stream.write(f"  [INFO] Calculating ATR ({atr_period}-period)...\n")

    if df_ohlc.empty or not all(col in df_ohlc.columns for col in ['High', 'Low', 'Close']):
        log_stream.write(f"  [WARN] Input DataFrame for ATR calculation is empty or missing 'High', 'Low', 'Close' columns. Returning empty Series.\n")
        return pd.Series(dtype=float)

    high_low = df_ohlc['High'] - df_ohlc['Low']
    high_prev_close = abs(df_ohlc['High'] - df_ohlc['Close'].shift(1))
    low_prev_close = abs(df_ohlc['Low'] - df_ohlc['Close'].shift(1))

    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    tr.name = 'True_Range'

    atr = tr.rolling(window=atr_period).mean()
    atr.name = f'ATR_{atr_period}'

    log_stream.write(f"  [OK] ATR calculation completed for {len(df_ohlc)} observations. Last ATR value: {atr.iloc[-1]:.4f} (at {atr.index[-1]}).\n")
    return atr

def compute_rls_expected_return_for_pair(
    *,
    log_stream,
    pair_name,
    pair_group,
    rls_estimators,
    latest_hf_combined_log_returns_df,
    latest_hf_fred_exog_df_row,
):
    if pair_group not in rls_estimators:
        log_stream.write(
            f"    [WARN] RLS estimator missing for group {pair_group} ({pair_name}).\n"
        )
        return None

    est = rls_estimators[pair_group]

    try:
        return infer_rls_expected_return(
            log_stream=log_stream,
            theta_rls=est["theta"],
            current_hf_combined_log_returns_df=latest_hf_combined_log_returns_df,
            latest_hf_fred_exog_df=latest_hf_fred_exog_df_row,
            lagged_hf_log_returns_df=est["latest_lagged_df"],  #  simpan saat update
            maxlags=est["maxlags"],
            endog_names_group=est["endog_names"],
            exog_names_group=est["exog_names"],
            target_endog_idx=est["pair_target_index"][pair_name],
        )
    except Exception as e:
        log_stream.write(
            f"    [ERROR] Failed computing RLS expected return for {pair_name}: {e}\n"
        )
        return None

def decide_trade(
    log_stream,
    pair_name,
    latest_actual_price,
    expected_return,  # Ini adalah predicted move (log return)
    forecast_std,     # satuan price!
    forecast_std_return, # versi satuan log return dari forecast_std
    hf_atr,
    equity,
    risk_pct,
    k_atr_stop,
    k_model_stop,
    snr_threshold,
    rls_param_deviation_score,
    rls_deviation_threshold,
    tp_rr_ratio=1.5,
    kalman_velocity=0.0,
    kalman_innovation_zscore=0.0,
    kalman_trend="FLAT",
):
    # 1. Hitung Predicted Price (Mean) berdasarkan Log Return
    # Price_next = Price_now * exp(expected_return)
    predicted_mean = latest_actual_price * np.exp(expected_return)

    log_stream.write(f"\n  [INFO] Deciding trade for {pair_name} (Price: {latest_actual_price:.4f}, Pred. Mean: {predicted_mean:.4f})...\n")

    trade_decision = {
        'signal': 'HOLD',
        'entry_price': latest_actual_price,
        'stop_loss': np.nan,
        'take_profit': np.nan,
        'position_units': 0,
        'rr_ratio': np.nan,
        'snr': np.nan,
        'reason': 'No signal generated'
    }

    # 2. Kalman Trigger: arah entry ditentukan oleh velocity + innovation z-score.
    kalman_velocity_threshold = float(getattr(parameter, "KALMAN_VELOCITY_THRESHOLD", 1e-6))
    kalman_entry_zscore = float(getattr(parameter, "KALMAN_ENTRY_ZSCORE", 0.25))
    vel_ratio = abs(kalman_velocity) / kalman_velocity_threshold
    z_ratio = abs(kalman_innovation_zscore) / kalman_entry_zscore

    # Jika salah satu sangat kuat (misal 2x lipat threshold), kita anggap valid.
    if (vel_ratio + z_ratio) < 1.5:
        trade_decision['reason'] = f'Combined Kalman Trigger Weak ({vel_ratio + z_ratio:.2f})'
        return trade_decision

    # 3. Cek Stabilitas RLS
    if rls_param_deviation_score is not None and rls_deviation_threshold is not None:
        if rls_param_deviation_score > rls_deviation_threshold:
            trade_decision['reason'] = f'RLS unstable ({rls_param_deviation_score:.4f} > {rls_deviation_threshold:.4f})'
            log_stream.write(f"    [WARN] {pair_name}: {trade_decision['reason']}\n")
            return trade_decision

    # 4. Validasi Data (Menggunakan forecast_std dari argumen)
    if np.isnan(latest_actual_price) or np.isnan(expected_return) or np.isnan(forecast_std) or forecast_std <= 0:
        trade_decision['reason'] = 'Invalid input data (NaN or non-positive values)'
        log_stream.write(f"    [WARN] {pair_name}: {trade_decision['reason']}\n")
        return trade_decision

    # 5. Dynamic Adjustments
    k_atr_stop_adj = k_atr_stop
    k_model_stop_adj = k_model_stop
    tp_rr_adj = tp_rr_ratio
    snr_thresh_adj = snr_threshold

    if rls_param_deviation_score is not None and not np.isnan(rls_param_deviation_score):
        # Gunakan scaling factor dari parameter global
        inc_factor = 1 + (rls_param_deviation_score * parameter.RLS_SCALING_FACTOR_SL)
        k_atr_stop_adj = min(k_atr_stop * inc_factor, k_atr_stop * parameter.RLS_SL_MAX_MULTIPLIER)
        k_model_stop_adj = min(k_model_stop * inc_factor, k_model_stop * parameter.RLS_SL_MAX_MULTIPLIER)

        red_factor = 1 - (rls_param_deviation_score * parameter.RLS_SCALING_FACTOR_TP)
        tp_rr_adj = max(parameter.RLS_TP_RR_MIN, tp_rr_ratio * red_factor)

        snr_thresh_adj = snr_threshold * (1 + (rls_param_deviation_score * parameter.RLS_SNR_INCREASE_FACTOR))

    # 6. SNR Calculation (Predicted Log Return / Return Std Dev)
    snr = expected_return / forecast_std_return
    trade_decision['snr'] = snr

    if abs(snr) < snr_thresh_adj:
        trade_decision['reason'] = f'Low SNR ({abs(snr):.2f} < {snr_thresh_adj:.2f})'
        log_stream.write(f"    [INFO] {pair_name}: {trade_decision['reason']}\n")
        return trade_decision

    # 7. SL Distance Calculation
    # Note: forecast_std (price) dikonversi ke price distance
    sl_dist_atr = k_atr_stop_adj * hf_atr
    sl_dist_model = k_model_stop_adj * forecast_std_return * latest_actual_price
    sl_dist = max(sl_dist_atr, sl_dist_model)

    if sl_dist <= 1e-9:
        trade_decision['reason'] = 'SL Distance too small'
        return trade_decision

    # 8. Signal Direction mengikuti Kalman trigger
    if kalman_trend == "UP" and kalman_velocity > 0:
        trade_decision['signal'] = 'BUY'
        direction = 1
    elif kalman_trend == "DOWN" and kalman_velocity < 0:
        trade_decision['signal'] = 'SELL'
        direction = -1
    else:
        return trade_decision

    sl_dist = abs(sl_dist)

    # Hitung Jarak Prediksi (Target Move) dalam satuan price.
    # Bila predicted_mean terlalu dekat/berlawanan arah dengan arah entry Kalman,
    # pakai minimal target berbasis RR agar TP tidak terlalu sempit.
    signed_prediction_move = (predicted_mean - latest_actual_price) * direction
    prediction_dist = max(signed_prediction_move, 0.0)
    min_target_dist = sl_dist * tp_rr_adj
    effective_target_dist = max(prediction_dist, min_target_dist)

    realized_rr = effective_target_dist / sl_dist if sl_dist > 0 else 0
    trade_decision['rr_ratio'] = realized_rr
    # FILTER KRUSIAL: Jika target keuntungan lebih kecil dari risiko, jangan masuk.
    # Batas minimal 0.8 atau 1.0 agar masuk akal setelah spread.

    if realized_rr < tp_rr_adj:
        trade_decision['signal'] = 'HOLD'
        trade_decision['reason'] = f'Bad RR Ratio ({realized_rr:.2f} < {tp_rr_adj:.2f})'
        log_stream.write(f"    [INFO] {pair_name}: {trade_decision['reason']} (Target: {effective_target_dist:.2f}, SL Dist: {sl_dist:.2f})\n")
        return trade_decision

    if direction == 1: # BUY
        # TP mengikuti target efektif (model atau RR minimum, mana yang lebih jauh).
        tp_price = latest_actual_price + effective_target_dist
        sl_price = latest_actual_price - sl_dist
    else: # SELL
        tp_price = latest_actual_price - effective_target_dist
        sl_price = latest_actual_price + sl_dist

    # Hitung Realized RR Ratio (Penting untuk log dan monitoring)
    # RR = Jarak TP / Jarak SL

    trade_decision['take_profit'] = max(tp_price, 0.000001)
    trade_decision['stop_loss'] = max(sl_price, 0.000001)
    # Position Sizing
    max_risk_usd = equity * risk_pct
    raw_units = max_risk_usd / sl_dist

    # Apply caps (Saat ini hard cap = 0.01 lot; di bawah 0.01 tidak dieksekusi)
    position_units = max(0.0, min(raw_units, 0.01))
    if position_units < 0.01:
        position_units = 0.0
        trade_decision['signal'] = 'HOLD'
        trade_decision['reason'] = 'Position size below min (0.01)'
        log_stream.write(f"    [INFO] {pair_name}: {trade_decision['reason']} (Raw: {raw_units:.4f})\n")
        return trade_decision

    trade_decision['position_units'] = round(position_units, 2)

    if trade_decision['signal'] != 'HOLD':
        log_stream.write(f"    [OK] {pair_name} {trade_decision['signal']} | SNR: {snr:.2f} | Units: {trade_decision['position_units']}\n")
        trade_decision['reason'] = f"SNR {abs(snr):.2f} > {snr_thresh_adj:.2f}"

    return trade_decision

def _estimate_forecast_std(pair_name: str, latest_price: float, confidence_level: float, kalman_metrics: Optional[dict] = None) -> float:
    realized_std = float(PAIR_REALIZED_STD_CACHE.get(pair_name, 0.0) or 0.0)
    if realized_std <= 0 and latest_price > 0:
        realized_std = max(latest_price * 0.0005, 1e-6)

    kalman_z = 0.0
    if isinstance(kalman_metrics, dict):
        kalman_z = abs(float(kalman_metrics.get(pair_name, {}).get("innovation_zscore", 0.0)))
    kalman_scale = 1.0 + min(kalman_z, 5.0) * 0.1
    confidence_scale = max(0.2, 1.0 - min(max(confidence_level, 0.0), 0.999))
    return float(max(realized_std * kalman_scale * confidence_scale, 1e-6))


def detect_price_deviation(log_stream, latest_actual_prices: dict, rls_forecasts: dict, kalman_metrics: Optional[dict] = None, confidence_level: float = 0.95):
    log_stream.write(f"\n[INFO] Detecting price deviations from model forecast (RLS + Kalman) at {int(confidence_level*100)}% confidence...\n")
    deviation_results = {}

    for pair_name, actual_price in latest_actual_prices.items():
        log_stream.write(f"  [INFO] Checking deviation for {pair_name}. Actual price: {actual_price:.4f}\n")
        deviation_info = {
            'ci_breach': False,
            'deviation_metric': np.nan,
            'actual_price': float(actual_price),
            'predicted_mean': np.nan,
            'lower_ci': np.nan,
            'upper_ci': np.nan,
            'forecast_std': np.nan
        }

        if pair_name in rls_forecasts and isinstance(rls_forecasts[pair_name], dict):
            predicted_mean = float(rls_forecasts[pair_name].get("rls_predicted_price", np.nan))
            if not np.isfinite(predicted_mean):
                log_stream.write(f"    [WARN] Invalid predicted mean for {pair_name}. Skipping deviation check.\n")
                deviation_results[pair_name] = deviation_info
                continue

            forecast_std = _estimate_forecast_std(pair_name, float(actual_price), confidence_level, kalman_metrics)
            z_score_value = norm.ppf(1 - (1 - confidence_level) / 2)
            lower_ci = predicted_mean - (z_score_value * forecast_std)
            upper_ci = predicted_mean + (z_score_value * forecast_std)

            deviation_info['predicted_mean'] = predicted_mean
            deviation_info['lower_ci'] = lower_ci
            deviation_info['upper_ci'] = upper_ci
            deviation_info['forecast_std'] = float(forecast_std)

            log_stream.write(f"    [INFO] Forecasted Mean: {predicted_mean:.4f}, CI: [{lower_ci:.4f}, {upper_ci:.4f}]\n")

            if actual_price < lower_ci:
                deviation_info['ci_breach'] = True
                deviation_info['deviation_metric'] = (actual_price - predicted_mean) / forecast_std
                log_stream.write(f"    [ALERT] Actual price {actual_price:.4f} is BELOW lower CI {lower_ci:.4f} for {pair_name}. Deviation: {deviation_info['deviation_metric']:.2f} std devs.\n")
            elif actual_price > upper_ci:
                deviation_info['ci_breach'] = True
                deviation_info['deviation_metric'] = (actual_price - predicted_mean) / forecast_std
                log_stream.write(f"    [ALERT] Actual price {actual_price:.4f} is ABOVE upper CI {upper_ci:.4f} for {pair_name}. Deviation: {deviation_info['deviation_metric']:.2f} std devs.\n")
            else:
                log_stream.write(f"    [INFO] Actual price {actual_price:.4f} is within CI for {pair_name}.\n")
        else:
            log_stream.write(f"    [WARN] No model forecast found for {pair_name}. Skipping deviation check.\n")

        deviation_results[pair_name] = deviation_info

    log_stream.write(f"\n[OK] Price deviation detection completed.\n")
    return deviation_results

def _run_kalman_filter_step(pair_name: str, observed_price: float):
    state_data = KALMAN_STATE_CACHE.get(pair_name)

    model_cfg = KALMAN_MODEL_PARAMS.get("M1", {})
    F = np.array(model_cfg.get("F", getattr(parameter, "KALMAN_F", [[1, 1], [0, 1]])), dtype=float)
    H = np.array(model_cfg.get("H", getattr(parameter, "KALMAN_H", [[1, 0]])), dtype=float)
    Q = np.array(model_cfg.get("Q", getattr(parameter, "KALMAN_Q", [[1e-4, 0], [0, 1e-4]])), dtype=float)
    R = np.array(model_cfg.get("R", getattr(parameter, "KALMAN_R", [[1e-6]])), dtype=float)

    if state_data is None:
        default_state = [observed_price, 0.0]
        x = np.array(model_cfg.get("initial_state", getattr(parameter, "KALMAN_INITIAL_STATE", default_state)), dtype=float).reshape(-1, 1)
        P = np.array(model_cfg.get("initial_P", getattr(parameter, "KALMAN_INITIAL_P", [[0.1, 0], [0, 0.1]])), dtype=float)
        state_data = {"x": x, "P": P, "innovation_history": []}

    x_prior = F @ state_data["x"]
    P_prior = F @ state_data["P"] @ F.T + Q

    z = np.array([[float(observed_price)]])
    innovation = z - (H @ x_prior)
    S = H @ P_prior @ H.T + R
    K = P_prior @ H.T @ np.linalg.pinv(S)

    x_post = x_prior + K @ innovation
    P_post = (np.eye(P_prior.shape[0]) - K @ H) @ P_prior

    state_data["x"] = x_post
    state_data["P"] = P_post
    innovation_value = float(innovation.ravel()[0])
    state_data["innovation_history"].append(innovation_value)
    if len(state_data["innovation_history"]) > int(getattr(parameter, "KALMAN_ZSCORE_WINDOW", 120)):
        state_data["innovation_history"] = state_data["innovation_history"][-int(getattr(parameter, "KALMAN_ZSCORE_WINDOW", 120)):]

    innovation_std = float(np.std(state_data["innovation_history"])) if len(state_data["innovation_history"]) > 1 else 1e-12
    if innovation_std <= 0:
        innovation_std = 1e-12

    innovation_zscore = innovation_value / innovation_std
    kalman_trend = "UP" if float(x_post[1, 0]) >= 0 else "DOWN"

    KALMAN_STATE_CACHE[pair_name] = state_data
    return {
        "filtered_price": float(x_post[0, 0]),
        "velocity": float(x_post[1, 0]),
        "innovation": innovation_value,
        "innovation_zscore": float(innovation_zscore),
        "trend": kalman_trend
    }


def _signal_from_return(expected_return_value: float) -> str:
    return "BUY" if expected_return_value >= 0 else "SELL"


def _compute_consensus_score(signal_d1: str, signal_h1: str, signal_m15: str) -> float:
    weight_d1 = float(getattr(parameter, "CONSENSUS_WEIGHT_D1", 0.5))
    weight_h1 = float(getattr(parameter, "CONSENSUS_WEIGHT_H1", 0.3))
    weight_m15 = float(getattr(parameter, "CONSENSUS_WEIGHT_M15", 0.2))

    return (
        (1.0 if signal_d1 == 'BUY' else -1.0) * weight_d1 +
        (1.0 if signal_h1 == 'BUY' else -1.0) * weight_h1 +
        (1.0 if signal_m15 == 'BUY' else -1.0) * weight_m15
    )


def send_monitoring_data_to_colab(data: dict, log_stream):
    data_converted = convert_numpy_floats(data)

    return vps_colab_connector.send_data_to_colab(
        endpoint="update_monitor_data",
        data=data_converted,
        colab_api_key=COLAB_API_KEY_FOR_MONITOR,
        colab_url_file_path=COLAB_URL_FILE_PATH,
        log_func=lambda msg: log_stream.write(msg + '\n')
    )

def send_signal_to_trade_engine(signal_data: dict, log_stream) -> bool:
    try:
        TE_API_KEY = parameter.TRADE_ENGINE_API_KEY
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TE_API_KEY
        }
        log_stream.write(f"    [INFO] Sending signal to Trade Engine API: {TRADE_ENGINE_API_URL} with API Key: {TE_API_KEY}\n")

        response = requests.post(TRADE_ENGINE_API_URL, headers=headers, json=signal_data, timeout=5)
        response.raise_for_status()
        log_stream.write(f"    [INFO] Signal sent to Trade Engine: {signal_data.get('pair_name', 'N/A')} {signal_data.get('action', 'N/A')}. Response: {response.json()}\n")
        return True
    except requests.exceptions.Timeout:
        log_stream.write(f"    [ERROR] Sending signal to Trade Engine timed out for {signal_data.get('pair_name', 'N/A')}.\n")
    except requests.exceptions.ConnectionError as e:
        log_stream.write(f"    [ERROR] Connection error sending signal to Trade Engine for {signal_data.get('pair_name', 'N/A')}: {e}. Is Trade Engine running?\n")
    except requests.exceptions.HTTPError as e:
        log_stream.write(f"    [ERROR] HTTP error sending signal to Trade Engine for {signal_data.get('pair_name', 'N/A')}: {e}. Response: {response.text}\n")
    except Exception as e:
        log_stream.write(f"    [ERROR] Unexpected error sending signal to Trade Engine for {signal_data.get('pair_name', 'N/A')}: {e}\n")
    return False



def run_single_monitoring_cycle(
    mt5_adapter_instance,
    pipeline_run_id_for_monitor,
    cycle_count,
    log_stream=sys.stdout,
    is_backtest=False,
    interval_seconds: Optional[float] = None,
    confidence_level: float = 0.95,
    monitoring_context: Optional[Dict[str, Any]] = None,
):
    """Run one monitoring cycle.

    - Backtest/default mode: lightweight heuristic cycle (safe fallback).
    - Live mode with ``monitoring_context``: full RLS + risk-gating cycle.
    """
    cycle_start_time = time.time()
    monitoring_context = monitoring_context or {}

    # Fallback path for backtest and minimal contexts.
    if not monitoring_context:
        hf_raw_data_dfs = fetch_high_frequency_data(
            log_stream,
            mt5_adapter_instance,
            MT5_TIMEFRAME_MAP,
            parameter.PAIRS,
            parameter.HF_LOOKBACK_DAYS,
            parameter.HF_BASE_INTERVAL,
        )

        hf_log_returns_dict, hf_combined_log_returns_df, _, _ = preprocess_high_frequency_data(
            log_stream,
            hf_raw_data_dfs,
            _apply_log_return_to_price,
            _combine_log_returns,
            _test_and_stationarize_data,
            prepare_high_frequency_exogenous_data,
            {},
            parameter.alpha,
        )

        latest_hf_actual_prices = {}
        trade_signals = {}

        for pair_name, df in hf_raw_data_dfs.items():
            if df.empty or "Close" not in df.columns:
                continue
            latest_hf_actual_prices[pair_name] = float(df["Close"].iloc[-1])

            signal = "HOLD"
            reason = "Insufficient data"
            if (
                pair_name in hf_log_returns_dict
                and not hf_log_returns_dict[pair_name].empty
                and len(hf_log_returns_dict[pair_name]) >= 2
            ):
                recent_ret = float(hf_log_returns_dict[pair_name].iloc[-1, 0])
                signal = "BUY" if recent_ret > 0 else "SELL" if recent_ret < 0 else "HOLD"
                reason = f"Simple replay heuristic from latest return={recent_ret:+.6e}"

            trade_signals[pair_name] = {
                "signal": signal,
                "entry_price": latest_hf_actual_prices[pair_name],
                "stop_loss": None,
                "take_profit": None,
                "position_units": 0,
                "reason": reason,
            }

            if signal in ("BUY", "SELL"):
                signal_payload = {
                    "signal_id": f"{pair_name.replace('/', '')}_{cycle_count}",
                    "action": signal,
                    "symbol": pair_name,
                    "entry_price": latest_hf_actual_prices[pair_name],
                    "pipeline_run_id": pipeline_run_id_for_monitor,
                }
                if not is_backtest:
                    send_signal_to_trade_engine(signal_payload, log_stream)
                else:
                    log_stream.write(f"[SIM] Signal generated: {signal_payload['action']} for {pair_name}\n")

        cycle_duration = float(time.time() - cycle_start_time)
        if interval_seconds is not None and not is_backtest:
            time_to_sleep = interval_seconds - cycle_duration
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        return {
            "cycle_number": cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latest_actual_prices": latest_hf_actual_prices,
            "trade_signals": trade_signals,
            "rls_metrics": {},
            "global_confidence": 0.0,
            "pipeline_run_id": pipeline_run_id_for_monitor,
            "cycle_duration_seconds": cycle_duration,
            "is_backtest": bool(is_backtest),
        }

    # Advanced live-cycle path (integrated from legacy loop block).
    skip_individual_trade_decisions = False
    current_mt5_timeframe_map = monitoring_context["current_mt5_timeframe_map"]
    final_stationarized_fred_data = monitoring_context.get("final_stationarized_fred_data", {})
    rls_estimators = monitoring_context["rls_estimators"]
    dcc_model_registry = monitoring_context.get("dcc_model_registry", {})
    dcc_metrics_cache = monitoring_context.get("dcc_metrics_cache", {})
    dcc_timeframe_last_close_map = monitoring_context.get("dcc_timeframe_last_close_map", {})
    timeframe_last_close_map = monitoring_context.get("timeframe_last_close_map", {})
    expected_return_state = monitoring_context.setdefault("expected_return_state", {})
    PAIR_TO_RLS_GROUP = monitoring_context["pair_to_rls_group"]
    news_manager_instance = monitoring_context["news_manager_instance"]

    log_stream.write(f"\n--- Monitoring Cycle {cycle_count} (Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})---\n")
    log_stream.flush()

    hf_raw_data_dfs = fetch_high_frequency_data(
        log_stream,
        mt5_adapter_instance,
        current_mt5_timeframe_map,
        parameter.PAIRS,
        parameter.HF_LOOKBACK_DAYS,
        parameter.HF_BASE_INTERVAL,
    )

    hf_log_returns_dict, hf_combined_log_returns_df, _, hf_fred_exog_aligned = preprocess_high_frequency_data(
        log_stream,
        hf_raw_data_dfs,
        _apply_log_return_to_price,
        _combine_log_returns,
        _test_and_stationarize_data,
        prepare_high_frequency_exogenous_data,
        final_stationarized_fred_data,
        parameter.alpha,
    )

    if hf_combined_log_returns_df.empty or len(hf_combined_log_returns_df) <= parameter.maxlag_test:
        log_stream.write("  [WARN] Not enough valid high-frequency log returns for RLS. Skipping this cycle.\n")
        log_stream.flush()
        return {
            "cycle_number": cycle_count,
            "timestamp": datetime.now().isoformat(),
            "latest_actual_prices": {},
            "deviation_results": {},
            "rls_forecast": {},
            "rls_health": {},
            "trade_signals": {},
            "parameter_deviations": {},
            "pipeline_run_id": pipeline_run_id_for_monitor,
            "cycle_log": "Not enough data for RLS.",
            "is_backtest": bool(is_backtest),
        }

    latest_hf_actual_prices, latest_hf_atrs = {}, {}
    for pair_name, df in hf_raw_data_dfs.items():
        if not df.empty and "Close" in df.columns:
            latest_hf_actual_prices[pair_name] = float(df["Close"].iloc[-1])
            atr_series = calculate_atr(log_stream, df)
            if not atr_series.empty:
                latest_hf_atrs[pair_name] = float(atr_series.iloc[-1])

    # Light integration: keep risk gates & signal generation with existing helpers.
    rls_forecasts, rls_metrics, parameter_deviations = {}, {}, {}
    confidence_per_group, maturity_per_group = {}, {}
    dcc_group_metrics, kalman_metrics, consensus_metrics = {}, {}, {}
    mean_reversion_candidates, trade_signals = {}, {}

    PAIR_REALIZED_STD_CACHE.clear()
    volatility_window = int(getattr(parameter, "RLS_VOLATILITY_WINDOW", 96))

    for estimator_key, estimator_data in rls_estimators.items():
        updated_theta = estimator_data.get("theta")
        baseline_theta_ref = estimator_data.get("baseline_theta_ref")
        group_name = estimator_data.get("group_name", estimator_key)
        timeframe_name = estimator_data.get("timeframe", "H1")
        estimator_label = f"{timeframe_name}::{group_name}"

        if updated_theta is None or baseline_theta_ref is None:
            continue
        deviation_norm = float(np.linalg.norm(updated_theta - baseline_theta_ref))
        parameter_deviations[estimator_label] = deviation_norm
        confidence = float(np.clip(1.0 / (1.0 + deviation_norm), 0.0, 1.0))
        maturity = float(np.clip(estimator_data.get("rls_update_count", 0) / max(parameter.RLS_MIN_UPDATES_FOR_CONFIDENCE, 1), 0.0, 1.0))
        confidence_per_group[estimator_label] = confidence
        maturity_per_group[estimator_label] = maturity
        rls_metrics[estimator_label] = {
            "confidence": confidence,
            "maturity": maturity,
            "deviation": deviation_norm,
            "pred_var": float(parameter.RLS_INITIAL_P_DIAG),
        }
        if timeframe_name == "H1":
            parameter_deviations[group_name] = deviation_norm
            confidence_per_group[group_name] = confidence
            maturity_per_group[group_name] = maturity

    global_rls_confidence, rls_param_deviation_score = _summarize_rls_global_metrics(
        confidence_map=confidence_per_group,
        maturity_map=maturity_per_group,
        deviation_map=parameter_deviations,
    )

    if global_rls_confidence < parameter.RLS_CONFIDENCE_ENTRY_THRESHOLD:
        skip_individual_trade_decisions = parameter._RLS_CONFIDENCE
    if rls_param_deviation_score > parameter.RLS_DEVIATION_CLOSE_ALL_THRESHOLD:
        if not is_backtest:
            send_signal_to_trade_engine({"signal_id": f"CLOSE_ALL_RISK_{pipeline_run_id_for_monitor}_{cycle_count}", "action": "CLOSE_ALL"}, log_stream)
        skip_individual_trade_decisions = parameter._RLS_DEVIATION_THRESHOLD

    news_status = news_manager_instance.get_news_status()
    if news_status.get("is_restricted"):
        skip_individual_trade_decisions = parameter.NEWS

    latest_row = hf_combined_log_returns_df.iloc[[-1]]
    exog_row = hf_fred_exog_aligned.iloc[[-1]] if not hf_fred_exog_aligned.empty else pd.DataFrame()

    for pair_name in latest_hf_actual_prices:
        if skip_individual_trade_decisions:
            trade_signals[pair_name] = {
                "signal": "HOLD",
                "entry_price": latest_hf_actual_prices[pair_name],
                "stop_loss": np.nan,
                "take_profit": np.nan,
                "position_units": 0,
                "rr_ratio": np.nan,
                "snr": np.nan,
                "reason": "Skipped by safety filter",
            }
            continue

        raw_ret = infer_rls_expected_return(
            log_stream=log_stream,
            pair_name=pair_name,
            rls_estimators=rls_estimators,
            current_hf_combined_log_returns_df=latest_row,
            latest_hf_fred_exog_df=exog_row,
            lagged_hf_log_returns_df=pd.DataFrame(),
        )
        if raw_ret is None or pair_name not in latest_hf_atrs:
            trade_signals[pair_name] = {"signal": "HOLD", "reason": "RLS unavailable or ATR missing"}
            continue

        prev_expected_return = expected_return_state.get(pair_name, float(raw_ret))
        rls_expected_return = _stabilize_expected_return(float(raw_ret), prev_expected_return)
        expected_return_state[pair_name] = float(rls_expected_return)

        predicted_mean = latest_hf_actual_prices[pair_name] * np.exp(rls_expected_return)
        rls_forecasts[pair_name] = {
            "rls_predicted_price": float(predicted_mean),
            "rls_expected_return_pct": float(rls_expected_return * 100),
        }

        account_info = mt5_adapter_instance.account_info()
        current_equity = account_info.equity if account_info is not None else parameter.EQUITY
        forecast_std = _estimate_forecast_std(pair_name, latest_hf_actual_prices[pair_name], confidence_level, kalman_metrics)
        forecast_std_return = forecast_std / max(latest_hf_actual_prices[pair_name], 1e-12)
        signal = decide_trade(
            log_stream=log_stream,
            pair_name=pair_name,
            latest_actual_price=latest_hf_actual_prices[pair_name],
            expected_return=rls_expected_return,
            forecast_std=forecast_std,
            forecast_std_return=forecast_std_return,
            hf_atr=latest_hf_atrs[pair_name],
            equity=current_equity,
            risk_pct=parameter.RISK_PER_TRADE_PCT,
            k_atr_stop=parameter.K_ATR_STOP,
            k_model_stop=parameter.K_MODEL_STOP,
            snr_threshold=parameter.SNR_THRESHOLD,
            rls_param_deviation_score=parameter_deviations.get(PAIR_TO_RLS_GROUP.get(pair_name), 0.0),
            rls_deviation_threshold=parameter.RLS_DEVIATION_THRESHOLD,
            tp_rr_ratio=parameter.TP_RR_RATIO,
            kalman_velocity=0.0,
            kalman_innovation_zscore=0.0,
            kalman_trend="FLAT",
        )

        trade_signals[pair_name] = signal
        if signal.get("signal") in ("BUY", "SELL") and not is_backtest:
            send_signal_to_trade_engine(
                {
                    "signal_id": f"{pair_name.replace('/', '')}",
                    "action": signal["signal"],
                    "symbol": pair_name,
                    "entry_price": signal.get("entry_price"),
                    "stop_loss": signal.get("stop_loss"),
                    "take_profit": signal.get("take_profit"),
                    "position_units": signal.get("position_units", 0),
                    "snr": signal.get("snr"),
                    "pipeline_run_id": pipeline_run_id_for_monitor,
                },
                log_stream,
            )

    deviation_results = detect_price_deviation(
        log_stream,
        latest_hf_actual_prices,
        rls_forecasts,
        kalman_metrics,
        confidence_level,
    )

    cycle_duration = float(time.time() - cycle_start_time)
    return {
        "cycle_number": cycle_count,
        "timestamp": datetime.now().isoformat(),
        "latest_actual_prices": convert_numpy_floats(latest_hf_actual_prices),
        "rls_health": convert_numpy_floats(rls_metrics),
        "deviation_results": convert_numpy_floats(deviation_results),
        "rls_forecast": format_for_dashboard(rls_forecasts, latest_hf_actual_prices),
        "trade_signals": convert_numpy_floats(trade_signals),
        "parameter_deviations": convert_numpy_floats(parameter_deviations),
        "dcc_metrics": convert_numpy_floats(dcc_group_metrics),
        "kalman_metrics": convert_numpy_floats(kalman_metrics),
        "consensus_metrics": convert_numpy_floats(consensus_metrics),
        "mean_reversion_candidates": convert_numpy_floats(mean_reversion_candidates),
        "pipeline_run_id": pipeline_run_id_for_monitor,
        "global_metrics": {
            "global_confidence": float(global_rls_confidence),
            "global_deviation": float(rls_param_deviation_score),
            "cycle_duration": cycle_duration,
        },
        "cycle_duration_seconds": cycle_duration,
        "news_status": convert_numpy_floats(news_status),
        "is_backtest": bool(is_backtest),
    }

def start_realtime_monitoring(
    total_duration_minutes,
    interval_seconds,
    confidence_level=0.95,
    pipeline_run_id: Optional[str] = None,
    log_output_path: Optional[str] = None
):
    from statsmodels.tsa.statespace.varmax import VARMAXResultsWrapper
    from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
    import numpy as np
    import os

    all_monitoring_results = []
    start_time = time.time()
    end_time = start_time + total_duration_minutes * 60
    cycle_count = 0
    expected_return_state: Dict[str, float] = {}

    if log_output_path:
        log_stream_main = open(log_output_path, 'a')
    else:
        log_stream_main = sys.stdout

    log_stream_main.write(f"\n[INFO] Starting real-time monitoring for {total_duration_minutes} minutes, checking every {interval_seconds} seconds.\n")
    log_stream_main.flush()

    # Initialize MetaTrader5 connection
    log_stream_main.write("[INFO] Initializing MetaTrader5...\n")
    mt5_adapter_instance = MT5Adapter(logger=lambda msg: log_stream_main.write(msg + '\n'))

    if not mt5_adapter_instance.initialize():
        log_stream_main.write(f"[ERROR] MetaTrader5 initialization failed, error code = {mt5_adapter_instance.last_error()}\n")
        log_stream_main.flush()
        if log_output_path: log_stream_main.close()
        mt5_adapter_instance.shutdown()
        return [], "MetaTrader5 initialization failed."

    if not mt5_adapter_instance.login(
        login=MT5_LOGIN,
        password=MT5_PASSWORD,
        server=MT5_SERVER
    ):
        log_stream_main.write(f"[ERROR] MetaTrader5 login failed, error code = {mt5_adapter_instance.last_error()}\n")
        log_stream_main.flush()
        if log_output_path: log_stream_main.close()
        mt5_adapter_instance.shutdown()
        return [], "MetaTrader5 login failed."

    log_stream_main.write("[OK] MetaTrader5 initialized and logged in successfully.\n")
    log_stream_main.flush()


    PAIR_TO_RLS_GROUP = {
       endog.replace("_Close_Log_Return", ""): group
       for group, endogs in parameter.VARX_ENDOG_GROUPS.items()
       for endog in endogs
       if endog.endswith("_Close_Log_Return")
   }

    # Define MT5_TIMEFRAME_MAP using the adapter's properties for fetch_high_frequency_data
    current_mt5_timeframe_map = {
        "1m": mt5_adapter_instance.TIMEFRAME_M1,
        "5m": mt5_adapter_instance.TIMEFRAME_M5,
        "15m": mt5_adapter_instance.TIMEFRAME_M15,
        "30m": mt5_adapter_instance.TIMEFRAME_M30,
        "1h": mt5_adapter_instance.TIMEFRAME_H1,
        "4h": mt5_adapter_instance.TIMEFRAME_H4,
        "1d": mt5_adapter_instance.TIMEFRAME_D1,
        "1w": mt5_adapter_instance.TIMEFRAME_W1,
        "1M": mt5_adapter_instance.TIMEFRAME_MN1,
    }

    pipeline_run_id_for_monitor = pipeline_run_id if pipeline_run_id is not None else "UNKNOWN_RUN_ID"

    from monitoring import NewsManager

    logger_news = lambda msg: (log_stream_main.write(f"[NEWS] {msg}\n"), log_stream_main.flush())
    news_manager_instance = NewsManager(data_dir=VPS_DATA_DIR, logger=logger_news)
    # Sync news to populate data, then load it
    logger_news("Attempting to sync news...")
    news_manager_instance.sync_news()
    logger_news("Loading local news...")
    news_manager_instance.load_local_news()

    restored_price_forecasts_with_intervals = {}
    final_stationarized_fred_data = {}
    fitted_varx_models = {}
    PAIR_TARGET_INDEX = {}
    for group, endogs in parameter.VARX_ENDOG_GROUPS.items():
        for idx, pair in enumerate(endogs):
            PAIR_TARGET_INDEX[pair] = idx

    log_stream_main.write(f"[INFO] Attempting to load data from VPS_DATA_DIR: {VPS_DATA_DIR}\n")
    log_stream_main.flush()

    try:
        forecast_path = parameter.FORECAST_OUTPUT_PATH
        with open(forecast_path, 'rb') as f:
            loaded_data = pickle.load(f)
            restored_price_forecasts_with_intervals = loaded_data.get("data", {})
        log_stream_main.write(f"[INFO] Loaded legacy restored forecasts from {forecast_path} (optional).\n")
    except FileNotFoundError:
        log_stream_main.write(f"[INFO] Legacy forecast data not found at {forecast_path}. Using model-based deviation logic.\n")
    except Exception as e:
        log_stream_main.write(f"[WARN] Failed to load legacy restored forecasts from {forecast_path}: {e}. Continuing with model-based logic.\n")
    log_stream_main.flush()

    try:
        fred_path = parameter.FRED_DATA_PATH
        with open(fred_path, 'rb') as f:
            loaded_data = pickle.load(f)
            final_stationarized_fred_data = loaded_data.get("data", {})
        log_stream_main.write(f"[OK] Successfully loaded final_stationarized_fred_data from {fred_path}\n")
    except FileNotFoundError:
        log_stream_main.write(f"[WARN] FRED data file not found at {fred_path}. Monitoring will be incomplete.\n")
    except Exception as e:
        log_stream_main.write(f"[ERROR] Failed to load final_stationarized_fred_data from {fred_path}: {e}\n")
    log_stream_main.flush()
    model_payload = {}
    try:
        models_path = parameter.FITTED_MODELS_PATH
        with open(models_path, 'rb') as f:
            loaded_data = pickle.load(f)
            model_payload = _extract_ensemble_payload(loaded_data)
        log_stream_main.write(f"[OK] Successfully loaded model payload from {models_path}\n")
    except FileNotFoundError:
        log_stream_main.write(f"[WARN] Fitted models file not found at {models_path}. Parameter update might be incomplete.\n")
    except Exception as e:
        log_stream_main.write(f"[ERROR] Failed to load fitted models from {models_path}: {e}\n")
    log_stream_main.flush()

    ensemble_models = model_payload.get("ensemble", {}) if isinstance(model_payload, dict) else {}
    dcc_model_registry = model_payload.get("dcc_garch", {}) if isinstance(model_payload, dict) else {}

    if not ensemble_models and isinstance(model_payload, dict):
        candidate_flat = {
            k: v for k, v in model_payload.items()
            if isinstance(v, dict) and "fitted_model" in v and "endog_names" in v
        }
        if candidate_flat:
            ensemble_models = {"H1": candidate_flat}

    for tf_name, tf_models in ensemble_models.items():
        if isinstance(tf_models, dict) and str(tf_models.get("model_type", "")).upper().startswith("KALMAN"):
            KALMAN_MODEL_PARAMS[str(tf_name).upper()] = tf_models

    fitted_varx_models = ensemble_models.get("H1", {}) if isinstance(ensemble_models, dict) else {}

    rls_estimators: Dict[str, Dict[str, Any]] = {}
    timeframe_last_close_map: Dict[str, Any] = {}
    dcc_metrics_cache: Dict[str, Dict[str, float]] = {}
    dcc_timeframe_last_close_map: Dict[str, Any] = {}

    if ensemble_models:
        log_stream_main.write(f"[INFO] Initializing adaptive estimators from ensemble payload.\n")
        log_stream_main.flush()

        for tf_name, tf_models in ensemble_models.items():
            if not isinstance(tf_models, dict):
                continue

            for group_name, model_res in tf_models.items():
                if not isinstance(model_res, dict):
                    continue

                model_type = str(model_res.get("model_type", "")).upper()
                fitted_model_obj = model_res.get('fitted_model')
                endog_names_group = model_res.get('endog_names', [])
                exog_names_group = model_res.get('exog_names', [])
                lags_used = int(model_res.get('lags_used', parameter.maxlag_test))

                estimator_key = f"{str(tf_name).upper()}::{group_name}"

                if model_type.startswith("KALMAN"):
                    KALMAN_MODEL_PARAMS[str(tf_name).upper()] = model_res
                    log_stream_main.write(f"  [OK] Kalman params registered for timeframe {tf_name}.\n")
                    continue

                if isinstance(fitted_model_obj, (VARMAXResultsWrapper, SARIMAXResultsWrapper)):
                    try:
                        n_endog_group = len(endog_names_group)
                        k_regressors = 1 + (lags_used * n_endog_group) + len(exog_names_group)
                        baseline_theta_ref = _extract_baseline_varx_params(
                            log_stream_main,
                            fitted_model_obj,
                            n_endog_group,
                            k_regressors,
                            endog_names_group,
                            exog_names_group,
                            lags_used
                        )

                        if baseline_theta_ref is None:
                            log_stream_main.write(f"  [WARN] Failed to extract baseline parameters for {estimator_key}. Skipping RLS init.\n")
                            continue

                        initial_theta = baseline_theta_ref
                        initial_P = parameter.RLS_INITIAL_P_DIAG * np.eye(k_regressors)

                        rls_estimators[estimator_key] = {
                            'theta': initial_theta,
                            'P': initial_P,
                            'baseline_theta_ref': baseline_theta_ref,
                            'n_endog': n_endog_group,
                            'k_regressors': k_regressors,
                            'endog_names': endog_names_group,
                            'exog_names': exog_names_group,
                            'maxlags': lags_used,
                            'rls_update_count': 0,
                            'pred_variance_history': [],
                            'innovation_history': [],
                            'last_update_bar_timestamp': None,
                            'last_Y_t': None,
                            'timeframe': str(tf_name).upper(),
                            'group_name': group_name,
                            'model_type': model_type
                        }
                        log_stream_main.write(
                            f"  [OK] RLS initialized for {estimator_key}. "
                            f"Theta shape: {initial_theta.shape}, P shape: {initial_P.shape}\n"
                        )

                    except Exception as e:
                        log_stream_main.write(f"  [ERROR] Failed to initialize RLS for {estimator_key}: {e}\n")
                else:
                    log_stream_main.write(f"  [WARN] Unsupported model type for {estimator_key}: {model_type}.\n")
                log_stream_main.flush()
    else:
        log_stream_main.write(f"[WARN] No ensemble models provided for estimator initialization.\n")
        log_stream_main.flush()

    if not ensemble_models:
        log_stream_main.write("[ERROR] Ensemble model payload missing. Cannot proceed with monitoring.\n")
        log_stream_main.flush()
        if log_output_path: log_stream_main.close()
        mt5_adapter_instance.shutdown()
        return [], "Ensemble model missing, monitoring aborted."

    try:
        while time.time() < end_time:
            cycle_count += 1
            log_stream_main.write(
                f"\n--- Monitoring Cycle {cycle_count} (Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})---\n"
            )
            log_stream_main.flush()

            current_cycle_results_summary = run_single_monitoring_cycle(
                mt5_adapter_instance=mt5_adapter_instance,
                pipeline_run_id_for_monitor=pipeline_run_id_for_monitor,
                cycle_count=cycle_count,
                log_stream=log_stream_main,
                is_backtest=False,
                interval_seconds=interval_seconds,
                confidence_level=confidence_level,
                monitoring_context={
                    "current_mt5_timeframe_map": current_mt5_timeframe_map,
                    "final_stationarized_fred_data": final_stationarized_fred_data,
                    "rls_estimators": rls_estimators,
                    "dcc_model_registry": dcc_model_registry,
                    "dcc_metrics_cache": dcc_metrics_cache,
                    "dcc_timeframe_last_close_map": dcc_timeframe_last_close_map,
                    "timeframe_last_close_map": timeframe_last_close_map,
                    "expected_return_state": expected_return_state,
                    "pair_to_rls_group": PAIR_TO_RLS_GROUP,
                    "news_manager_instance": news_manager_instance,
                },
            )

            all_monitoring_results.append(current_cycle_results_summary)
            send_monitoring_data_to_colab(current_cycle_results_summary, log_stream_main)

            log_stream_main.write(
                f"--- Monitoring Cycle {cycle_count} (Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}). "
                f"Cycle Duration: {current_cycle_results_summary.get('cycle_duration_seconds', 0.0):.2f} seconds. "
                f"Pipeline Run ID: {pipeline_run_id_for_monitor}---\n"
            )
            log_stream_main.flush()

    except Exception as e:
        log_stream.write(f"[CRITICAL ERROR] Monitoring loop encountered an unhandled exception: {e}\n")
        import traceback
        log_stream.write(traceback.format_exc())
        return all_monitoring_results, f"Critical error during monitoring: {e}"

    finally:
        log_stream.write("\n[INFO] Real-time monitoring finished.\n")
        log_stream.flush()
        if log_output_path: log_stream_main.close()
        mt5_adapter_instance.shutdown()

    return all_monitoring_results, "Monitoring completed successfully."
