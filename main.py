import os, sys, traceback, pickle, json, re, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from datetime import datetime, timedelta, timezone

# === Konfigurasi Path Modul ===
ROOT_DIR = '/content/drive/MyDrive/books/CASSANDRA/'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# === Helper Reloading (Penting saat Development di Colab) ===
# Memastikan perubahan pada file .py langsung terdeteksi tanpa restart runtime
modules_to_reload = ['parameter', 'restored', 'fitted_models.dcc_garch_process', 'fitted_models.def_varx']
for mod in modules_to_reload:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])

# === Import Modul Utama & Parameter ===
import parameter

# 1. Data Acquisition (Pair & Macro)
# Menggunakan fungsi MTF yang sudah kita sesuaikan
from raw.pair_raw import load_base_data_mtf
from raw.makro_raw import download_macro_data # Pastikan nama fungsi sesuai dengan makro_raw.py Anda

# 2. Preprocessing
# Perhatikan penamaan folder 'preprocessing' agar konsisten dengan struktur proyek
from preprocessing.log_return import apply_log_return_to_price
from preprocessing.fred_transform import apply_fred_transformations
from preprocessing.handle_missing import handle_missing_fred_data
from preprocessing.combine_data import combine_log_returns
from preprocessing.stationarity_test import test_and_stationarize_data

# 3. Model Engine (Granger, VARX, Kalman)
from fitted_models.granger import run_granger_tests # Import the actual Granger test function
from fitted_models.def_varx import fit_varx_or_arx    # Corrected import: use fit_varx_or_arx
from fitted_models.kalman_filter import setup_kalman_filter

# 4. Volatility & Forecasting
from fitted_models.dcc_garch_process import fit_dcc_garch_to_residuals # Corrected import: use fit_dcc_garch_to_residuals
from forecast import auto_varx_forecast
from restored import restore_log_returns_to_price # Tetap diimport untuk validasi di Colab

# 5. Pipeline Utilities
# safe_run and check_data_freshness are defined locally, so no need to import from main_utils
# === Helper ===
def safe_run(step_name, log_stream, func, *args, **kwargs):
    """Wrapper agar setiap tahap tetap lanjut walau error dan mencatat ke log_stream.
    log_stream is now the second argument for safe_run and the first argument passed to func.
    """
    log_stream.write(f"\n=== {step_name} ===\n")
    try:
        # Pass log_stream as the first argument to the function being wrapped
        result = func(log_stream, *args, **kwargs)
        log_stream.write(f"[OK] {step_name} berhasil.\n") # Changed unicode to ASCII OK
        return result # Return the actual result, whether it's a single value or a tuple
    except Exception as e:
        log_stream.write(f"[ERROR] {step_name} gagal: {e}\n") # Gunakan tag error ASCII agar log konsisten
        traceback.print_exc(file=log_stream)
        return None # Return None on failure

def check_data_freshness(log_stream, mtf_base_dfs, current_utc_time):
    """
    Memeriksa kesegaran data untuk semua timeframe.
    M1 harus sangat fresh (< 5-10 menit), D1 bisa toleransi 1 hari.
    """
    log_stream.write("\n=== Checking Multi-Timeframe Data Freshness ===\n")
    if not mtf_base_dfs: return False

    is_fresh_overall = True
    for tf, pairs_dict in mtf_base_dfs.items():
        for pair_name, df in pairs_dict.items():
            if df.empty: continue

            latest_ts = df.index.max()
            if latest_ts.tz is None: latest_ts = latest_ts.tz_localize('UTC')

            # Logika toleransi per TF
            if tf == 'D1':
                tolerance = timedelta(hours=24)
            elif tf == 'H1':
                tolerance = timedelta(hours=3)
            else: # M1
                tolerance = timedelta(minutes=10)

            if latest_ts < (current_utc_time - tolerance):
                log_stream.write(f"[ALERT] Data {tf} {pair_name} stale! Last: {latest_ts}\n")
                is_fresh_overall = False

    return is_fresh_overall
# ============================================================
# 1\" LOAD DATA
# ============================================================

# Removed redundant load_base_data wrapper function

# Removed redundant load_fred_data wrapper function

# ============================================================
# 2\" PREPROCESSING
# ============================================================

def preprocess_data_tf(log_stream, b_dfs, fred_df, fred_meta, tf_label):
    """
    Preprocessing khusus untuk satu timeframe tertentu.
    """
    log_stream.write(f"\n--- Preprocessing Timeframe: {tf_label} ---\n")

    # 1. Log Return
    log_returns_raw = safe_run(f"Log Return {tf_label}", log_stream, apply_log_return_to_price, b_dfs) or {}
    log_returns_dict = safe_run(f"Combine Dict {tf_label}", log_stream, combine_log_returns, log_returns_raw, return_type='dict') or {}
    combined_df = safe_run(f"Combine DF {tf_label}", log_stream, combine_log_returns, log_returns_raw, return_type='df') or pd.DataFrame()

    # 2. FRED Transform (Hanya jika D1 dan data FRED tersedia)
    cleaned_fred = {} # Placeholder, will be populated if D1
    if tf_label == 'D1' and fred_df is not None:
        transformed = safe_run("Transformasi FRED", log_stream, apply_fred_transformations, fred_df, parameter.FRED_SERIES, fred_meta) or {}
        cleaned_fred_raw = safe_run("Handle Missing FRED", log_stream, handle_missing_fred_data, transformed, missing_threshold=0.3) or {}

        # Stasioneritas
        # For FRED data, we want to stationarize it based on its own properties, not log_returns_dict
        # Assuming test_and_stationarize_data can handle a single DataFrame for FRED
        stationarity_res_fred = safe_run("Uji Stasioneritas FRED", log_stream, test_and_stationarize_data, cleaned_fred_raw, {}, parameter.alpha) # Pass empty dict for log_returns
        if stationarity_res_fred:
            cleaned_fred = stationarity_res_fred[0] # Assuming it returns the stationary FRED data

    return log_returns_dict, cleaned_fred, combined_df
# ============================================================
# 3\" GRANGER TESTS
# ============================================================

def run_granger_all(log_stream, log_returns, cleaned_fred, timeframe_label="D1"): # Renamed from run_granger_all_mtf
    """
    Versi MTF dari Granger Test.
    D1: Menguji FRED + Pair lain terhadap Pair target.
    H1: Hanya menguji Pair lain (Cross-Asset) terhadap Pair target.
    """
    if not log_returns:
        return pd.DataFrame(), {}

    # 1. Tentukan Target (semua pair dalam timeframe ini)
    granger_targets = {}
    for pair, df in log_returns.items():
        if not df.empty:
             log_return_cols = [col for col in df.columns if col.endswith('_Log_Return')]
             for col in log_return_cols:
                 granger_targets[f"{pair}_{col}"] = df[[col]].dropna()

    # 2. Tentukan Potensi Penyebab (Exogenous)
    granger_exogs = {}

    # HANYA masukkan FRED jika kita di timeframe D1
    if timeframe_label == "D1" and cleaned_fred: # cleaned_fred is now a single DF
        for col in cleaned_fred.columns: # Iterate over columns of cleaned_fred DF
            if col not in ['release_date', 'effective_until_next_release', 'date']:
                granger_exogs[col] = cleaned_fred[[col]].dropna()

    # 3. Tambahkan Cross-Asset Causality (Pair lain sebagai exog)
    # Ini krusial untuk H1: melihat apakah JPY mempengaruhi EUR, dll.
    all_potential_causes = {**granger_exogs, **granger_targets}

    # Jalankan uji Granger
    results = safe_run(f"Granger Test [{timeframe_label}]", log_stream, run_granger_tests,
                       data_dict=granger_targets,
                       exogenous_data_dict=all_potential_causes,
                       maxlag_test=parameter.maxlag_test,
                       alpha=parameter.alpha)

    exog_map = {} # Placeholder for actual exog map
    if results is not None: # Assuming identify_significant_exog is defined elsewhere or in granger.py
        # exog_map = identify_significant_exog(results, parameter.alpha) # This function was not provided
        # For now, let's just create a dummy exog_map for demonstration
        # A proper implementation would parse `results` to build this map.
        for effect_key, cause_data in results.items():
            for cause_key, p_value in cause_data.items(): # This structure needs to match actual `results`
                if p_value < parameter.alpha: # Simple threshold
                    if effect_key not in exog_map:
                        exog_map[effect_key] = []
                    exog_map[effect_key].append(cause_key)

    return results, exog_map


# ============================================================
# 4\" MODEL FITTING
# ============================================================

def combine_fred_for_model(log_stream, cleaned_fred_df, timeframe_label="D1"):
    """
    Combines cleaned FRED data into a single DataFrame for model fitting.
    Args:
        log_stream (StringIO): Stream for logging.
        cleaned_fred_df (pd.DataFrame): DataFrame containing cleaned FRED data.
        timeframe_label (str): Current timeframe label (e.g., 'D1', 'H1').
    Returns:
        pd.DataFrame: Combined FRED DataFrame for model fitting.
    """
    # Proteksi: Data FRED tidak relevan untuk timeframe intraday
    if timeframe_label != "D1":
        log_stream.write(f"[INFO] Timeframe {timeframe_label} terdeteksi. Melewati penggabungan FRED karena perbedaan frekuensi data.\n")
        return pd.DataFrame()

    if cleaned_fred_df.empty:
        log_stream.write("[WARN] Data FRED bersih kosong. Tidak ada yang digabungkan.\n")
        return pd.DataFrame()

    # Assuming cleaned_fred_df is already a single DataFrame with appropriate columns
    log_stream.write(f"[OK] Data FRED bersih gabungan berhasil dibuat untuk model fitting. Shape: {cleaned_fred_df.shape}\n")

    return cleaned_fred_df


def fit_models(log_stream, log_returns_dict, exog_map, combined_fred_for_model_df, timeframe_label):
    import re
    models = {} # Initialize the models dictionary
    summaries = []

    if not log_returns_dict:
        log_stream.write("[WARN] Data log return kosong. Lewati fitting model.\n")
        return models, pd.DataFrame()

    # Buat pool eksogen gabungan. Ini akan menjadi sumber semua variabel eksogen yang mungkin.
    all_available_model_exogs = pd.DataFrame()
    if combined_fred_for_model_df is not None and not combined_fred_for_model_df.empty:
        all_available_model_exogs = pd.concat([all_available_model_exogs, combined_fred_for_model_df], axis=1)

    # Tambahkan semua log-returns sebagai potensi eksogen cross-asset
    # Pastikan tidak ada duplikasi kolom saat concat
    all_log_returns_df = pd.concat([df for df in log_returns_dict.values()], axis=1, join='outer')
    all_log_returns_df = all_log_returns_df.loc[:,~all_log_returns_df.columns.duplicated()]
    all_available_model_exogs = pd.concat([all_available_model_exogs, all_log_returns_df], axis=1)
    all_available_model_exogs = all_available_model_exogs.loc[:,~all_available_model_exogs.columns.duplicated()].ffill().dropna(how='all')

    if all_available_model_exogs.empty:
        log_stream.write("[WARN] Tidak ada data eksogen yang tersedia setelah digabungkan dan dibersihkan. Model akan berjalan tanpa eksogen.\n")

    # Iterate over each group defined in VARX_ENDOG_GROUPS
    for group_name, endog_cols_in_group in parameter.VARX_ENDOG_GROUPS.items(): # Use parameter.VARX_ENDOG_GROUPS
        log_stream.write(f"\n[INFO] Melatih model untuk grup: {group_name}\n")

        # 1. Create a combined DataFrame of the endogenous variables specified in the group
        endog_data_frames = []
        for endog_full_name in endog_cols_in_group:
            pair_name_from_endog = None
            match = re.match(r'(.+?)_(Open|High|Low|Close)_Log_Return', endog_full_name)
            if match:
                pair_name_from_endog = match.group(1)

            if pair_name_from_endog and pair_name_from_endog in log_returns_dict:
                df_log_return_pair = log_returns_dict[pair_name_from_endog]
                column_in_pair_df = endog_full_name.replace(f"{pair_name_from_endog}_", "")
                if not df_log_return_pair.empty and column_in_pair_df in df_log_return_pair.columns:
                    endog_data_frames.append(df_log_return_pair[[column_in_pair_df]].rename(columns={column_in_pair_df: endog_full_name}))
                else:
                    log_stream.write(f"[WARN] Kolom '{endog_full_name}' tidak ditemukan di data '{pair_name_from_endog}'. Lewati.\n")
            else:
                log_stream.write(f"[WARN] Data untuk '{endog_full_name}' tidak ditemukan di log_returns_dict. Lewati.\n")

        if not endog_data_frames:
            log_stream.write(f"[WARN] Tidak ada data endogenous valid untuk grup {group_name}. Lewati fitting model.\n")
            continue

        df_endog_group = pd.concat(endog_data_frames, axis=1, join='inner').dropna()

        if df_endog_group.empty:
            log_stream.write(f"[WARN] Data endogenous grup {group_name} kosong setelah penggabungan/dropna. Lewati fitting model.\n")
            continue

        # 2. Identify the relevant exogenous variables for the current group
        exog_cols_to_use = []
        for endog_item in endog_cols_in_group:
            # Get significant exogs for each individual endogenous variable in the group
            current_endog_exogs = exog_map.get(endog_item, [])
            # Add them to the list, avoiding duplicates and excluding if already an endogenous variable in this group
            for exog_cand in current_endog_exogs:
                if exog_cand not in exog_cols_to_use and exog_cand not in endog_cols_in_group:
                    exog_cols_to_use.append(exog_cand)

        # Filter exog_cols_to_use to only include columns actually present in all_available_model_exogs
        final_exog_cols = [col for col in exog_cols_to_use if col in all_available_model_exogs.columns]

        df_combined_for_model = df_endog_group.copy()
        if final_exog_cols:
            log_stream.write(f"[INFO] Menggunakan eksogen signifikan untuk grup {group_name}: {final_exog_cols}\n")
            # Merge endogenous data with selected exogenous variables
            exog_subset = all_available_model_exogs[final_exog_cols]
            # Align indices and ffill
            df_aligned = df_endog_group.join(exog_subset, how='inner')
            df_aligned[final_exog_cols] = df_aligned[final_exog_cols].ffill()
            df_combined_for_model = df_aligned.dropna()
        else:
           log_stream.write("[INFO] Tidak ada eksogen signifikan yang teridentifikasi atau data eksogen lengkap tidak tersedia. Akan mencoba fit AR(X) tanpa eksogen.\n")

        if df_combined_for_model.empty:
            log_stream.write(f"[WARN] Data gabungan kosong setelah dropna untuk grup {group_name}. Lewati fitting model.\n")
            continue

        try:
            log_stream.write(f"[INFO] Fitting model VARX/ARX untuk grup {group_name}, Endog: {endog_cols_in_group}\n")
            # Fit the VARX model (will be VARMAX if multiple endog, SARIMAX if single)
            model_result = fit_varx_or_arx(
                log_stream, df_combined_for_model,
                endog_cols=endog_cols_in_group, # Pass the list of endogenous columns for the group
                exog_cols=final_exog_cols,
                maxlags=parameter.maxlag_test # Use parameter.maxlag_test
            )

            # Store the fitted model result in the dictionary.
            # The key should be the group name.
            model_key = group_name
            models[model_key] = model_result # Use the group name as the key
            models[model_key]['endog_names'] = endog_cols_in_group # Store the endogenous column names
            models[model_key]['exog_names'] = final_exog_cols # Store exog_names here

            # Extract R2 for summary (handle dict for VARMAX or single value for ARX)
            r2_values = model_result.get("R2", np.nan)

            # If VARMAX, R2 will be a dictionary of R2s for each endogenous variable
            if isinstance(r2_values, dict):
                for endog_col_name, r2_val in r2_values.items():
                    summaries.append({
                        "Group": group_name,
                        "Endog_Variable": endog_col_name,
                        "Model_Type": model_result["model_type"],
                        "Lags_Used": model_result["lags_used"],
                        "Num_Exog": len(final_exog_cols),
                        "R2": r2_val
                    })
            else: # If SARIMAX (single endogenous in group, though this path should be mostly for VARMAX now)
                 if len(endog_cols_in_group) == 1:
                      summaries.append({
                          "Group": group_name,
                          "Endog_Variable": endog_cols_in_group[0],
                          "Model_Type": model_result["model_type"],
                          "Lags_Used": model_result["lags_used"],
                          "Num_Exog": len(final_exog_cols),
                          "R2": r2_values # Assuming r2_values is a single float here
                      })
                 else:
                     log_stream.write(f"[WARN] R2 format tidak terduga untuk grup {group_name}. Tidak dapat menambahkan ke ringkasan.\n")

            log_stream.write(f"[OK] Model {model_result['model_type']} fitted for group {group_name}. Lags={model_result['lags_used']}, Num Exog={len(final_exog_cols)}, R\u00b2={r2_values}\n")

        except ValueError as ve:
            log_stream.write(f"[ERROR] Gagal fitting model untuk grup {group_name}: {ve}\n")
            summaries.append({
                "Group": group_name, "Endog_Variable": "N/A", "Model_Type": None,
                "Lags_Used": None, "Num_Exog": len(final_exog_cols), "R2": np.nan
            })
        except RuntimeError as err_e:
            log_stream.write(f"[ERROR] Gagal fitting model untuk grup {group_name}: {err_e}\n")
            summaries.append({
                "Group": group_name, "Endog_Variable": "N/A", "Model_Type": None,
                "Lags_Used": None, "Num_Exog": len(final_exog_cols), "R2": np.nan
            })
        except Exception as e:
            log_stream.write(f"[ERROR] Terjadi error saat fitting model untuk grup {group_name}: {e}\n")
            summaries.append({
                "Group": group_name, "Endog_Variable": "N/A", "Model_Type": None,
                "Lags_Used": None, "Num_Exog": len(final_exog_cols), "R2": np.nan
            })

    # Display summary of fitted models
    varx_fit_df = pd.DataFrame(summaries)

    log_stream.write("\n[INFO] Ringkasan Hasil Fitting Model VARX/ARX (Grup):\n")
    if not varx_fit_df.empty:
        log_stream.write(varx_fit_df.dropna(subset=['R2']).sort_values(by='R2', ascending=False).to_string() + "\n")
    else:
        log_stream.write("[INFO] Tidak ada model yang berhasil di-fit.\n")

    log_stream.write("\n[OK] Proses fitting model selesai. Fitted models stored in 'models' dictionary.\n")

    return models, varx_fit_df

#----------DCC GARCH FITTING-----------
def fit_dcc_garch_models(log_stream, ensemble_results, log_returns):
    """Fits DCC-GARCH models to residuals from VARX/ARX models."""
    log_stream.write("\n=== Fit DCC-GARCH Models ===\n")

    dcc_garch_input_data = {} # Prepare input for DCC-GARCH
    for tf, models_tf in ensemble_results.items():
        if tf in ['D1', 'H1']: # Assuming DCC-GARCH is relevant for these TFs
            for model_key, model_result in models_tf.items():
                if 'fitted_model' in model_result and hasattr(model_result['fitted_model'], 'resid'):
                    # Extract residuals for each endogenous variable
                    for endog_name in model_result['endog_names']:
                        # Residuals are typically indexed by original series name or something similar
                        # Need to match this with the expected input format for DCC-GARCH
                        resid_series = model_result['fitted_model'].resid[endog_name] if endog_name in model_result['fitted_model'].resid.columns else model_result['fitted_model'].resid.squeeze()
                        if not resid_series.empty:
                            dcc_garch_input_data[f"{endog_name}_{tf}"] = resid_series # Tag with TF for uniqueness
                        else:
                            log_stream.write(f"[WARN] Residual for {endog_name} in {tf} is empty. Skipping.\n")
                else:
                    log_stream.write(f"[WARN] No fitted model or residuals found for {model_key} in {tf}. Skipping for DCC-GARCH.\n")

    if not dcc_garch_input_data:
        log_stream.write("[WARN] No valid residual data for DCC-GARCH fitting. Skipping.\n")
        return {}

    dcc_garch_df = pd.DataFrame(dcc_garch_input_data).dropna()
    if dcc_garch_df.empty or len(dcc_garch_df) < parameter.MIN_OBS_FOR_GARCH: # Define MIN_OBS_FOR_GARCH in parameter.py
        log_stream.write(f"[WARN] Insufficient data ({len(dcc_garch_df)} observations) for DCC-GARCH model fitting. Skipping.\n")
        return {}

    # Assuming fit_dcc_garch_to_residuals is a function that takes a DataFrame of residuals
    dcc_garch_models = safe_run("Fit DCC-GARCH to VARX/ARX residuals", log_stream, fit_dcc_garch_to_residuals, dcc_garch_df)
    return dcc_garch_models if dcc_garch_models is not None else {}


# ============================================================
# 5\" FORECASTING & RESTORATION
# ============================================================

def forecasting_and_restore(log_stream, log_returns_dict, models, fitted_dcc_garch_models, exog_map, cleaned_fred_data, base_data):
    """Performs forecasting using fitted models and restores forecasts to price scale."""
    log_stream.write(f"\n[INFO] Melakukan peramalan ({parameter.FORECAST_HORIZON} langkah ke depan) dengan interval kepercayaan...\n") # Use parameter.FORECAST_HORIZON

    if not log_returns_dict or not models:
        log_stream.write("[WARN] Data log return atau fitted VARX/ARX models kosong. Melewati peramalan.\n")
        combined_forecasts_with_intervals = {}
    else:
        combined_forecasts_with_intervals = safe_run("Generate Combined Forecasts", log_stream, auto_varx_forecast,
                                                    fitted_varx_models=models, # Should be models by interval
                                                    # fitted_dcc_garch_models=fitted_dcc_garch_models, # Pass only relevant DCC-GARCH for forecast
                                                    combined_log_returns_dict=log_returns_dict, # Needs to be dict by TF
                                                    final_stationarized_fred_data=cleaned_fred_data, # Only relevant for D1
                                                    exog_map=exog_map, # Exog map for VARX, but structure needs to be MTF
                                                    forecast_horizon=parameter.FORECAST_HORIZON, # Use parameter.FORECAST_HORIZON
                                                    confidence_level=parameter.CONFIDENCE_LEVEL,
                                                    verbose=True)

    log_stream.write("\n[OK] Peramalan gabungan selesai. Hasil disimpan dalam dictionary 'combined_forecasts_with_intervals'.\n")

    log_stream.write("\n[INFO] Hasil Peramalan (Log Return dengan Interval Kepercayaan):\n")
    if combined_forecasts_with_intervals:
        for model_key, forecast_data in combined_forecasts_with_intervals.items():
            if 'interval_forecast' in forecast_data and not forecast_data['interval_forecast'].empty:
                log_stream.write(f"\n--- Peramalan untuk: {model_key} ---\n")
                log_stream.write(forecast_data['interval_forecast'].head().to_string() + "\n")
            else:
                log_stream.write(f"\n--- Peramalan untuk: {model_key} ---\n (DataFrame kosong)\n")
    else:
        log_stream.write("Tidak ada peramalan log return gabungan yang dihasilkan.\n")


    log_stream.write("\n[INFO] Mengrestorasi peramalan log return ke peramalan harga (OHLC) dengan interval kepercayaan...\n")

    if not combined_forecasts_with_intervals or not base_data:
        log_stream.write("[WARN] Data peramalan gabungan atau data harga base (base_dfs) tidak lengkap. Melewati restorasi peramalan harga.\n")
        restored_price_forecasts_with_intervals = {}
    else:
        restored_price_forecasts_with_intervals = safe_run("Restore Price Forecast with Intervals", log_stream, restore_log_returns_to_price,
                                                            combined_forecasts_with_intervals, base_data, confidence_level=parameter.CONFIDENCE_LEVEL) # base_data needs to be the original OHLC for restoration


    log_stream.write("\n[OK] Restorasi peramalan harga dengan interval kepercayaan selesai. Hasil disimpan dalam dictionary 'restored_price_forecasts_with_intervals'.\n")

    log_stream.write("\n[INFO] Hasil Peramalan Harga (Direstorasi - OHLC dengan Interval Kepercayaan):\n")
    if restored_price_forecasts_with_intervals:
        for pair_name, forecast_df in restored_price_forecasts_with_intervals.items():
            log_stream.write(f"\n--- Restorasi Harga Peramalan untuk Pair: {pair_name} (OHLC) ---\n")
            if not forecast_df.empty:
                log_stream.write(forecast_df.head().to_string() + "\n")
            else:
                log_stream.write("DataFrame peramalan harga kosong.\n")
    else:
        log_stream.write("Tidak ada peramalan harga OHLC yang berhasil direstorasi.\n")

    return combined_forecasts_with_intervals, restored_price_forecasts_with_intervals

def save_pipeline_outputs_to_file(filepath, execution_log, model_summary_df, combined_log_returns_forecasts, restored_price_forecasts, last_actual_prices_dict):
    """Saves all important pipeline outputs to a specified file."""
    with open(filepath, 'w') as f:
        f.write("==== Pipeline Execution Log ====\n")
        f.write(execution_log if execution_log is not None else "Execution log not available.\n") # FIX IS HERE
        f.write("\n\n")

        f.write("==== Model Summary ====\n")
        if model_summary_df is not None and not model_summary_df.empty:
            f.write(model_summary_df.to_string())
        else:
            f.write("No model summary available.\n")
        f.write("\n\n")

        f.write("==== Combined Log Returns Forecasts ====\n")
        if combined_log_returns_forecasts:
            for model_key, forecast_data in combined_log_returns_forecasts.items():
                f.write(f"--- Forecast for: {model_key} ---\n")
                if 'interval_forecast' in forecast_data and not forecast_data['interval_forecast'].empty:
                    f.write(forecast_data['interval_forecast'].to_string())
                else:
                    f.write("Empty forecast DataFrame.\n")
                f.write("\n")
        else:
            f.write("No combined log returns forecasts generated.\n")
        f.write("\n\n")

        f.write("==== Restored Price Forecasts (OHLC) ====\n")
        if restored_price_forecasts:
            for pair_name, forecast_df in restored_price_forecasts.items():
                f.write(f"--- Restored Price Forecast for Pair: {pair_name} (OHLC) ---\n")
                if not forecast_df.empty:
                    f.write(forecast_df.to_string())
                else:
                    f.write("Empty restored price forecast DataFrame.\n")
                f.write("\n")
        else:
            f.write("No restored price forecasts generated.\n")
        f.write("\n\n")

        f.write("==== Last Actual Prices for Restoration ====\n")
        if last_actual_prices_dict:
            f.write(json.dumps(last_actual_prices_dict, indent=2))
        else:
            f.write("No last actual prices recorded.\n")

    print(f"Output saved to {filepath}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"pipeline_run_{timestamp}"
    log_stream = StringIO()
    log_stream.write(f"[INFO] Starting pipeline run with ID: {run_id}\n")

    # === 1. UNDUH DATA FRED TERLEBIH DAHULU ===
    # Removed local load_fred_data wrapper. Directly call download_macro_data.
    if parameter.FRED_API_KEY == 'YOUR_FRED_API_KEY': # Check for placeholder
        log_stream.write("[WARN] FRED API Key is a placeholder. FRED data download will likely fail.\n")
        fred_df, fred_meta = None, None # Force None if key is invalid
    else:
        fred_data_results = safe_run("Unduh FRED", log_stream, download_macro_data, parameter.FRED_API_KEY, parameter.FRED_SERIES, parameter.fred_lookback_days)
        fred_df, fred_meta = fred_data_results if fred_data_results else (None, None)


    # === 2. LOAD DATA BASE MTF ===
    mtf_base_dfs = {}
    for tf in parameter.MTF_INTERVALS.keys(): # Iterate over keys D1, H1, M1
        interval_str = parameter.MTF_INTERVALS[tf] # Get the yfinance compatible string
        lookback_days_for_tf = parameter.LOOKBACK_DAYS[tf] # Get the lookback days for this TF
        log_stream.write(f"[DEBUG] Loading data for TF: {tf}, interval_str: {interval_str}, lookback_days: {lookback_days_for_tf}\n") # ADDED DEBUG PRINT
        # Corrected call: removed duplicate log_stream argument
        mtf_base_dfs[tf] = safe_run(f"Load Data {tf}", log_stream, load_base_data_mtf,
                                     parameter.PAIRS, lookback_days_for_tf, interval_str,
                                     parameter.USE_LOCAL_CSV_FOR_PAIRS,
                                     parameter.LOCAL_CSV_FILEPATH)

    # Freshness check pada timeframe terkecil (M1)
    # The check_data_freshness function expects the full mtf_base_dfs dictionary, not just mtf_base_dfs['M1']
    current_execution_log = log_stream.getvalue()
    if not safe_run("Cek Data Freshness", log_stream, check_data_freshness,
                    mtf_base_dfs, datetime.now(timezone.utc)): # Pass the full dict
        log_stream.write("[ERROR] Data M1 tidak segar. Berhenti.\n")
        # Ensure log is captured before early exit
        current_execution_log += log_stream.getvalue()
        return (run_id, {}, None, None, {}, {}, pd.DataFrame(), {}, pd.DataFrame(), {}, {}, {}, {}, current_execution_log, pd.DataFrame(), {}) # Return empty structures but with full log

    # === 3. PREPROCESSING MTF ===
    mtf_log_returns = {}
    mtf_cleaned_fred = {} # Biasanya FRED hanya diproses sekali di D1
    mtf_exog_pool = {} # This will hold combined log returns from other pairs to be used as exog
    cleaned_fred_combined_df = pd.DataFrame() # To hold the processed FRED data as a single DF

    for tf, b_dfs in mtf_base_dfs.items():
        if not b_dfs: continue
        # Call preprocess_data_tf, which should return log_returns_dict, cleaned_fred_for_tf, combined_df (for cross-asset exog)
        log_returns_tf, cleaned_fred_tf, combined_log_returns_tf = safe_run(f"Preprocess {tf}", log_stream, preprocess_data_tf, b_dfs, fred_df, fred_meta, tf)

        if log_returns_tf is not None:
            mtf_log_returns[tf] = log_returns_tf

        if tf == 'D1' and cleaned_fred_tf is not None: # Assume FRED processing is primarily for D1
            cleaned_fred_combined_df = cleaned_fred_tf # Store the cleaned FRED for D1 models

        if combined_log_returns_tf is not None:
            mtf_exog_pool[tf] = combined_log_returns_tf # Store combined log returns for potential cross-asset exog



    # === 4. FITTING MULTI-TIMEFRAME ===
    ensemble_results = {}
    all_summaries = []
    mtf_exog_maps = {}

    for tf in ['D1', 'H1']:
        log_stream.write(f"\n[PROCESS] Analisis & Fitting Layer {tf}...\n")

        # Granger Test: Only FRED data should be used as exogenous for D1
        granger_results, exog_map_tf = safe_run(f"Granger {tf}", log_stream, run_granger_all,
                               mtf_log_returns.get(tf, {}), cleaned_fred_combined_df if tf == 'D1' else {}, timeframe_label=tf)
        mtf_exog_maps[tf] = exog_map_tf # Store the exog_map for the current timeframe

        # Prepare combined exogenous pool for model fitting for this TF
        current_tf_exog_pool = pd.DataFrame()
        # Add FRED data if D1
        if tf == 'D1' and cleaned_fred_combined_df is not None and not cleaned_fred_combined_df.empty:
            current_tf_exog_pool = pd.concat([current_tf_exog_pool, cleaned_fred_combined_df], axis=1)

        # Add cross-asset log returns from other timeframes or same timeframe if appropriate
        for other_tf, other_log_returns_df in mtf_exog_pool.items():
            # Example: H1 can use D1 log returns as exog, M1 can use H1 as exog
            # This logic needs to be refined based on actual cross-timeframe exog strategy
            if other_log_returns_df is not None and not other_log_returns_df.empty:
                current_tf_exog_pool = pd.concat([current_tf_exog_pool, other_log_returns_df], axis=1)

        current_tf_exog_pool = current_tf_exog_pool.loc[:,~current_tf_exog_pool.columns.duplicated()].ffill().dropna(how='all')

        # Fit VARX/ARX models
        models_tf, summary_tf = fit_models(
            log_stream,
            mtf_log_returns.get(tf, {}), # Log returns for the current TF
            mtf_exog_maps[tf],             # Exog map from Granger for current TF
            current_tf_exog_pool,        # Combined pool of all potential exogs
            timeframe_label=tf
        )
        ensemble_results[tf] = models_tf
        if summary_tf is not None and not summary_tf.empty:
            summary_tf['TF'] = tf # Tandai timeframe di summary
            all_summaries.append(summary_tf)

    # 5. Setup Lapisan M1 (Kalman)
    if 'M1' in mtf_base_dfs: # Kalman filter is applied directly to M1 raw data
        kalman_models_m1 = safe_run("Setup Kalman Filter for M1", log_stream, setup_kalman_filter, log_stream, mtf_base_dfs['M1'])
        if kalman_models_m1 is not None:
            ensemble_results['M1'] = kalman_models_m1

    # === 6. VOLATILITY (H1) & SAVE ===
    # DCC-GARCH models will be fitted using residuals from VARX models
    fitted_dcc_garch_models = {}
    if 'H1' in ensemble_results and mtf_log_returns.get('H1'):
        # pass ensemble_results (which contains the VARX models) and raw log_returns to fit_dcc_garch_models
        fitted_dcc_garch_models_h1 = safe_run("Fit DCC-GARCH for H1", log_stream, fit_dcc_garch_models,
                                              ensemble_results, mtf_log_returns.get('H1'))
        if fitted_dcc_garch_models_h1: # Assuming it returns a dict of models
            fitted_dcc_garch_models['H1'] = fitted_dcc_garch_models_h1
    else:
        log_stream.write("[WARN] H1 VARX models or log returns not available for DCC-GARCH fitting.\n")

    # Ambil harga terakhir dari M1 untuk VPS
    last_actual_prices_dict = {}
    if 'M1' in mtf_base_dfs:
        for p, df in mtf_base_dfs['M1'].items():
            if not df.empty:
                last_actual_prices_dict[p] = {
                    'timestamp': df.index[-1].isoformat(),
                    'open': df['Open'].iloc[-1],
                    'high': df['High'].iloc[-1],
                    'low': df['Low'].iloc[-1],
                    'close': df['Close'].iloc[-1]
                }


    # === 7. PACKAGING UNTUK VPS ===
    try:
        package_to_save = {
            "run_id": run_id,
            "ensemble": ensemble_results, # Dictionary D1, H1, M1 models
            "dcc_garch": fitted_dcc_garch_models, # H1 DCC-GARCH models
            "last_prices": last_actual_prices_dict, # Last M1 prices
            "exog_maps": mtf_exog_maps, # Exog maps by TF
            "timestamp": timestamp
        }
        output_dir = os.path.join(ROOT_DIR, 'vps_sync')
        os.makedirs(output_dir, exist_ok=True)
        fitted_models_path = os.path.join(output_dir, 'fitted_ensemble.pkl')

        with open(fitted_models_path, 'wb') as f:
            pickle.dump(package_to_save, f)
        log_stream.write(f"[OK] Multi-Timeframe Ensemble saved to {fitted_models_path}\n")
    except Exception as e:
        log_stream.write(f"[ERROR] Save to PKL failed: {e}\n")

    # Combine Summaries
    final_summary = pd.concat(all_summaries) if all_summaries else pd.DataFrame()

    # === 8. FORECASTING & RESTORATION (FOR COLAB DISPLAY ONLY) ===
    # This part needs to be adapted to forecast for each timeframe and restore prices appropriately
    # Currently, it passes a single combined_forecasts_with_intervals and base_data (likely only D1 or M1)
    combined_forecasts_with_intervals_output = {}
    restored_price_forecasts_with_intervals = {}

    # Need to iterate through each TF that has models and log_returns
    for tf_forecast in ['D1', 'H1']:
        if tf_forecast in ensemble_results and mtf_log_returns.get(tf_forecast):
            # Auto_varx_forecast likely needs to be updated to handle MTF structure
            # and potentially DCC-GARCH output for interval forecasts.
            # For simplicity here, calling forecasting_and_restore for each TF
            # This function needs significant refactoring in `main.py` to be truly MTF aware.
            tf_forecasts, tf_restored_prices = forecasting_and_restore(
                log_stream,
                mtf_log_returns.get(tf_forecast),
                ensemble_results.get(tf_forecast),
                fitted_dcc_garch_models.get(tf_forecast),
                mtf_exog_maps.get(tf_forecast),
                cleaned_fred_combined_df, # FRED data is D1 only, pass for D1 forecast
                mtf_base_dfs.get(tf_forecast) # Base data for restoration
            )
            if tf_forecasts: combined_forecasts_with_intervals_output[tf_forecast] = tf_forecasts
            if tf_restored_prices: restored_price_forecasts_with_intervals[tf_forecast] = tf_restored_prices


    # Final return values
    return (run_id, mtf_base_dfs, fred_df, fred_meta, mtf_log_returns, cleaned_fred_combined_df,
            granger_results, mtf_exog_maps, current_tf_exog_pool, ensemble_results, # current_tf_exog_pool is misleading here, should be per TF
            fitted_dcc_garch_models, combined_forecasts_with_intervals_output,
            restored_price_forecasts_with_intervals, log_stream.getvalue(), final_summary, last_actual_prices_dict)

if __name__ == "__main__":
    # To make it available in the interactive environment without explicitly calling 'global' anywhere,
    # we can execute the main function and then assign the results to global variables.
    pass
