# ☴☴ GRANGER TEST: Bidirectional Makro ↔ Pair (Module)
# ============================================================
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import numpy as np

warnings.filterwarnings("ignore")

def find_best_granger_lag(data, cause, effect, maxlag):
    """Return best lag (1..maxlag) that gives smallest p-value for ssr_ftest, plus the p-value.
    data should be a DataFrame with columns [effect, cause].
    """
    best_p = 1.0
    best_lag = None
    for lag in range(1, maxlag+1):
        try:
            # Ensure data has enough observations for the current lag
            if len(data) < lag + 2:
                 continue # Skip if not enough data for this lag

            res = grangercausalitytests(data[[effect, cause]], maxlag=lag, verbose=False)
            # Check if the result for the current lag exists and has the ssr_ftest p-value
            if lag in res and res[lag] and res[lag][0] and 'ssr_ftest' in res[lag][0] and len(res[lag][0]['ssr_ftest']) > 1:
                 p = res[lag][0]['ssr_ftest'][1]
                 if p < best_p:
                     best_p = p
                     best_lag = lag
            # Handle cases where grangercausalitytests might return unexpected structure
            elif lag in res and res[lag] and len(res[lag]) > 0:
                 # Try to find a p-value in other test statistics if ssr_ftest is missing
                 for test_result in res[lag]:
                      if isinstance(test_result, dict):
                           for test_name, test_values in test_result.items():
                                if isinstance(test_values, (list, tuple)) and len(test_values) > 1:
                                     p = test_values[1]
                                     if p < best_p:
                                         best_p = p
                                         best_lag = lag


        except Exception as e:
             # print(f"\u26a0\ufe0f Error finding lag {lag} for {cause} -> {effect}: {e}")
             continue # Continue to next lag even if one fails
    return best_lag, best_p

def run_granger_tests(log_stream, data_dict, maxlag_test, alpha, target_columns_suffix='_Close_Log_Return', exogenous_data_dict=None):
    """
    Runs Granger causality tests between specific series within data_dict or between data_dict
    and exogenous_data_dict, focusing on columns ending with target_columns_suffix.

    Args:
        log_stream (StringIO): Stream to write log messages.
        data_dict (dict): Dictionary of pandas DataFrames, where keys are pair names
                          and values are DataFrames containing multiple series (e.g., OHLC log returns).
        maxlag_test (int): Maximum lag for Granger test.
        alpha (float): Significance level.
        target_columns_suffix (str): Suffix to identify the target columns within each pair's DataFrame.
                                     Defaults to '_Close_Log_Return'.
        exogenous_data_dict (pd.DataFrame, optional): A single DataFrame with exogenous data
                                                      (e.g., FRED data). If provided, tests are run
                                                      from its columns to data_dict's target columns.

    Returns:
        pd.DataFrame: DataFrame containing significant Granger test results.
    """
    log_stream.write("\n[INFO] Menjalankan Granger causality tests...\n")

    all_granger_results = []

    if not data_dict:
        log_stream.write("  [WARN] Input data_dict is empty. Skipping Granger tests.\n")
        return pd.DataFrame()

    # Prepare all potential effect series (full column names and their single-column DataFrames)
    effect_series_pool = {}
    for pair_name, df_pair in data_dict.items():
        if not isinstance(df_pair, pd.DataFrame) or df_pair.empty:
            continue
        for col_name in df_pair.columns:
            if col_name.endswith(target_columns_suffix):
                effect_series_pool[col_name] = df_pair[[col_name]].dropna()

    if not effect_series_pool:
        log_stream.write(f"  [WARN] No series ending with '{target_columns_suffix}' found in data_dict for effects. Skipping Granger tests.\n")
        return pd.DataFrame()

    # Determine potential cause series
    cause_series_pool = {}
    test_type = "Within-DataDict" # Default to testing within the provided data_dict

    # Add exogenous data from FRED (if D1) or other exogenous sources
    if exogenous_data_dict is not None and not exogenous_data_dict.empty:
        for col_name in exogenous_data_dict.columns:
            if col_name not in ['release_date', 'effective_until_next_release', 'date']: # Exclude non-numeric date columns
                cause_series_pool[col_name] = exogenous_data_dict[[col_name]].dropna()
        test_type = "Exogenous -> Target" # Indicate FRED is present

    # Add other pair's Close_Log_Return columns from data_dict itself as potential causes
    for pair_name, df_pair in data_dict.items():
        if not isinstance(df_pair, pd.DataFrame) or df_pair.empty:
            continue
        for col_name in df_pair.columns:
            if col_name.endswith(target_columns_suffix): # Only consider Close_Log_Return for cross-asset causality
                # Ensure we don't overwrite if FRED has a column with the same name, though unlikely
                cause_series_pool[col_name] = df_pair[[col_name]].dropna()

    if test_type == "Within-DataDict":
        log_stream.write(f"\n--- Running Within-DataDict Granger tests (Effect & Cause: {target_columns_suffix}) ---\n")
    else: # If FRED data was present
        log_stream.write(f"\n--- Running Exogenous -> Target Granger tests (Effect: {target_columns_suffix}, Causes: FRED + other {target_columns_suffix}) ---\n")

    # Iterate through each target series (effect)
    for effect_full_name, df_effect_series in effect_series_pool.items():
        if df_effect_series.empty:
            continue

        log_stream.write(f"\n[INFO] Testing relationships for effect: {effect_full_name}\n")

        # Iterate through each potential cause series
        for cause_full_name, df_cause_series in cause_series_pool.items():
            # Skip testing a series against itself
            if cause_full_name == effect_full_name:
                continue

            if df_cause_series.empty:
                continue

            # Combine the effect and cause dataframes, aligning by index
            df_combined_for_granger = pd.merge(
                df_effect_series, # This is already a single column DF
                df_cause_series,  # This is already a single column DF
                left_index=True,
                right_index=True,
                how='inner'
            ).dropna()

            if df_combined_for_granger.empty:
                continue

            # Ensure data has enough observations for the test
            if len(df_combined_for_granger) < maxlag_test + 2:
                continue

            try:
                # Pass the full column names directly to find_best_granger_lag
                best_lag, best_p = find_best_granger_lag(df_combined_for_granger, cause_full_name, effect_full_name, maxlag=maxlag_test)

                # Store significant results
                if best_lag is not None and best_p < alpha:
                    all_granger_results.append({
                        "Relation": test_type,
                        "Cause": cause_full_name,
                        "Effect": effect_full_name,
                        "Best_Lag": best_lag,
                        "PValue": best_p,
                        "Observations": len(df_combined_for_granger)
                    })
            except Exception as e:
                log_stream.write(f"[ERROR] Granger test failed for {cause_full_name} -> {effect_full_name}: {e}\n")

    # Convert results list to DataFrame
    granger_df = pd.DataFrame(all_granger_results)

    if not granger_df.empty:
        log_stream.write("\n[INFO] Ringkasan Hasil Uji Granger:\n")
        # Format PValue for better readability
        granger_df['PValue'] = granger_df['PValue'].apply(lambda x: f"{x:.4f}")
        log_stream.write(granger_df.to_string() + "\n")
    else:
        log_stream.write(f"\n[INFO] Tidak ada hubungan Granger signifikan ditemukan pada alpha = {alpha}.\n")

    return granger_df


def identify_significant_exog(log_stream, granger_results_df, alpha):
    """
    Identifies significant exogenous variables (causes) for each target variable (effect)
    from Granger test results.

    Args:
        granger_results_df (pd.DataFrame): DataFrame containing Granger test results
                                          (e.g., output from run_granger_tests).
                                          Expected columns: 'Cause', 'Effect', 'PValue'.
        alpha (float): Significance level.

    Returns:
        dict: Dictionary where keys are target series names (effects) and values are
              lists of significant exogenous series names (causes).
    """
    # log_stream.write("\n\ud83d\udd0d Mengidentifikasi variabel eksogen signifikan berdasarkan hasil uji Granger...\n") # Removed log_stream here for consistency

    significant_exog_map = {}

    if granger_results_df is None or granger_results_df.empty:
        return significant_exog_map

    # Pastikan alpha adalah float
    try:
        alpha_val = float(alpha)
    except:
        alpha_val = 0.05

    # Pastikan PValue benar-benar numerik dan buang yang gagal dikonversi (NaN)
    temp_df = granger_results_df.copy()
    temp_df['PValue'] = pd.to_numeric(temp_df['PValue'], errors='coerce')

    # Filter hanya yang signifikan secara statistik
    significant_results = temp_df[temp_df['PValue'] < alpha_val].dropna(subset=['PValue'])

    for _, row in significant_results.iterrows():
        effect_name = row['Effect']
        cause_name = row['Cause']

        if effect_name not in significant_exog_map:
            significant_exog_map[effect_name] = []

        if cause_name not in significant_exog_map[effect_name]:
            significant_exog_map[effect_name].append(cause_name)

    return significant_exog_map
