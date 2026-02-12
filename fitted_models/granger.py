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

def run_granger_tests(log_stream, data_dict, maxlag_test, alpha, target_pairs=None, exogenous_data_dict=None):
    """
    Runs Granger causality tests between series within data_dict or between data_dict
    and exogenous_data_dict.

    Args:
        log_stream (StringIO): Stream to write log messages.
        data_dict (dict): Dictionary of pandas DataFrames, where keys are series names
                          and values are DataFrames containing the series data.
                          Expected to have a primary column representing the series value.
        maxlag_test (int): Maximum lag for Granger test.
        alpha (float): Significance level.
        target_pairs (list, optional): List of series names from data_dict to test as effects.
                                       If None, all series in data_dict are tested as effects.
        exogenous_data_dict (dict, optional): Dictionary of pandas DataFrames with exogenous data.
                                              Keys are series names, values are DataFrames.
                                              If provided, Granger tests are run from exogenous_data_dict
                                              to data_dict (Exogenous -> Target).

    Returns:
        pd.DataFrame: DataFrame containing significant Granger test results.
    """
    log_stream.write("\n[INFO] Menjalankan Granger causality tests...\n")

    all_granger_results = []

    if not data_dict:
        log_stream.write("  [WARN] Input data_dict is empty. Skipping Granger tests.\n")
        return pd.DataFrame()

    # Determine target series (effects)
    if target_pairs is None:
        target_pairs = list(data_dict.keys())

    # Determine exogenous series (causes)
    if exogenous_data_dict is None:
        # If no exogenous data is provided, test within data_dict (Pair <-> Pair)
        exogenous_data_dict = data_dict
        test_type = "Within-DataDict"
        log_stream.write("\n--- Running Within-DataDict Granger tests ---\n")
    else:
        test_type = "Exogenous -> Target"
        log_stream.write("\n--- Running Exogenous -> Target Granger tests ---\n")


    # Iterate through each target series (effect)
    for effect_name in target_pairs:
        df_effect = data_dict.get(effect_name)
        if df_effect is None or df_effect.empty:
            log_stream.write(f"  [WARN] Data for effect series '{effect_name}' not found or is empty. Skipping.\n")
            continue

        # Assuming the value column is the first column after the index
        if len(df_effect.columns) == 0:
             log_stream.write(f"  [WARN] No value column found for effect series '{effect_name}'. Skipping.\n")
             continue # Skip if no value column

        effect_col = df_effect.columns[0] # Assume the first column is the value


        log_stream.write(f"\n[INFO] Testing relationships for effect: {effect_name} ({effect_col})\n")

        # Iterate through each potential cause series
        for cause_name, df_cause in exogenous_data_dict.items():
            # Skip testing a series against itself if testing within data_dict
            if test_type == "Within-DataDict" and cause_name == effect_name:
                continue

            if df_cause is None or df_cause.empty:
                # log_stream.write(f"  [INFO] Data for cause series '{cause_name}' is empty. Skipping.\n")
                continue # Skip if cause data is empty


            # Assuming the value column is the first column after the index
            if len(df_cause.columns) == 0:
                 # log_stream.write(f"  [WARN] No value column found for cause series '{cause_name}'. Skipping.\n")
                 continue # Skip if cause data has no value column

            cause_col = df_cause.columns[0] # Assume the first column is the value


            # Combine the effect and cause dataframes, aligning by index
            # Use inner join to ensure we only have periods where both series have data
            df_combined_for_granger = pd.merge(
                df_effect[[effect_col]],
                df_cause[[cause_col]],
                left_index=True,
                right_index=True,
                how='inner'
            ).dropna() # Drop any remaining NaNs


            if df_combined_for_granger.empty:
                # log_stream.write(f"  [INFO] Combined data for {cause_name} -> {effect_name} is empty after merge/dropna. Skipping.\n")
                continue # Skip if combined data is empty

            # Ensure data has enough observations for the test
            if len(df_combined_for_granger) < maxlag_test + 2:
                 # log_stream.write(f"  [INFO] Not enough observations ({len(df_combined_for_granger)}) for Granger test {cause_name} -> {effect_name} (need > {maxlag_test + 1}). Skipping.\n")
                 continue # Skip if not enough data


            try:
                # Use the find_best_granger_lag function
                # Pass the column names for cause and effect
                best_lag, best_p = find_best_granger_lag(df_combined_for_granger, cause_col, effect_col, maxlag=maxlag_test)

                # Store significant results
                if best_lag is not None and best_p < alpha:
                    all_granger_results.append({
                        "Relation": f"{test_type}",
                        "Cause": cause_name, # Store the original series name
                        "Effect": effect_name, # Store the original series name
                        "Cause_Column": cause_col,
                        "Effect_Column": effect_col,
                        "Best_Lag": best_lag,
                        "PValue": best_p,
                        "Observations": len(df_combined_for_granger)
                    })
                    # log_stream.write(f"[OK] {cause_name} ({cause_col}) -> {effect_name} ({effect_col}): lag={best_lag}, p={best_p:.4f}\n")
                # else:
                #      log_stream.write(f"  [INFO] {cause_name} -> {effect_name}: p-value > alpha ({alpha}) or no significant lag found. p={best_p:.4f}\n")


            except Exception as e:
                log_stream.write(f"[ERROR] Granger test failed for {cause_name} ({cause_col}) -> {effect_name} ({effect_col}): {e}\n")


    # Convert results list to DataFrame
    granger_df = pd.DataFrame(all_granger_results)

    if not granger_df.empty:
        log_stream.write("\n[INFO] Ringkasan Hasil Uji Granger:\n")
        # Format PValue for better readability
        granger_df['PValue'] = granger_df['PValue'].apply(lambda x: f"{x:.4f}")
        # display(granger_df) # Removed display from module
        log_stream.write(granger_df.to_string() + "\n")
    else:
        log_stream.write(f"\n[INFO] Tidak ada hubungan Granger signifikan ditemukan pada alpha = {alpha}.\n")

    return granger_df


def identify_significant_exog(granger_results_df, alpha):
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
