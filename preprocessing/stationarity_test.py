#
# cell_id: stationarity_test
# ============================================================
# ⌒️ Uji Stasioneritas dengan ADF dan PP Test & Otomatis Differencing jika Perlu
# ============================================================

from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
import pandas as pd
import numpy as np
# Assuming log_return_dfs, cleaned_transformed_fred_data, alpha are available or imported

def test_and_stationarize_data(log_stream, log_return_dfs, cleaned_transformed_fred_data, alpha=0.05, max_attempts=2):
    """
    Performs stationarity tests (ADF, PP) and applies differencing if needed.

    Args:
        log_stream (StringIO): Stream to write log messages.
        log_return_dfs (dict): Dictionary of DataFrames with log-return data for pairs.
        cleaned_transformed_fred_data (dict): Dictionary of DataFrames with transformed FRED data (missing handled).
        alpha (float): Significance level for stationarity tests.
        max_attempts (int): Maximum number of differencing attempts for FRED data.

    Returns:
        tuple: A tuple containing:
            - list: List of dictionaries with stationarity test results.
            - dict: Dictionary of DataFrames with final stationarized FRED data.
    """
    log_stream.write("\n[INFO] Melakukan uji stasioneritas (ADF dan PP test) dan differencing jika perlu...\n") # Changed print

    stationarity_results = [] # List to store test results
    final_stationarized_fred_data = {} # Dictionary to store the final stationarized FRED data

    # Test Price Log Returns (should already be stationary)
    log_stream.write("\n--- Menguji Stasioneritas Log Return Harga ---\n") # Changed print
    if log_return_dfs:
        for pair_name, df_log_return in log_return_dfs.items():
            if df_log_return is not None and not df_log_return.empty:
                # Iterate over all log return columns for this pair
                log_return_cols_to_test = [col for col in df_log_return.columns if col.endswith('_Log_Return')]

                for col in log_return_cols_to_test:
                    series = df_log_return[col].dropna()
                    if len(series) > 10: # Need enough observations for the tests
                        try:
                            # ADF Test
                            adf_result = adfuller(series)
                            adf_pvalue = adf_result[1]
                            is_adf_stationary = adf_pvalue < alpha

                            # PP Test
                            pp_result = PhillipsPerron(series)
                            pp_pvalue = pp_result.pvalue
                            is_pp_stationary = pp_pvalue < alpha

                            stationarity_results.append({
                                "Series": pair_name + f" ({col})", # Include column name
                                "Test": "ADF",
                                "PValue": adf_pvalue,
                                "Is_Stationary": is_adf_stationary
                            })
                            stationarity_results.append({
                                "Series": pair_name + f" ({col})", # Include column name
                                "Test": "PP",
                                "PValue": pp_pvalue,
                                "Is_Stationary": is_pp_stationary
                            })
                            log_stream.write(f"  [INFO] {pair_name} ({col}): ADF p={adf_pvalue:.4f} ({'Stationary' if is_adf_stationary else 'Non-Stationary'}), PP p={pp_pvalue:.4f} ({'Stationary' if is_pp_stationary else 'Non-Stationary'})\n") # Changed print

                        except Exception as e:
                            log_stream.write(f"[ERROR] Gagal melakukan uji stasioneritas untuk {pair_name} ({col}): {e}\n") # Changed print
                    else:
                        log_stream.write(f"[INFO] Data tidak cukup untuk uji stasioneritas untuk {pair_name} ({col}).\n") # Changed print
            else:
                log_stream.write(f"[WARN] Data log return tidak valid atau kosong untuk {pair_name}.\n") # Changed print
    else:
        log_stream.write("\n[INFO] Tidak ada data log return untuk diuji.\n") # Changed print


    # Test and potentially re-transform FRED Data
    log_stream.write("\n--- Menguji Stasioneritas Data FRED yang Ditransformasi & Differencing Otomatis ---\n") # Changed print
    if cleaned_transformed_fred_data:
        for series_name, df_transformed in cleaned_transformed_fred_data.items():
            transformed_col_names = [col for col in df_transformed.columns if col.endswith("_Transformed")]
            if transformed_col_names:
                transformed_col_name = transformed_col_names[0]
                if df_transformed is not None and not df_transformed.empty and transformed_col_name in df_transformed.columns:
                    original_transformed_series = df_transformed[transformed_col_name].dropna()
                    current_series = original_transformed_series.copy()
                    is_stationary = False
                    attempts = 0

                    series_test_results = []

                    while not is_stationary and attempts < max_attempts and len(current_series) > 10:
                        prefix = "" if attempts == 0 else f"Differenced (Order {attempts}) "
                        log_stream.write(f"  [INFO] Menguji {prefix}{series_name} (Transformed FRED)...\n") # Changed print

                        try:
                            # ADF Test
                            adf_result = adfuller(current_series)
                            adf_pvalue = adf_result[1]
                            is_adf_stationary = adf_pvalue < alpha

                            # PP Test
                            pp_result = PhillipsPerron(current_series)
                            pp_pvalue = pp_result.pvalue
                            is_pp_stationary = pp_pvalue < alpha

                            is_stationary = is_adf_stationary and is_pp_stationary # Consider stationary if BOTH tests pass

                            series_test_results.append({
                                "Series": f"{series_name} (Transformed FRED){' ' + prefix.strip() if prefix else ''}",
                                "Test": "ADF",
                                "PValue": adf_pvalue,
                                "Is_Stationary": is_adf_stationary
                            })
                            series_test_results.append({
                                "Series": f"{series_name} (Transformed FRED){' ' + prefix.strip() if prefix else ''}",
                                "Test": "PP",
                                "PValue": pp_pvalue,
                                "Is_Stationary": is_pp_stationary
                            })

                            log_stream.write(f"    [INFO] ADF p={adf_pvalue:.4f} ({'Stationary' if is_adf_stationary else 'Non-Stationary'}), PP p={pp_pvalue:.4f} ({'Stationary' if is_pp_stationary else 'Non-Stationary'})\n") # Changed print

                            if not is_stationary and attempts < max_attempts -1:
                                 log_stream.write(f"    -> Tidak stasioner. Menerapkan differencing order {attempts + 1}...\n") # Changed print
                                 current_series = current_series.diff().dropna()
                                 attempts += 1
                            elif not is_stationary and attempts == max_attempts -1:
                                log_stream.write(f"    -> Masih tidak stasioner setelah {max_attempts} kali differencing.\n") # Changed print
                                attempts += 1 # Increment attempts to exit loop

                        except Exception as e:
                            log_stream.write(f"[ERROR] Gagal melakukan uji stasioneritas atau differencing untuk {series_name}: {e}\n") # Changed print
                            break # Exit while loop on error

                    # Add results of all attempts for this series to the main results list
                    stationarity_results.extend(series_test_results)

                    # Store the final (hopefully stationarized) series
                    if is_stationary:
                         final_stationarized_fred_data[series_name] = current_series.to_frame(name=f"{series_name}_FinalTransformed")
                         # Need to re-align effective_until_next_release index if differenced
                         if attempts > 0 and "effective_until_next_release" in df_transformed.columns:
                             # Pastikan index terurut naik (monotonic increasing)
                             eff_dates = df_transformed["effective_until_next_release"].sort_index()
                             current_idx = current_series.index.sort_values()

                             # Reindex dengan forward-fill yang aman
                             aligned_effective_dates = eff_dates.reindex(current_idx, method="ffill")

                             # Sesuaikan kembali urutan ke index asli jika perlu
                             aligned_effective_dates = aligned_effective_dates.reindex(current_series.index)

                             final_stationarized_fred_data[series_name]["effective_until_next_release"] = aligned_effective_dates
                         elif "effective_until_next_release" in df_transformed.columns:
                             # If not differenced, just copy the effective dates and align
                              final_stationarized_fred_data[series_name]["effective_until_next_release"] = df_transformed["effective_until_next_release"].reindex(current_series.index)


                    else:
                         log_stream.write(f"[WARN] {series_name} tetap tidak stasioner setelah {max_attempts} kali differencing. Tidak menyertakan dalam data final.\n") # Changed print
                         final_stationarized_fred_data[series_name] = pd.DataFrame() # Store empty if not stationary


                else:
                    log_stream.write(f"[WARN] Data FRED yang ditransformasi awal tidak valid atau kosong untuk {series_name}.\n") # Changed print
                    final_stationarized_fred_data[series_name] = pd.DataFrame() # Store empty if initial data is bad
            else:
                 log_stream.write(f"[WARN] Kolom nilai transformasi tidak ditemukan di DataFrame '{series_name}'. Melewati uji stasioneritas.\n") # Changed print
                 final_stationarized_fred_data[series_name] = pd.DataFrame() # Store empty if initial data is bad


    else:
        log_stream.write("\n[INFO] Tidak ada data FRED yang ditransformasi awal untuk diuji.\n") # Changed print


    # Display results in a DataFrame
    stationarity_df = pd.DataFrame(stationarity_results)
    if not stationarity_df.empty:
        log_stream.write("\n[INFO] Ringkasan Hasil Uji Stasioneritas:\n") # Changed print
        # Format PValue for better readability
        stationarity_df['PValue'] = stationarity_df['PValue'].apply(lambda x: f"{x:.4f}")
        #display(stationarity_df) # Removed display from module
        log_stream.write(stationarity_df.to_string() + "\n") # Log DataFrame
    else:
        log_stream.write("\n[INFO] Tidak ada hasil uji stasioneritas yang tersedia.\n") # Changed print

    # Display information about final stationarized data
    log_stream.write("\n[INFO] Data FRED Final Stationer:\n") # Changed print
    if final_stationarized_fred_data:
        for name, df in final_stationarized_fred_data.items():
            if not df.empty:
                log_stream.write(f"  [INFO] {name}: {len(df)} observasi, kolom: {list(df.columns)}\n") # Changed print
                # Optional: display head of the final data
                # display(df.head()) # Removed display
            else:
                log_stream.write(f"  [INFO] {name}: Data kosong (tidak mencapai stasioneritas).\n") # Changed print
    else:
        log_stream.write("[INFO] Tidak ada data FRED yang berhasil distasionerkan.\n") # Changed print


    log_stream.write("\n[OK] Uji stasioneritas dan differencing otomatis selesai.\n") # Changed print
    return stationarity_results, final_stationarized_fred_data
