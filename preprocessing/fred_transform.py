#
# cell_id: 2fCFQ1-lHJZK
# ============================================================
# ↔️ Terapkan Transformasi pada Data FRED (Differencing / Log Transform)
# ============================================================
import pandas as pd
import numpy as np
from parameter import FRED_TRANSFORM_POLICY
# Assuming fred_data and FRED_SERIES are available in the global scope or imported

def macro_varx_transform(series, policy):
    if policy == "log_return":
        return np.log(series).diff()

    elif policy == "rolling_return_20":
        return np.log(series).diff().rolling(20).mean()

    elif policy == "credit_spread_proxy":
        return series.diff().rolling(20).mean()

    elif policy == "log_level_zscore_60":
        x = np.log(series.clip(lower=1e-9))
        return (x - x.rolling(60).mean()) / x.rolling(60).std()

    elif policy == "level_and_diff":
        df = pd.DataFrame()
        df["level"] = (series - series.rolling(252).mean()) / series.rolling(252).std()
        df["diff"] = series.diff()
        return df

    elif policy == "level":
        return series

    else:
        return series.diff()

def get_weights_ffd(d, threshold, size):
    """Menghitung bobot untuk Fixed-Width Window Fractional Diff."""
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold: break
        w.append(w_k)
    return np.array(w[::-1]).reshape(-1, 1)

def apply_frac_diff_ffd(series, d, threshold=1e-4):
    if d == 0: return series
    # Pastikan size tidak lebih besar dari panjang data
    current_size = min(len(series) // 2, 50)
    weights = get_weights_ffd(d, threshold, size=current_size)
    width = len(weights)
    series_values = series.values
    output = []
    for i in range(width, len(series)):
        val = np.dot(weights.T, series_values[i-width:i])[0]
        output.append(val)
    return pd.Series(output, index=series.index[width:])

def apply_fred_transformations(log_stream, fred_data, FRED_SERIES, fred_metadata=None):
    """
    Applies differencing or log transform to FRED data series.

    Args:
        log_stream (StringIO): Stream to write log messages.
        fred_data (pd.DataFrame): DataFrame with combined FRED data, indexed by release date.
                                   Columns are the desired names from FRED_SERIES.
        FRED_SERIES (dict): Dictionary mapping desired names to FRED series IDs.
        fred_metadata (list, optional): List of metadata dictionaries for FRED series.

    Returns:
        dict: Dictionary where keys are original FRED series names and values
              are DataFrames with the transformed series (column name ending in '_Transformed').
              Includes 'effective_until_next_release' column if available.
    """
    log_stream.write("\n[INFO] Menerapkan transformasi pada data FRED agar stasioner...\n")
    transformed_fred_data = {}

    # Helper function to apply transformation
    def apply_transformation(series, series_name, freq):
        """Applies differencing or log transform based on series name and frequency."""
        is_index_like = False
        if fred_metadata:
             for meta in fred_metadata:
                 if meta.get("series_id") == FRED_SERIES.get(series_name):
                      title = meta.get("title", "")
                      if "Index" in series_name or "WEI" in series_name or "Index" in title or "WEI" in title:
                          is_index_like = True
                          break
        # Fallback if metadata not available or doesn't match
        if not is_index_like and ("Index" in series_name or "WEI" in series_name):
             is_index_like = True

        if len(series) < 20:
            log_stream.write(f"  [WARN] Data {series_name} terlalu pendek ({len(series)}). Menggunakan diff(1).\n")
            return series.diff().dropna()

        # This part of the function was not well-formed in the original code,
        # It should use FRED_TRANSFORM_POLICY for `series_id` if provided. 
        # However, the previous code in apply_fred_transformations was already 
        # using policy directly from FRED_TRANSFORM_POLICY. So, simplifying this helper.
        policy = FRED_TRANSFORM_POLICY.get(series_name, "diff") # Use series_name as key for policy
        if is_index_like or policy == "log_return":
            log_stream.write(f"  [INFO] Menerapkan Log Return Transform pada {series_name}\n")
            return np.log(series).diff().dropna() # Log Return agar stasioner
        elif policy == "level_and_diff":
            log_stream.write(f"  [INFO] Menerapkan Level and Diff Transform pada {series_name}\n")
            return macro_varx_transform(series, policy)
        elif policy == "log_diff": # Explicitly handle log_diff
            log_stream.write(f"  [INFO] Menerapkan Log Diff Transform pada {series_name}\n")
            return np.log(series).diff().dropna()
        else: # Default to simple differencing if no specific policy or not index-like
            log_stream.write(f"  [INFO] Menerapkan Differencing (default) pada {series_name}\n")
            return series.diff().dropna()


    # Check if the input is a DataFrame as expected from download_macro_data
    if isinstance(fred_data, pd.DataFrame) and not fred_data.empty:
        log_stream.write("  [INFO] Processing FRED data (DataFrame format)...\n")

        # Iterate through the columns of the combined FRED DataFrame (excluding date/effective columns)
        value_cols = [col for col in fred_data.columns if col not in ["release_date", "effective_until_next_release", "date"]]

        if not value_cols:
             log_stream.write("  [WARN] Tidak ada kolom nilai yang terdeteksi dalam DataFrame FRED. Melewati transformasi.\n")
             return transformed_fred_data # Return empty dict


        for name in value_cols:
            series_id = FRED_SERIES.get(name) # Get FRED Series ID from dictionary

            if name not in fred_data.columns:
                 log_stream.write(f"  [WARN] Kolom '{name}' tidak ditemukan dalam DataFrame. Melewati transformasi.\n")
                 continue

            if fred_data[name].dropna().empty:
                 log_stream.write(f"  [WARN] Seri '{name}' kosong atau hanya NaN setelah dropna. Melewati transformasi.\n")
                 transformed_fred_data[name] = pd.DataFrame() # Store empty DataFrame
                 continue

            try:
                # Get metadata to infer frequency if available
                meta = None
                if series_id and fred_metadata:
                     for m in fred_metadata:
                         if m.get("series_id") == series_id:
                             meta = m
                             break
                freq = meta.get("frequency", "Unknown") if meta else "Unknown"

                # Apply transformation to the series
                # Pass FRED_SERIES and name to apply_transformation so it can check policy if needed
                transformed_series = apply_transformation(fred_data[name].dropna(), name, freq)


                if transformed_series is not None and not transformed_series.empty:
                    if isinstance(transformed_series, pd.DataFrame):
                        # Jika sudah DataFrame (seperti level_and_diff), gunakan langsung
                        transformed_df = transformed_series.copy()
                        # Beri suffix agar konsisten dengan pipeline Anda
                        transformed_df.columns = [f"{name}_{col}_Transformed" for col in transformed_df.columns]
                    else:
                        # Jika Series, baru gunakan .to_frame()
                        transformed_df = transformed_series.to_frame(name=f"{name}_Transformed")

                    # Keep the effective_until_next_release column if it exists and align indices
                    if "effective_until_next_release" in fred_data.columns:
                        # Need to align the effective dates with the transformed series index
                        # Use reindex with forward fill to propagate the last known effective date
                        # Ensure both indices are timezone-aware before reindexing
                        fred_data_index_utc = pd.to_datetime(fred_data.index, errors='coerce', utc=True)
                        transformed_df_index_utc = pd.to_datetime(transformed_df.index, errors='coerce', utc=True)

                        if not fred_data_index_utc.isna().all() and not transformed_df_index_utc.isna().all():
                            effective_dates_aligned = fred_data["effective_until_next_release"].reindex(transformed_df_index_utc, method='ffill')
                            # Reindex back to original transformed index, ensuring timezone consistency
                            transformed_df["effective_until_next_release"] = effective_dates_aligned.reindex(transformed_df.index)

                    transformed_fred_data[name] = transformed_df
                    log_stream.write(f"  [OK] {name}: Transformasi berhasil ({len(transformed_df)} observasi)\n")
                else:
                   log_stream.write(f"  [WARN] Transformasi menghasilkan data kosong untuk {name}.\n")
                   transformed_fred_data[name] = pd.DataFrame() # Store empty DataFrame

            except Exception as e:
                log_stream.write(f"  [ERROR] Gagal menerapkan transformasi untuk {name}: {e}\n")
                transformed_fred_data[name] = pd.DataFrame() # Store empty DataFrame on failure

    else:
        log_stream.write("[WARN] Variabel 'fred_data' bukan DataFrame atau kosong.\n")


    log_stream.write("\n[OK] Proses transformasi data FRED selesai.\n")
    return transformed_fred_data
