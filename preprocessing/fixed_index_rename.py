#
# cell_id: oB-IEpJpukto
# ============================================================
# 🧩 FIX: Normalisasi dan validasi index + kolom FRED sebelum Granger test
# ============================================================
import pandas as pd

def normalize_and_validate_fred_data(final_stationarized_fred_data, FRED_SERIES):
    """
    Normalizes and validates the index and column names of final stationarized FRED data.

    Args:
        final_stationarized_fred_data (dict): Dictionary of DataFrames with final stationarized FRED data.
                                              Keys are series names, values are DataFrames.
        FRED_SERIES (dict): Dictionary mapping desired names to FRED series IDs.

    Returns:
        dict: Dictionary of DataFrames with normalized and validated data.
    """
    print("\n🧩 FIX: Normalisasi dan validasi index + kolom FRED...")
    processed_fred_data = {}

    if final_stationarized_fred_data:
        # Create a mapping from original series names (from keys) to FRED_SERIES keys
        # This assumes the keys in final_stationarized_fred_data are the original names
        # from the download step or a consistent naming convention.
        # If the keys are still like 'series_0', this mapping needs adjustment.
        # Let's assume the keys are the desired names from FRED_SERIES.keys()
        fred_series_names = list(FRED_SERIES.keys())

        for series_name, df in final_stationarized_fred_data.items():
            if df is None or df.empty:
                processed_fred_data[series_name] = pd.DataFrame()
                continue

            # --- Pastikan index bertipe datetime ---
            # Check if index is already datetime and has a reasonable min year
            if not isinstance(df.index, pd.DatetimeIndex) or (not df.index.empty and df.index.min().year < 1980):
                # Try to find a suitable date column if index is not datetime
                time_col = None
                for col in ["effective_until_next_release", "release_date", "date"]:
                    if col in df.columns:
                        time_col = col
                        break
                if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
                     df.index = df[time_col]
                     df = df.drop(columns=[time_col]) # Drop the column after setting as index
                else:
                    # Attempt to convert existing index
                    df.index = pd.to_datetime(df.index, errors="coerce", utc=True)

            # Drop rows with invalid (NaT) index
            df = df[~df.index.isna()]

            # --- Pastikan timezone-aware (UTC) ---
            if df.index.tz is None:
                try:
                    df = df.tz_localize("UTC", nonexistent='shift_forward', ambiguous='NaT')
                except Exception as e:
                    print(f"⚠️ Gagal localize timezone for {series_name}: {e}. Skipping timezone localization.")
                    pass # Continue without timezone if localization fails


            # --- Drop duplikat index dan sort ---
            # Keep the last observation for duplicate indices
            df = df[~df.index.duplicated(keep="last")].sort_index()

            # --- Rename kolom nilai ke nama asli FRED_SERIES ---
            # Find the column ending with "_FinalTransformed"
            value_cols = [col for col in df.columns if col.endswith("_FinalTransformed")]
            if value_cols:
                 old_value_col = value_cols[0]
                 # Rename it to the original series name (assuming key is original name)
                 if series_name in fred_series_names: # Check if the key is one of the desired names
                     df.rename(columns={old_value_col: series_name}, inplace=True)
                 else:
                     print(f"⚠️ Seri name '{series_name}' not found in FRED_SERIES keys. Value column '{old_value_col}' not renamed.")

            processed_fred_data[series_name] = df
            print(f"  • {series_name}: Index dan kolom divalidasi. Shape: {df.shape}")

    else:
        print("ℹ️ Tidak ada data FRED stationer final untuk dinormalisasi.")

    print("\n✅ Normalisasi dan validasi FRED data selesai.")
    return processed_fred_data

# Example usage (optional, for testing the function within the cell)
# Assuming final_stationarized_fred_data and FRED_SERIES are defined in global scope or imported
# if 'final_stationarized_fred_data' in globals() and 'FRED_SERIES' in globals():
#     processed_fred_data = normalize_and_validate_fred_data(final_stationarized_fred_data, FRED_SERIES)
#     print("\nReturned processed_fred_data keys:")
#     print(processed_fred_data.keys())
#     if processed_fred_data:
#         first_series = list(processed_fred_data.keys())[0]
#         if not processed_fred_data[first_series].empty:
#             print(f"\nHead of {first_series} Processed Data:")
#             display(processed_fred_data[first_series].head())
#         else:
#             print("\nTidak ada data FRED yang diproses untuk ditampilkan.")
# else:
#     print("\nRequired variables (final_stationarized_fred_data, FRED_SERIES) not found. Cannot run example.")
