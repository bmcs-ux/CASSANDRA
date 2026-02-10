#
# cell_id: handle_missing_fred_data
# ============================================================
# ♻️ Menangani Missing Data pada Data FRED yang Ditransformasi
# ============================================================
import pandas as pd

def handle_missing_fred_data(log_stream, transformed_fred_data, missing_threshold=0.30):
    """
    Handles missing data in transformed FRED data by forward filling or dropping series.

    Args:
        log_stream (StringIO): Stream to write log messages.
        transformed_fred_data (dict): Dictionary of DataFrames with transformed FRED data.
                                      Keys are series names, values are DataFrames.
                                      Expected to have a column ending in '_Transformed'.
        missing_threshold (float): Percentage of missing data allowed before dropping a series.

    Returns:
        dict: Dictionary of DataFrames with missing data handled.
    """
    log_stream.write("\n[INFO] Menangani missing data pada data FRED yang ditransformasi...\n") # Changed print
    cleaned_transformed_fred_data = {}
    series_to_drop = []

    if transformed_fred_data:
        for series_name, df_transformed in transformed_fred_data.items():
            if df_transformed is not None and not df_transformed.empty:
                # Find the transformed value column
                transformed_col_names = [col for col in df_transformed.columns if col.endswith("_Transformed")]
                if transformed_col_names:
                    transformed_col_name = transformed_col_names[0]
                    # Calculate percentage of missing values in the transformed value column
                    missing_percentage = df_transformed[transformed_col_name].isnull().mean()

                    if missing_percentage > missing_threshold:
                        log_stream.write(f"[WARN] Seri '{series_name}' memiliki {missing_percentage:.2%} missing data, melebihi ambang batas {missing_threshold:.0%}. Menandai untuk dihapus.\n") # Changed print
                        series_to_drop.append(series_name)
                    else:
                        # Apply interpolation (forward fill) for series not marked for dropping
                        # We apply ffill on the transformed value column
                        df_cleaned = df_transformed.copy()
                        df_cleaned[transformed_col_name] = df_cleaned[transformed_col_name].ffill()
                        # Also forward fill the effective_until_next_release column if it exists
                        if "effective_until_next_release" in df_cleaned.columns:
                             df_cleaned["effective_until_next_release"].ffill(inplace=True)

                        cleaned_transformed_fred_data[series_name] = df_cleaned
                        log_stream.write(f"  [INFO] Seri '{series_name}': Missing data ({missing_percentage:.2%}) diisi dengan forward fill.\n") # Changed print
                else:
                     log_stream.write(f"[WARN] Kolom nilai transformasi tidak ditemukan di DataFrame '{series_name}'. Melewati penanganan missing data.\n") # Changed print
                     # Keep the original data in case the column name was unexpected
                     cleaned_transformed_fred_data[series_name] = df_transformed.copy()

            else:
                log_stream.write(f"[WARN] Data FRED yang ditransformasi kosong atau tidak valid untuk '{series_name}'. Melewati penanganan missing data.\n") # Changed print
                cleaned_transformed_fred_data[series_name] = pd.DataFrame() # Store empty if original was empty

        # Remove series marked for dropping from the cleaned data dictionary
        for series_name in series_to_drop:
            if series_name in cleaned_transformed_fred_data:
                del cleaned_transformed_fred_data[series_name]
                log_stream.write(f"[INFO] Seri '{series_name}' dihapus karena missing data.\n") # Changed print

    else:
        log_stream.write("[INFO] Tidak ada data FRED yang ditransformasi untuk ditangani missing datanya.\n") # Changed print


    log_stream.write("\n[INFO] Ringkasan Data FRED Setelah Penanganan Missing Data:\n") # Changed print
    if cleaned_transformed_fred_data:
        for name, df in cleaned_transformed_fred_data.items():
            if not df.empty:
                transformed_col_names = [col for col in df.columns if col.endswith("_Transformed")]
                if transformed_col_names:
                    transformed_col_name = transformed_col_names[0]
                    remaining_missing_percentage = df[transformed_col_name].isnull().mean()
                    log_stream.write(f"  [INFO] {name}: {len(df)} observasi, {remaining_missing_percentage:.2%} missing data.\n") # Changed print
                else:
                    log_stream.write(f"  [INFO] {name}: {len(df)} observasi, kolom nilai transformasi tidak ditemukan.\n") # Changed print
            else:
                log_stream.write(f"  [INFO] {name}: Data kosong.\n") # Changed print
    else:
        log_stream.write("[INFO] Tidak ada data FRED yang berhasil ditangani missing datanya.\n") # Changed print


    log_stream.write("\n[OK] Penanganan missing data selesai.\n") # Changed print
    return cleaned_transformed_fred_data
