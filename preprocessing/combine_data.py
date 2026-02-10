# cell_id: combine_log_returns
# ============================================================
# ↔ Gabungkan semua log return harga ke dalam satu DataFrame ATAU Dictionary of DataFrames (Module)
# ============================================================
import pandas as pd

def combine_log_returns(log_stream, log_return_dfs, return_type='dict'):
    """
    Combines log return data (Open, High, Low, Close) from multiple pairs into
    a single DataFrame or a dictionary of DataFrames.

    Args:
        log_stream (StringIO): Stream to write log messages.
        log_return_dfs (dict): Dictionary of pandas DataFrames, where keys are pair names
                               and values are DataFrames with log return data
                               (e.g. 'Open_Log_Return', 'High_Log_Return', 'Low_Log_Return', 'Close_Log_Return').
        return_type (str): 'df' to return a single combined DataFrame,
                           'dict' to return a dictionary of DataFrames (one per pair).

    Returns:
        pd.DataFrame or dict: Combined DataFrame or dictionary of DataFrames based on return_type.
    """
    log_stream.write(f"  [INFO] Menggabungkan log return harga ke dalam format '{return_type}'...\n")

    if not log_return_dfs:
        log_stream.write(f"  [WARN] Input log_return_dfs kosong. Tidak ada yang digabungkan.\n")
        return pd.DataFrame() if return_type == 'df' else {}

    # Semua kolom log return yang diharapkan
    possible_cols = ['Open_Log_Return', 'High_Log_Return', 'Low_Log_Return', 'Close_Log_Return']

    # === Mode: Return sebagai DataFrame gabungan ===
    if return_type == 'df':
        combined_df = pd.DataFrame()
        first_pair = True

        for pair_name, df_log_return in log_return_dfs.items():
            if df_log_return is not None and not df_log_return.empty:
                # Filter for columns that end with '_Log_Return'
                available_cols = [c for c in df_log_return.columns if c.endswith('_Log_Return')]


                if not available_cols:
                    log_stream.write(f"  [WARN] Tidak ditemukan kolom log return valid (ending in '_Log_Return') untuk {pair_name}. Dilewati.\n")
                    continue

                # Buat nama kolom gabungan seperti EUR/USD_Close_Log_Return
                df_renamed = df_log_return[available_cols].rename(
                    columns={col: f"{pair_name}_{col}" for col in available_cols}
                )

                if first_pair:
                    combined_df = df_renamed.copy()
                    first_pair = False
                else:
                    combined_df = combined_df.merge(
                        df_renamed,
                        left_index=True,
                        right_index=True,
                        how='outer'
                    )
            else:
                log_stream.write(f"  [WARN] Log return data tidak valid atau kosong untuk {pair_name}. Dilewati.\n")

        # Bersihkan hasil
        combined_df = combined_df.ffill().dropna(how='all')
        log_stream.write(f"  [OK] Penggabungan selesai (DataFrame). Shape: {combined_df.shape}\n")
        return combined_df

    # === Mode: Return sebagai Dictionary per pair ===
    elif return_type == 'dict':
        combined_dict = {}

        for pair_name, df_log_return in log_return_dfs.items():
            if df_log_return is not None and not df_log_return.empty:
                # Keep all available log return columns
                available_cols = [c for c in df_log_return.columns if c.endswith('_Log_Return')]

                if not available_cols:
                    log_stream.write(f"  [WARN] Tidak ditemukan kolom log return valid untuk {pair_name}. Dilewati.\n")
                    continue

                # Copy only the available log return columns
                df_pair = df_log_return[available_cols].copy()

                # Create a 'Log_Return' alias for 'Close_Log_Return' if it exists, for convenience
                if 'Close_Log_Return' in df_pair.columns and 'Log_Return' not in df_pair.columns:
                    df_pair['Log_Return'] = df_pair['Close_Log_Return'] # Add the alias

                combined_dict[pair_name] = df_pair

            else:
                log_stream.write(f"  [WARN] Log return data tidak valid atau kosong untuk {pair_name}. Dilewati.\n")

        log_stream.write(f"  [OK] Penggabungan selesai (Dictionary). Dictionaries created for: {list(combined_dict.keys())}\n")
        return combined_dict

    # === Mode tidak valid ===
    else:
        log_stream.write(f"  [ERROR] return_type '{return_type}' tidak valid. Gunakan 'df' atau 'dict'. Mengembalikan DataFrame kosong.\n")
        return pd.DataFrame()
