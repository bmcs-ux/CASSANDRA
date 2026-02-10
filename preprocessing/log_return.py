#
# cell_id: apply_log_return_to_price
# ============================================================
# 📈 Terapkan Log-Return pada Data Harga (untuk OHLC)
# ============================================================
import pandas as pd
import numpy as np
# Assuming base_dfs is available in the global scope or imported

def apply_log_return_to_price(log_stream, base_dfs, price_cols=['Open', 'High', 'Low', 'Close']):
    """
    Applies log-return transformation to specified price columns of each pair in base_dfs.

    Args:
        log_stream (StringIO): Stream to write log messages.
        base_dfs (dict): Dictionary of pandas DataFrames, where keys are pair names
                         and values are DataFrames with price data (including specified price_cols).
        price_cols (list): List of column names to apply log-return transformation to.

    Returns:
        dict: Dictionary of pandas DataFrames, where keys are pair names
              and values are DataFrames containing the log-return columns
              (e.g., 'Open_Log_Return', 'High_Log_Return', etc.).
    """
    log_stream.write("\n[INFO] Menerapkan transformasi log-return pada data harga (OHLC)...\n") # Changed print to log_stream.write
    log_return_dfs = {}
    for pair_name, df_base in base_dfs.items():
        if df_base is not None and not df_base.empty:
            try:
                # Ensure index is sorted for correct differencing
                df_base_sorted = df_base.sort_index()
                pair_log_returns = pd.DataFrame(index=df_base_sorted.index) # DataFrame to hold log returns for this pair

                for col in price_cols:
                    if col in df_base_sorted.columns:
                        # Hitung log return untuk kolom ini
                        log_return_col_name = f"{col}_Log_Return"
                        pair_log_returns[log_return_col_name] = np.log(df_base_sorted[col] / df_base_sorted[col].shift(1))
                    else:
                        log_stream.write(f"[WARN] Kolom '{col}' tidak ditemukan untuk {pair_name}. Melewati log-return untuk kolom ini.\n") # Changed print to log_stream.write

                # Drop the first row which will have NaNs after shift(1) for all columns
                pair_log_returns = pair_log_returns.dropna(how='all') # Drop row if all new columns are NaN

                if not pair_log_returns.empty:
                     log_return_dfs[pair_name] = pair_log_returns.copy()
                     log_stream.write(f"  [INFO] {pair_name}: Log-return dihitung untuk {list(pair_log_returns.columns)} ({len(pair_log_returns)} observasi)\n") # Changed print to log_stream.write
                else:
                    log_stream.write(f"[WARN] Log-return DataFrame kosong setelah dropna untuk {pair_name}. Melewati log-return.\n") # Changed print to log_stream.write
                    log_return_dfs[pair_name] = pd.DataFrame() # Store empty DataFrame if no log returns calculated


            except Exception as e:
                log_stream.write(f"[ERROR] Gagal hitung log-return untuk {pair_name}: {e}\n") # Changed print to log_stream.write
                log_return_dfs[pair_name] = pd.DataFrame() # Store empty DataFrame on failure
        else:
            log_stream.write(f"[WARN] Data harga tidak valid atau kosong untuk {pair_name}. Melewati log-return.\n") # Changed print to log_stream.write
            log_return_dfs[pair_name] = pd.DataFrame() # Store empty DataFrame

    log_stream.write("\n[OK] Proses log-return selesai.\n") # Changed print to log_stream.write
    return log_return_dfs
