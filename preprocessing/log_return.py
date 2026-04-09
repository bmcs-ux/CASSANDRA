import pandas as pd
import numpy as np
import polars as pl

def apply_log_return_to_price(log_stream, mtf_data_dict, price_columns=None):
    """
    Applies log return transformation to specified price columns for each DataFrame
    in a multi-timeframe dictionary.

    Args:
        log_stream (StringIO): Stream to write log messages.
        mtf_data_dict (dict): Dictionary where keys are pair names and values are
                              pd.DataFrames with OHLC data.
        price_columns (list, optional): List of price columns to transform.
                                        Defaults to ['Open', 'High', 'Low', 'Close'].

    Returns:
        dict: Dictionary with pair names as keys and DataFrames containing
              log return columns added for each pair.
    """
    if price_columns is None:
        price_columns = ['Open', 'High', 'Low', 'Close']

    log_returns_dict = {}
    for pair_name, df_original in mtf_data_dict.items():
        if not isinstance(df_original, pd.DataFrame) or df_original.empty:
            log_stream.write(f"[WARN] Data for pair '{pair_name}' is not a DataFrame or is empty. Skipping log return calculation.\n")
            log_returns_dict[pair_name] = pd.DataFrame() # Ensure key exists but with empty DF
            continue

        lf = pl.from_pandas(df_original.reset_index()).lazy()
        applied_to_any_column = False
        expressions = []
        for col in price_columns:
            if col in df_original.columns:
                expressions.append(
                    (pl.col(col).log() - pl.col(col).shift(1).log()).alias(f'{pair_name}_{col}_Log_Return')
                )
                applied_to_any_column = True
            else:
                log_stream.write(f"[WARN] Kolom '{col}' tidak ditemukan di DataFrame untuk {pair_name}. Dilewati.\n")

        if applied_to_any_column:
            index_col = df_original.index.name or "index"
            df_polars = lf.select([pl.col(index_col)] + expressions).drop_nulls().collect(streaming=True)
            df_pd = pd.DataFrame(df_polars.to_dict(as_series=False)).set_index(index_col)
            log_return_cols = [col for col in df_pd.columns if col.endswith('_Log_Return')]
            log_returns_dict[pair_name] = df_pd[log_return_cols]
            if log_returns_dict[pair_name].empty:
                log_stream.write(f"[WARN] No valid log returns generated for pair '{pair_name}' after dropping NaNs.\n")
        else:
            log_stream.write(f"[WARN] No log returns calculated for pair '{pair_name}'. Returning empty DataFrame for it.\n")
            log_returns_dict[pair_name] = pd.DataFrame()

    return log_returns_dict
