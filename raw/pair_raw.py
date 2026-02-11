#%%writefile '/content/drive/MyDrive/books/CASSANDRA/raw/pair_raw.py'
#
import os, sys
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import warnings
from parameter import PAIRS, LOOKBACK_DAYS, MTF_INTERVALS, USE_LOCAL_CSV_FOR_PAIRS, LOCAL_CSV_FILEPATH # Updated parameter imports
warnings.filterwarnings("ignore")

# NEW HELPER FUNCTION: _apply_lookback_filter
def _apply_lookback_filter(log_stream, df, lookback_days, pair_name):
    """
    Filters a DataFrame to include only data from the last `lookback_days`
    relative to the latest timestamp in the DataFrame.
    """
    if df.empty:
        log_stream.write(f"  [WARN] DataFrame for {pair_name} is empty, skipping lookback filter.\n")
        return df

    # Ensure index is datetime and UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')

    latest_date = df.index.max()
    cutoff_date = latest_date - timedelta(days=lookback_days)

    filtered_df = df[df.index >= cutoff_date]

    if len(filtered_df) < len(df):
        log_stream.write(f"  [INFO] Data for {pair_name} filtered to last {lookback_days} days. Original: {len(df)} rows, Filtered: {len(filtered_df)} rows.\n")

    if len(filtered_df) < 2: # Need at least 2 observations for log returns
        log_stream.write(f"  [WARN] Data for {pair_name} has fewer than 2 observations after lookback filter. May cause issues downstream.\n")

    return filtered_df


# cell_id: download_base_symbol
def download_base_symbol(log_stream, symbol, lookback_days, interval):
    """
    Downloads historical data for a given symbol from Yahoo Finance.
    Returns a pandas DataFrame.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    log_stream.write(f"[INFO] Downloading {symbol} with interval {interval} from {start_date.date()} to {end_date.date()}\n")
    try:
        # Removed show_errors=False as it is not a valid argument for yf.download
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        if data is not None and not data.empty:
             if not isinstance(data.index, pd.DatetimeIndex):
                  data.index = pd.to_datetime(data.index, errors='coerce', utc=True)
             elif data.index.tz is None:
                  data.index = data.index.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')
             else:
                  data.index = data.index.tz_convert('UTC')

             data = data[~data.index.isna()]
             log_stream.write(f"[OK] Downloaded {len(data)} rows for {symbol} ({interval}).\n")
             return data
        else:
            log_stream.write(f"[WARN] No data downloaded for {symbol} with interval {interval} for last {lookback_days} days.\n")
            return None
    except Exception as e:
        log_stream.write(f"[ERROR] Failed to download data for {symbol} ({interval}): {e}\n")
        return None

# Di dalam fungsi load_base_data_mtf
# Interval yang didukung: ['1d', '1h', '1m']

def load_base_data_mtf(log_stream, PAIRS, lookback_days_config, interval_str, USE_LOCAL_CSV_FOR_PAIRS, LOCAL_CSV_FILEPATH):
    """
    Loads base data for multiple pairs, either from local CSV or by downloading from Yahoo Finance.
    Handles Multi-Timeframe (MTF) data by using interval_str.

    Args:
        log_stream (StringIO): Stream to write log messages.
        PAIRS (dict): Dictionary of trading pairs and their Yahoo Finance symbols.
        lookback_days_config (int): Number of days to look back for data (from parameters).
        interval_str (str): Yahoo Finance interval string (e.g., '1d', '1h', '1m').
        USE_LOCAL_CSV_FOR_PAIRS (bool): Whether to attempt loading from local CSV.
        LOCAL_CSV_FILEPATH (str): Base path for local CSV files.

    Returns:
        dict: A dictionary where keys are pair names and values are pandas DataFrames.
    """
    base_dfs = {}

    # Ini memastikan pencarian file konsisten dengan format akhiran Anda.
    clean_tf = interval_str.lower().replace('min', 'm').replace('60m', 'h1').replace('1h', 'h1')
    if clean_tf == '1m': clean_tf = 'm1' # Standarisasi ke m1

    # 2. Konstruksi path: mencari file dengan format [NAMA_FILE]_[TF].csv
    # Contoh: combined_data_final_complete_m1.csv
    file_dir = os.path.dirname(LOCAL_CSV_FILEPATH)
    file_base = os.path.basename(LOCAL_CSV_FILEPATH).rsplit('.', 1)[0]
    file_path_for_interval = os.path.join(file_dir, f"{file_base}_{clean_tf}.csv")
    # --------------

    if USE_LOCAL_CSV_FOR_PAIRS and os.path.exists(file_path_for_interval):
        log_stream.write(f"[INFO] Attempting to load MTF ({interval_str}) data from local CSV: {file_path_for_interval}\n")
        try:
            df_all = pd.read_csv(file_path_for_interval, index_col=0, parse_dates=True)

            if not df_all.empty:
                if df_all.index.tz is None:
                    df_all.index = df_all.index.tz_localize('UTC')
                else:
                    df_all.index = df_all.index.tz_convert('UTC')
                log_stream.write(f"[OK] Loaded {len(df_all)} rows from local CSV for {interval_str}.\n")

                # Split the combined DataFrame back into individual pair DataFrames
                for pair_name, symbol in PAIRS.items():
                    # Filter columns that start with the pair_name (case-insensitive)
                    # Example: 'EURUSD_Open', 'EURUSD_High', etc.
                    cols_for_pair = [col for col in df_all.columns if col.lower().startswith(f"{pair_name.lower()}_")]
                    if cols_for_pair:
                        pair_df = df_all[cols_for_pair].copy()
                        # Rename columns by removing the pair_name prefix
                        pair_df.columns = [col.replace(f"{pair_name}_", "") for col in pair_df.columns]

                        # Apply lookback filter to ensure consistency with downloaded data
                        base_dfs[pair_name] = _apply_lookback_filter(log_stream, pair_df, lookback_days_config, pair_name)
                        log_stream.write(f"[INFO] Extracted {len(base_dfs[pair_name])} rows for {pair_name} from local CSV.\n")
                    else:
                        log_stream.write(f"[WARN] No columns found for {pair_name} in local CSV for interval {interval_str}. Skipping.\n")
                local_load_success = True
            else:
                log_stream.write(f"[WARN] Local CSV for {interval_str} is empty: {file_path_for_interval}. Attempting download.\n")
        except Exception as e:
            log_stream.write(f"[ERROR] Failed to load from local CSV for {interval_str}: {e}. Attempting download.\n")
    else:
        log_stream.write(f"[INFO] Local CSV not used or not found for {interval_str}. Attempting download.\n")

    if not base_dfs or not local_load_success: # If local load failed or not attempted, download
        log_stream.write(f"[INFO] Downloading data for interval {interval_str} from Yahoo Finance...\n")
        for pair_name, symbol in PAIRS.items():
            df = download_base_symbol(log_stream, symbol, lookback_days_config, interval_str)
            if df is not None and not df.empty:
                # Filter columns to only include OHLCV
                ohlcv_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]
                if ohlcv_cols:
                    base_dfs[pair_name] = df[ohlcv_cols].copy()
                    log_stream.write(f"[OK] Successfully processed {pair_name} for interval {interval_str}.\n")
                else:
                    log_stream.write(f"[WARN] No OHLCV data found for {pair_name} ({symbol}) for interval {interval_str}. Skipping.\n")
            else:
                log_stream.write(f"[WARN] No data retrieved for {pair_name} ({symbol}) for interval {interval_str} from Yahoo Finance.\n")

    if not base_dfs:
        log_stream.write(f"[ERROR] No base data obtained for interval {interval_str} after all attempts.\n")

    return base_dfs
