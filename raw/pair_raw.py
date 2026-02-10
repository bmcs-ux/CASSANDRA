#
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import warnings
from parameter import PAIRS, lookback_days, base_interval # REMOVED USE_LOCAL_CSV_FOR_PAIRS, LOCAL_CSV_FILEPATH
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
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        if data is not None and not data.empty:
             if not isinstance(data.index, pd.DatetimeIndex):
                  data.index = pd.to_datetime(data.index, errors='coerce', utc=True)
             elif data.index.tz is None:
                  data.index = data.index.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')
             else:
                  data.index = data.index.tz_convert('UTC')

             data = data[~data.index.isna()]

             return data
        else:
            log_stream.write(f"[WARN] Tidak ada data yang diunduh untuk {symbol} dengan interval {interval} dalam {lookback_days} hari terakhir.\n")
            return None
    except Exception as e:
        log_stream.write(f"[WARN] Gagal mengunduh data untuk {symbol}: {e}\n")
        return None

# Di dalam fungsi load_base_data_mtf
# Interval yang didukung: ['1d', '1h', '1m']

def load_base_data_mtf(log_stream, PAIRS, lookback_days, base_interval, USE_LOCAL_CSV_FOR_PAIRS, BASE_PATH):
    base_dfs = {}
    local_load_success = False
    
    # Sesuaikan nama file berdasarkan interval (MTF Friendly)
    # Contoh: /path/to/data_1h.csv
    file_path = f"{BASE_PATH}_{base_interval}.csv" 

    if USE_LOCAL_CSV_FOR_PAIRS:
        log_stream.write(f"[INFO] Mencoba memuat data MTF ({base_interval}) dari: {file_path}\n")
        try:
            # Menggunakan chunking atau pemuatan standar
            df_all = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            if not df_all.empty:
                # Pastikan timezone UTC untuk konsistensi antar TF
                if df_all.index.tz is None:
                    df_all.index = df_all.index.tz_localize('UTC')
                else:
                    df_all.index = df_all.index.tz_convert('UTC')

                for pair_name in PAIRS.keys():
                    # Filter kolom yang diawali NamaPair (Case Insensitive)
                    cols = [c for c in df_all.columns if c.lower().startswith(f"{pair_name.lower()}_")]
                    if cols:
                        sub_df = df_all[cols].copy()
                        # Bersihkan prefix untuk mendapatkan nama standar (Open, High, Low, Close)
                        sub_df.columns = [c.split('_')[-1] for c in sub_df.columns]
                        
                        # Filter lookback spesifik untuk timeframe ini
                        sub_df = _apply_lookback_filter(log_stream, sub_df, lookback_days, pair_name)
                        
                        if not sub_df.empty:
                            base_dfs[pair_name] = sub_df
                
                if base_dfs:
                    local_load_success = True
                    log_stream.write(f"[OK] Berhasil memuat {len(base_dfs)} pairs dari CSV lokal {base_interval}.\n")
        except FileNotFoundError:
            log_stream.write(f"[WARN] File {file_path} tidak ditemukan. Memulai download.\n")
        except Exception as e:
            log_stream.write(f"[ERROR] Gagal memuat CSV MTF: {e}\n")

    if not local_load_success:
        log_stream.write(f"[INFO] Fallback: Mengunduh data baru untuk interval {base_interval}\n")
        download_count = 0
        
        for name, sym in PAIRS.items():
            try:
                # download_base_symbol sekarang harus menerima base_interval secara eksplisit
                log_stream.write(f"[DOWNLOAD] Meminta {sym} [{base_interval}] untuk {lookback_days} hari terakhir...\n")
                
                df_dl = download_base_symbol(log_stream, sym, lookback_days, base_interval)
                
                if df_dl is not None and not df_dl.empty:
                    # Normalisasi struktur kolom yfinance yang sering berubah
                    if isinstance(df_dl.columns, pd.MultiIndex):
                        df_dl.columns = [c[0] if isinstance(c, tuple) else c for c in df_dl.columns]
                    
                    # Standarisasi Nama Kolom (MTF standard)
                    rename_map = {c: c.capitalize() for c in df_dl.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
                    df_dl = df_dl.rename(columns=rename_map)
                    
                    # Simpan hanya kolom inti
                    df_dl = df_dl[[c for c in ['Open', 'High', 'Low', 'Close'] if c in df_dl.columns]]
                    
                    # Filter Lookback Akhir
                    df_dl = _apply_lookback_filter(log_stream, df_dl, lookback_days, name)
                    
                    base_dfs[name] = df_dl
                    download_count += 1
                else:
                    log_stream.write(f"[WARN] Data {name} tidak tersedia untuk interval {base_interval}.\n")
                    
            except Exception as e:
                log_stream.write(f"[ERROR] Gagal mengunduh {name} ({base_interval}): {e}\n")

        # Opsi: Simpan hasil download ke CSV lokal agar run berikutnya lebih cepat
        if download_count > 0:
            # Fungsi pembantu untuk menyimpan MTF data
            # save_mtf_to_local(base_dfs, file_path) 
            log_stream.write(f"[OK] Selesai. {download_count} pair siap dalam buffer {base_interval}.\n")
            
    return base_dfs
