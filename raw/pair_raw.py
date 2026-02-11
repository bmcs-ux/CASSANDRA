from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")


INTERVAL_SUFFIX_MAP = {
    '1m': ['m1', '1m'],
    '5m': ['m5', '5m'],
    '15m': ['m15', '15m'],
    '30m': ['m30', '30m'],
    '1h': ['h1', '1h', '60m'],
    '1d': ['d1', '1d'],
}


def _apply_lookback_filter(log_stream, df, lookback_days, pair_name):
    """
    Filters a DataFrame to include only data from the last `lookback_days`
    relative to the latest timestamp in the DataFrame.
    """
    if df.empty:
        log_stream.write(f"  [WARN] DataFrame for {pair_name} is empty, skipping lookback filter.\n")
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')

    latest_date = df.index.max()
    cutoff_date = latest_date - timedelta(days=lookback_days)

    filtered_df = df[df.index >= cutoff_date]

    if len(filtered_df) < len(df):
        log_stream.write(
            f"  [INFO] Data for {pair_name} filtered to last {lookback_days} days. "
            f"Original: {len(df)} rows, Filtered: {len(filtered_df)} rows.\n"
        )

    if len(filtered_df) < 2:
        log_stream.write(
            f"  [WARN] Data for {pair_name} has fewer than 2 observations after lookback filter. "
            "May cause issues downstream.\n"
        )

    return filtered_df


def _resolve_local_csv_path(base_path, base_interval):
    """
    Resolve local CSV path for a given timeframe.

    Supports naming patterns in data_base with timeframe suffixes, e.g.:
    - combined_data_final_complete_m1.csv
    - combined_data_final_complete_1h.csv
    """
    base_path_obj = Path(base_path)
    interval_key = str(base_interval).lower()
    suffixes = INTERVAL_SUFFIX_MAP.get(interval_key, [interval_key])

    search_roots = []
    if base_path_obj.is_dir():
        search_roots.append(base_path_obj)
    else:
        if base_path_obj.parent.exists():
            search_roots.append(base_path_obj.parent)
        fallback_data_base = Path.cwd() / 'data_base'
        if fallback_data_base.exists() and fallback_data_base not in search_roots:
            search_roots.append(fallback_data_base)

    stems = []
    if not base_path_obj.is_dir() and base_path_obj.name:
        stems.append(base_path_obj.stem)
    stems.extend(['combined_data_final_complete', 'combined_data'])

    seen = set()
    candidate_paths = []
    for root in search_roots:
        for stem in stems:
            for suffix in suffixes:
                for pattern in [
                    f"{stem}_{suffix}.csv",
                    f"{stem}.csv_{suffix}.csv",
                    f"*_{suffix}.csv",
                ]:
                    for found in sorted(root.glob(pattern)):
                        norm = str(found.resolve())
                        if norm not in seen:
                            seen.add(norm)
                            candidate_paths.append(found)

    return candidate_paths[0] if candidate_paths else None


def _read_local_base_data(log_stream, file_path, pairs, lookback_days):
    base_dfs = {}
    df_all = pd.read_csv(file_path, index_col=0, parse_dates=True)

    if df_all.empty:
        log_stream.write(f"[WARN] CSV lokal {file_path} kosong.\n")
        return base_dfs

    if not isinstance(df_all.index, pd.DatetimeIndex):
        df_all.index = pd.to_datetime(df_all.index, errors='coerce', utc=True)
    if df_all.index.tz is None:
        df_all.index = df_all.index.tz_localize('UTC')
    else:
        df_all.index = df_all.index.tz_convert('UTC')

    for pair_name in pairs.keys():
        cols = [c for c in df_all.columns if c.lower().startswith(f"{pair_name.lower()}_")]
        if not cols:
            continue

        sub_df = df_all[cols].copy()
        sub_df.columns = [c.split('_')[-1].capitalize() for c in sub_df.columns]
        sub_df = sub_df[[c for c in ['Open', 'High', 'Low', 'Close'] if c in sub_df.columns]]
        sub_df = _apply_lookback_filter(log_stream, sub_df, lookback_days, pair_name)

        if not sub_df.empty:
            base_dfs[pair_name] = sub_df

    return base_dfs


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

        log_stream.write(
            f"[WARN] Tidak ada data yang diunduh untuk {symbol} dengan interval {interval} "
            f"dalam {lookback_days} hari terakhir.\n"
        )
        return None
    except Exception as err:
        log_stream.write(f"[WARN] Gagal mengunduh data untuk {symbol}: {err}\n")
        return None


def load_base_data_mtf(log_stream, pairs, lookback_days, base_interval, use_local_csv_for_pairs, base_path):
    base_dfs = {}
    local_load_success = False

    if use_local_csv_for_pairs:
        file_path = _resolve_local_csv_path(base_path, base_interval)

        if file_path is not None and file_path.exists():
            log_stream.write(f"[INFO] Mencoba memuat data MTF ({base_interval}) dari: {file_path}\n")
            try:
                base_dfs = _read_local_base_data(log_stream, file_path, pairs, lookback_days)
                if base_dfs:
                    local_load_success = True
                    log_stream.write(f"[OK] Berhasil memuat {len(base_dfs)} pairs dari CSV lokal {base_interval}.\n")
                else:
                    log_stream.write(f"[WARN] CSV lokal {file_path} tidak memiliki pasangan yang cocok.\n")
            except Exception as err:
                log_stream.write(f"[ERROR] Gagal memuat CSV MTF: {err}\n")
        else:
            log_stream.write(
                f"[WARN] File CSV lokal untuk interval {base_interval} tidak ditemukan "
                "(contoh akhiran: _m1, _h1, _d1). Memulai download.\n"
            )

    if not local_load_success:
        log_stream.write(f"[INFO] Fallback: Mengunduh data baru untuk interval {base_interval}\n")
        download_count = 0

        for name, sym in pairs.items():
            try:
                log_stream.write(
                    f"[DOWNLOAD] Meminta {sym} [{base_interval}] untuk {lookback_days} hari terakhir...\n"
                )

                df_dl = download_base_symbol(log_stream, sym, lookback_days, base_interval)

                if df_dl is not None and not df_dl.empty:
                    if isinstance(df_dl.columns, pd.MultiIndex):
                        df_dl.columns = [c[0] if isinstance(c, tuple) else c for c in df_dl.columns]

                    rename_map = {
                        c: c.capitalize()
                        for c in df_dl.columns
                        if c.lower() in ['open', 'high', 'low', 'close', 'volume']
                    }
                    df_dl = df_dl.rename(columns=rename_map)

                    df_dl = df_dl[[c for c in ['Open', 'High', 'Low', 'Close'] if c in df_dl.columns]]
                    df_dl = _apply_lookback_filter(log_stream, df_dl, lookback_days, name)

                    base_dfs[name] = df_dl
                    download_count += 1
                else:
                    log_stream.write(f"[WARN] Data {name} tidak tersedia untuk interval {base_interval}.\n")

            except Exception as err:
                log_stream.write(f"[ERROR] Gagal mengunduh {name} ({base_interval}): {err}\n")

        if download_count > 0:
            log_stream.write(f"[OK] Selesai. {download_count} pair siap dalam buffer {base_interval}.\n")

    return base_dfs
