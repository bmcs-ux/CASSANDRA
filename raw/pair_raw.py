from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
import requests
import warnings
import polars as pl

warnings.filterwarnings("ignore")


INTERVAL_SUFFIX_MAP = {
    '1m': ['m1', '1m'],
    '5m': ['m5', '5m'],
    '15m': ['m15', '15m'],
    '30m': ['m30', '30m'],
    '1h': ['h1', '1h', '60m'],
    '1d': ['d1', '1d'],
}

RESAMPLE_INTERVAL_MAP = {
    '1m': '1min',
    '5m': '5min',
    '15m': '15min',
    '30m': '30min',
    '1h': '1H',
    '1d': '1D',
}


def _apply_lookback_filter(log_stream, df, lookback_days, pair_name):
    if df.empty:
        log_stream.write(f"  [WARN] DataFrame for {pair_name} is empty, skipping lookback filter.\n")
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce', utc=True)

    if df.index.hasnans:
        before_drop = len(df)
        df = df[~df.index.isna()]
        dropped = before_drop - len(df)
        if dropped > 0:
            log_stream.write(
                f"  [WARN] Data for {pair_name} contains {dropped} invalid timestamps and they were dropped.\n"
            )

    if df.empty:
        log_stream.write(f"  [WARN] DataFrame for {pair_name} has no valid timestamps after parsing.\n")
        return df

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
    df_pl = pl.read_csv(str(file_path), try_parse_dates=False, infer_schema_length=2000)
    all_cols = df_pl.columns
    if not all_cols:
        log_stream.write(f"[WARN] CSV lokal {file_path} kosong.\n")
        return base_dfs

    index_col = all_cols[0]
    df_all = pd.DataFrame(df_pl.rename({index_col: "__index__"}).to_dict(as_series=False))
    df_all["__index__"] = pd.to_datetime(df_all["__index__"], errors="coerce", utc=True)
    df_all = df_all.dropna(subset=["__index__"]).set_index("__index__").sort_index()

    if df_all.empty:
        log_stream.write(f"[WARN] CSV lokal {file_path} kosong.\n")
        return base_dfs

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


def _build_exness_urls(url_segment, start_date, end_date):
    """
    Build Exness tick ZIP URLs.
    Rule:
    - Current month => daily ZIP.
    - Past months => monthly ZIP.
    """
    urls = []
    seen_monthly_urls = set()
    today = date.today()
    current_iter_date = start_date

    while current_iter_date <= end_date:
        if current_iter_date.year == today.year and current_iter_date.month == today.month:
            urls.append(
                f"https://ticks.ex2archive.com/ticks/{url_segment}/{current_iter_date.year}/{current_iter_date.month:02d}/"
                f"{current_iter_date.day:02d}/Exness_{url_segment}_{current_iter_date.year}_{current_iter_date.month:02d}_{current_iter_date.day:02d}.zip"
            )
            current_iter_date += timedelta(days=1)
        else:
            monthly_url = (
                f"https://ticks.ex2archive.com/ticks/{url_segment}/{current_iter_date.year}/{current_iter_date.month:02d}/"
                f"Exness_{url_segment}_{current_iter_date.year}_{current_iter_date.month:02d}.zip"
            )
            if monthly_url not in seen_monthly_urls:
                urls.append(monthly_url)
                seen_monthly_urls.add(monthly_url)

            if current_iter_date.month == 12:
                current_iter_date = date(current_iter_date.year + 1, 1, 1)
            else:
                current_iter_date = date(current_iter_date.year, current_iter_date.month + 1, 1)

    return urls


def _extract_first_csv_from_zip_bytes(zip_bytes):
    with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith('.csv')]
        if not csv_names:
            return None
        with zf.open(csv_names[0]) as f:
            return pd.read_csv(f)


def _download_exness_pair_ohlc(log_stream, url_list, pair_name, resample_interval):
    all_ohlc_dfs = []

    for url_file in url_list:
        try:
            response = requests.get(url_file, timeout=30)
            response.raise_for_status()

            df = _extract_first_csv_from_zip_bytes(response.content)
            if df is None or df.empty:
                log_stream.write(f"[WARN] ZIP tanpa CSV valid: {url_file}\n")
                continue

            if 'Timestamp' not in df.columns or 'Bid' not in df.columns:
                log_stream.write(f"[WARN] CSV tidak punya kolom Timestamp/Bid: {url_file}\n")
                continue

            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
            df = df.dropna(subset=['Timestamp', 'Bid']).set_index('Timestamp').sort_index()
            df = df[~df.index.duplicated(keep='first')]

            ohlc_df = df['Bid'].resample(resample_interval).ohlc().dropna()
            if ohlc_df.empty:
                continue
            ohlc_df.columns = ['Open', 'High', 'Low', 'Close']
            all_ohlc_dfs.append(ohlc_df)

        except requests.RequestException as err:
            log_stream.write(f"[WARN] Gagal unduh tick Exness {url_file}: {err}\n")
        except zipfile.BadZipFile:
            log_stream.write(f"[WARN] ZIP tidak valid dari URL: {url_file}\n")
        except Exception as err:
            log_stream.write(f"[WARN] Error proses data tick {pair_name} ({url_file}): {err}\n")

    if not all_ohlc_dfs:
        return pd.DataFrame()

    combined_ohlc_df = pd.concat(all_ohlc_dfs, axis=0).sort_index()
    combined_ohlc_df = combined_ohlc_df[~combined_ohlc_df.index.duplicated(keep='first')]
    return combined_ohlc_df


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
                "(contoh akhiran: _m1, _h1, _d1). Lanjut ke fallback Exness.\n"
            )

    if not local_load_success:
        log_stream.write(f"[INFO] Fallback: Mengunduh data tick Exness untuk interval {base_interval}\n")
        download_count = 0
        resample_interval = RESAMPLE_INTERVAL_MAP.get(str(base_interval).lower(), str(base_interval))
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        for pair_name in pairs.keys():
            url_segment = pair_name.upper()
            url_list = _build_exness_urls(url_segment, start_date, end_date)
            if not url_list:
                log_stream.write(f"[WARN] URL Exness kosong untuk {pair_name}.\n")
                continue

            log_stream.write(
                f"[DOWNLOAD] Exness {pair_name} [{base_interval}] dari {start_date} s/d {end_date} ({len(url_list)} URL).\n"
            )
            pair_df = _download_exness_pair_ohlc(log_stream, url_list, pair_name, resample_interval)
            if pair_df.empty:
                log_stream.write(f"[WARN] Data Exness kosong untuk {pair_name} ({base_interval}).\n")
                continue

            pair_df = _apply_lookback_filter(log_stream, pair_df, lookback_days, pair_name)
            pair_df = pair_df[[c for c in ['Open', 'High', 'Low', 'Close'] if c in pair_df.columns]]

            if not pair_df.empty:
                base_dfs[pair_name] = pair_df
                download_count += 1

        if download_count > 0:
            log_stream.write(f"[OK] Exness fallback selesai. {download_count} pair siap untuk {base_interval}.\n")
        else:
            log_stream.write(f"[ERROR] Exness fallback gagal: tidak ada pair valid untuk {base_interval}.\n")

    return base_dfs


def download_imputation_special_assets(log_stream, assets, lookback_days, base_interval, existing_pairs=None):
    """Unduh instrumen khusus imputasi yang belum tersedia pada dict pair timeframe terkait."""
    existing_pairs = existing_pairs or {}
    downloaded_assets = {}
    resample_interval = RESAMPLE_INTERVAL_MAP.get(str(base_interval).lower(), str(base_interval))
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    existing_names = set(existing_pairs.keys())
    for asset in assets or []:
        func_pair_name = asset.get('func_pair_name')
        url_segment = asset.get('url_segment')
        if not func_pair_name or not url_segment:
            continue

        if func_pair_name in existing_names:
            continue

        url_list = _build_exness_urls(url_segment.upper(), start_date, end_date)
        if not url_list:
            log_stream.write(f"[WARN] URL Exness kosong untuk aset imputasi {func_pair_name}.\n")
            continue

        log_stream.write(
            f"[DOWNLOAD] Aset imputasi {func_pair_name} [{base_interval}] dari {start_date} s/d {end_date} ({len(url_list)} URL).\n"
        )
        pair_df = _download_exness_pair_ohlc(log_stream, url_list, func_pair_name, resample_interval)
        if pair_df.empty:
            log_stream.write(f"[WARN] Data aset imputasi kosong untuk {func_pair_name} ({base_interval}).\n")
            continue

        pair_df = _apply_lookback_filter(log_stream, pair_df, lookback_days, func_pair_name)
        pair_df = pair_df[[c for c in ['Open', 'High', 'Low', 'Close'] if c in pair_df.columns]]
        if pair_df.empty:
            continue

        downloaded_assets[func_pair_name] = pair_df

    log_stream.write(f"[INFO] Aset khusus imputasi terunduh: {len(downloaded_assets)} ({base_interval}).\n")
    return downloaded_assets
