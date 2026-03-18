import os, sys, traceback, pickle, json, re, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from datetime import datetime, timedelta, timezone

# === Konfigurasi Path Modul ===
ROOT_DIR = '/content/drive/MyDrive/books/CASSANDRA/'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# === Helper Reloading (Penting saat Development di Colab) ===
# Memastikan perubahan pada file .py langsung terdeteksi tanpa restart runtime
modules_to_reload = ['parameter', 'restored', 'fitted_models.dcc_garch_process', 'fitted_models.def_varx', 'fitted_models.dcc_garch']
for mod in modules_to_reload:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])

# === Import Modul Utama & Parameter ===
import parameter
import fitted_models.dcc_garch # Explicitly import to ensure it's in sys.modules for reloading
from fitted_models.dcc_garch import DCCGARCH # Tambahkan impor DCCGARCH di sini

# 1. Data Acquisition (Pair & Macro)
# Menggunakan fungsi MTF yang sudah kita sesuaikan
from raw.pair_raw import load_base_data_mtf, download_imputation_special_assets
from raw.makro_raw import download_macro_data # Pastikan nama fungsi sesuai dengan makro_raw.py Anda

# 2. Preprocessing
# Perhatikan penamaan folder 'preprocessing' agar konsisten dengan struktur proyek
from preprocessing.log_return import apply_log_return_to_price
from preprocessing.fred_transform import apply_fred_transformations
from preprocessing.handle_missing import handle_missing_fred_data
from preprocessing.combine_data import combine_log_returns
from preprocessing.stationarity_test import test_and_stationarize_data
from preprocessing.loop_chained_imputation import apply_loop_berantai_imputation

# 3. Model Engine (Granger, VARX, Kalman)
from fitted_models.granger import run_granger_tests, identify_significant_exog # Import the actual Granger test function
from fitted_models.def_varx import fit_varx_or_arx    # Corrected import: use fit_varx_or_arx
from fitted_models.kalman_filter import setup_kalman_filter

# 4. Volatility & Forecasting
from fitted_models.dcc_garch_process import fit_dcc_garch_to_residuals, prepare_residuals_for_dcc_garch # Corrected import: use fit_dcc_garch_to_residuals
from forecast import auto_varx_forecast
from restored import restore_log_returns_to_price # Tetap diimport untuk validasi di Colab

# 5. Pipeline Utilities
# safe_run and check_data_freshness are defined locally, so no need to import from main_utils
# === Helper ===
def safe_run(step_name, log_stream, func, *args, **kwargs):
    """Wrapper agar setiap tahap tetap lanjut walau error dan mencatat ke log_stream.
    log_stream is now the second argument for safe_run and the first argument passed to func.
    """
    log_stream.write(f"\n=== {step_name} ===\n")
    try:
        # Pass log_stream as the first argument to the function being wrapped
        result = func(log_stream, *args, **kwargs)
        log_stream.write(f"[OK] {step_name} berhasil.\n") # Changed unicode to ASCII OK
        return result # Return the actual result, whether it's a single value or a tuple
    except Exception as e:
        log_stream.write(f"[ERROR] {step_name} gagal: {e}\n") # Gunakan tag error ASCII agar log konsisten
        traceback.print_exc(file=log_stream)
        return None # Return None on failure

def check_data_freshness(log_stream, mtf_base_dfs, current_utc_time):
    """
    Memeriksa kesegaran data untuk semua timeframe.
    Catatan: fungsi ini bersifat informatif dan tidak lagi menghentikan pipeline.
    """
    log_stream.write("\n=== Checking Multi-Timeframe Data Freshness ===\n")
    if not mtf_base_dfs:
        return False

    is_fresh_overall = True
    for tf, pairs_dict in mtf_base_dfs.items():
        for pair_name, df in pairs_dict.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            latest_ts = df.index.max()
            if latest_ts.tz is None:
                latest_ts = latest_ts.tz_localize('UTC')

            if tf == 'D1':
                tolerance = timedelta(hours=24)
            elif tf == 'H1':
                tolerance = timedelta(hours=3)
            else:  # M1
                tolerance = timedelta(minutes=10)

            if latest_ts < (current_utc_time - tolerance):
                log_stream.write(f"[ALERT] Data {tf} {pair_name} stale! Last: {latest_ts}\n")
                is_fresh_overall = False

    return is_fresh_overall


def align_mtf_data_to_common_close(log_stream, mtf_base_dfs):
    """
    Menyamakan horizon waktu antar-timeframe agar training tetap berjalan walau M1 tidak fresh.

    Strategi:
    1) Gunakan common close berbasis D1 (timeframe paling rendah frekuensinya) jika tersedia.
    2) Jika D1 tidak tersedia, gunakan minimum latest timestamp lintas semua timeframe.
    3) Potong seluruh data TF pada index <= cutoff tersebut.
    """
    if not mtf_base_dfs:
        return mtf_base_dfs

    d1_dict = mtf_base_dfs.get('D1', {})
    d1_last_ts = []
    for df in d1_dict.values():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        ts = df.index.max()
        if ts.tz is None:
            ts = ts.tz_localize('UTC')
        d1_last_ts.append(ts)

    if d1_last_ts:
        reference_close_ts = min(d1_last_ts)
        log_stream.write(f"[INFO] Menggunakan D1 common close sebagai cutoff: {reference_close_ts}\n")
    else:
        all_latest = []
        for pairs_dict in mtf_base_dfs.values():
            for df in pairs_dict.values():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                ts = df.index.max()
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')
                all_latest.append(ts)

        if not all_latest:
            log_stream.write("[WARN] Tidak ada timestamp valid untuk alignment MTF.\n")
            return mtf_base_dfs

        reference_close_ts = min(all_latest)
        log_stream.write(f"[WARN] D1 tidak tersedia. Fallback cutoff lintas TF: {reference_close_ts}\n")

    aligned = {}
    for tf, pairs_dict in mtf_base_dfs.items():
        aligned[tf] = {}
        for pair_name, df in pairs_dict.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                aligned[tf][pair_name] = df
                continue

            idx = df.index
            if idx.tz is None:
                df = df.copy()
                df.index = idx.tz_localize('UTC')

            df_cut = df[df.index <= reference_close_ts].copy()
            if df_cut.empty:
                log_stream.write(
                    f"[WARN] Setelah alignment, data {tf} {pair_name} kosong (cutoff={reference_close_ts}).\n"
                )
            aligned[tf][pair_name] = df_cut

    return aligned


def summarize_dataframe(df, label):
    """Ringkas statistik data inti untuk keperluan output review sebelum preprocessing."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            'label': label,
            'shape': (0, 0),
            'total_missing': 0,
            'missing_per_column': {},
            'index_start': None,
            'index_end': None,
            'dtypes': {},
        }

    return {
        'label': label,
        'shape': df.shape,
        'total_missing': int(df.isna().sum().sum()),
        'missing_per_column': df.isna().sum().sort_values(ascending=False).to_dict(),
        'index_start': df.index.min(),
        'index_end': df.index.max(),
        'dtypes': {c: str(t) for c, t in df.dtypes.items()},
    }


def _log_dataframe_summary(log_stream, summary):
    log_stream.write(f"\n[SUMMARY] {summary['label']}\n")
    log_stream.write(f"  shape={summary['shape']}\n")
    log_stream.write(f"  total_missing={summary['total_missing']}\n")
    log_stream.write(f"  index_range=({summary['index_start']}, {summary['index_end']})\n")
    if summary['missing_per_column']:
        top_missing = list(summary['missing_per_column'].items())[:10]
        log_stream.write("  top_missing_cols=\n")
        for col, miss_count in top_missing:
            log_stream.write(f"    - {col}: {int(miss_count)}\n")


def _combine_mtf_pair_ohlc(tf_pairs_dict):
    """Gabungkan dict pair->OHLC menjadi satu DataFrame berkolom <PAIR>_<OHLC>."""
    combined = None
    for pair_name, df in tf_pairs_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
        if not cols:
            continue

        renamed = df[cols].rename(columns={c: f"{pair_name}_{c}" for c in cols})
        combined = renamed.copy() if combined is None else combined.merge(renamed, left_index=True, right_index=True, how='outer')

    return pd.DataFrame() if combined is None else combined.sort_index()


def _split_mtf_pair_ohlc(combined_df, original_tf_pairs_dict):
    """Pisahkan DataFrame gabungan kembali ke dict pair->OHLC mengikuti pasangan asli."""
    rebuilt = {}
    for pair_name, orig_df in original_tf_pairs_dict.items():
        if not isinstance(orig_df, pd.DataFrame) or orig_df.empty:
            rebuilt[pair_name] = orig_df
            continue

        pair_cols = [c for c in combined_df.columns if c.startswith(f"{pair_name}_")]
        if not pair_cols:
            rebuilt[pair_name] = orig_df
            continue

        tmp = combined_df[pair_cols].copy().rename(columns={c: c.replace(f"{pair_name}_", "") for c in pair_cols})
        ordered_cols = [c for c in orig_df.columns if c in tmp.columns]
        rebuilt[pair_name] = tmp[ordered_cols] if ordered_cols else tmp

    return rebuilt


def _plot_missing_overview(df, title):
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"[INFO] Plot dilewati: {title} kosong.")
        return

    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0].head(20)
    if missing_counts.empty:
        print(f"[INFO] Plot dilewati: {title} tidak punya missing value.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=missing_counts.index, y=missing_counts.values, ax=ax, color='orange')
    ax.set_title(f"Missing Value Overview - {title}")
    ax.set_ylabel("Jumlah Null")
    ax.set_xlabel("Kolom")
    ax.tick_params(axis='x', rotation=75)
    fig.tight_layout()
    plt.show(block=False)
    plt.close(fig)
    print(f"[INFO] Plot missing untuk '{title}' berhasil ditampilkan.")


def _print_dataframe_summary(summary):
    print(f"\n[SUMMARY] {summary['label']}")
    print(f"  shape={summary['shape']}")
    print(f"  total_missing={summary['total_missing']}")
    print(f"  index_range=({summary['index_start']}, {summary['index_end']})")
    if summary['missing_per_column']:
        for col, miss_count in list(summary['missing_per_column'].items())[:10]:
            print(f"    - {col}: {int(miss_count)}")


def _clear_console_output():
    print("\n" + "=" * 80)
    print("[INFO] Opsi dijalankan. Menu diperbarui.")
    print("=" * 80)


def _input_menu(prompt, valid_choices, default_choice):
    try:
        raw_choice = input(prompt)
        choice = (raw_choice or "").strip().lower()
    except EOFError:
        return default_choice
    if len(choice) > 1:
        choice = choice[0]
    return choice if choice in valid_choices else default_choice


def _ensure_pkl_dir(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)


def _save_pickle(log_stream, obj, path, label):
    try:
        _ensure_pkl_dir(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        log_stream.write(f"[OK] {label} disimpan ke {path}\n")
        print(f"[INFO] {label} disimpan: {path}")
    except Exception as err:
        log_stream.write(f"[WARN] Gagal menyimpan {label} ke {path}: {err}\n")
        print(f"[WARN] Gagal simpan {label}: {err}")


def _load_pickle_if_exists(log_stream, path, label):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        log_stream.write(f"[OK] {label} dimuat dari cache {path}\n")
        return loaded
    except Exception as err:
        log_stream.write(f"[WARN] Cache {label} di {path} tidak dapat dibaca: {err}\n")
        return None


def _save_parquet(log_stream, mtf_base_dfs, base_dir, label, asset_registry):
    import polars as pl

    try:
        for tf, pair_dict in (mtf_base_dfs or {}).items():
            for pair, df in (pair_dict or {}).items():
                if pair not in asset_registry or df is None:
                    continue

                meta = asset_registry[pair]
                asset_class = meta['asset_class']
                symbol = meta['symbol']

                if isinstance(df, pl.DataFrame):
                    parquet_df = df.clone()
                else:
                    pandas_df = df.copy()
                    index_name = pandas_df.index.name or '__index__'
                    parquet_df = pl.from_pandas(pandas_df.reset_index().rename(columns={pandas_df.index.name or 'index': index_name}))

                dir_path = os.path.join(
                    base_dir,
                    f"asset_class={asset_class}",
                    f"symbol={symbol}",
                    f"timeframe={tf}",
                )
                os.makedirs(dir_path, exist_ok=True)

                now = datetime.utcnow()
                file_name = f"data_{now.year}_{now.month:02d}.parquet"
                parquet_df.write_parquet(os.path.join(dir_path, file_name))

        log_stream.write(f"[OK] {label} saved as parquet\n")
        print(f"[INFO] {label} disimpan sebagai parquet di {base_dir}")
    except Exception as err:
        log_stream.write(f"[WARN] Failed saving parquet: {err}\n")
        print(f"[WARN] Gagal simpan parquet {label}: {err}")


def _load_parquet_lazy(base_dir, asset_registry):
    import glob
    import polars as pl

    result = {}

    for pair, meta in (asset_registry or {}).items():
        asset_class = meta['asset_class']
        symbol = meta['symbol']
        path = os.path.join(
            base_dir,
            f"asset_class={asset_class}",
            f"symbol={symbol}",
            "timeframe=*",
        )

        files = glob.glob(os.path.join(path, "*.parquet"))
        if not files:
            continue

        files_by_tf = {}
        for file_path in files:
            tf = file_path.split("timeframe=")[-1].split(os.sep)[0]
            files_by_tf.setdefault(tf, []).append(file_path)

        for tf, tf_files in files_by_tf.items():
            pdf = pl.scan_parquet(sorted(tf_files)).collect().to_pandas()
            index_col = '__index__' if '__index__' in pdf.columns else None
            if index_col is not None:
                pdf[index_col] = pd.to_datetime(pdf[index_col], utc=True, errors='coerce')
                pdf = pdf.set_index(index_col).sort_index()
                pdf.index.name = None if index_col == '__index__' else index_col
            result.setdefault(tf, {})[pair] = pdf

    return result


def review_and_confirm_mtf_data(log_stream, mtf_base_dfs, fred_df, interactive=False, imputation_assets_by_tf=None):
    """
    Review data MTF sebelum PREPROCESSING MTF.

    Opsi: plotting, imputasi loop berantai, compare dengan FRED, save Parquet, back (ulang menu), konfirmasi lanjut.
    """
    log_stream.write("\n[INFO] Review mtf_base_dfs sebelum PREPROCESSING MTF\n")
    initial_mtf_snapshot = {tf: {p: df.copy() for p, df in pairs_dict.items()} for tf, pairs_dict in mtf_base_dfs.items()}
    pending_imputation = False

    for tf, pairs_dict in mtf_base_dfs.items():
        tf_combined = _combine_mtf_pair_ohlc(pairs_dict)
        summary = summarize_dataframe(tf_combined, f"mtf_base_dfs[{tf}] (sebelum menu)")
        _log_dataframe_summary(log_stream, summary)
        _print_dataframe_summary(summary)

    if not interactive:
        log_stream.write("[INFO] Interactive review MTF dinonaktifkan. Lanjut otomatis.\n")
        return mtf_base_dfs

    while True:
        choice = _input_menu(
            "\n[MTF MENU] Pilih: [p]lot, [i]mputasi_loop_berantai, [c]ompare_fred, [s]ave_parquet, [b]ack, [k]onfirmasi : ",
            {'p', 'i', 'c', 's', 'b', 'k'},
            'k',
        )
        if choice == 'p':
            for tf, pairs_dict in mtf_base_dfs.items():
                _plot_missing_overview(_combine_mtf_pair_ohlc(pairs_dict), f"MTF {tf}")
            _clear_console_output()
            continue

        if choice == 'i':
            for tf, pairs_dict in mtf_base_dfs.items():
                combined = _combine_mtf_pair_ohlc(pairs_dict)
                special_pairs = (imputation_assets_by_tf or {}).get(tf, {})
                if special_pairs:
                    special_combined = _combine_mtf_pair_ohlc(special_pairs)
                    if not special_combined.empty:
                        combined = pd.concat([combined, special_combined], axis=1)

                if combined.empty:
                    continue
                try:
                    imputed, stats = apply_loop_berantai_imputation(log_stream, combined)
                    before_missing = int(combined.isna().sum().sum())
                    after_missing = int(imputed.isna().sum().sum())
                    log_stream.write(f"[INFO] Imputasi loop berantai diterapkan untuk {tf}. stats={stats}\n")
                    log_stream.write(
                        f"[INFO] Ringkasan imputasi {tf}: missing_before={before_missing}, "
                        f"missing_after={after_missing}, reduced={before_missing - after_missing}\n"
                    )
                    print(
                        f"[IMPUTASI] {tf}: missing_before={before_missing}, "
                        f"missing_after={after_missing}, reduced={before_missing - after_missing}"
                    )
                    mtf_base_dfs[tf] = _split_mtf_pair_ohlc(imputed, pairs_dict)
                    pending_imputation = True
                except ValueError as err:
                    log_stream.write(f"[WARN] Imputasi loop berantai {tf} dilewati: {err}\n")
                    print(f"[WARN] Imputasi loop berantai {tf} dilewati: {err}")
            _clear_console_output()
            continue

        if choice == 'c':
            fred_summary = summarize_dataframe(fred_df, 'fred_df')
            _log_dataframe_summary(log_stream, fred_summary)
            _print_dataframe_summary(fred_summary)
            for tf, pairs_dict in mtf_base_dfs.items():
                tf_summary = summarize_dataframe(_combine_mtf_pair_ohlc(pairs_dict), f"mtf_base_dfs[{tf}]")
                _log_dataframe_summary(log_stream, tf_summary)
                _print_dataframe_summary(tf_summary)
                log_stream.write(
                    f"[COMPARE] {tf}: total_missing={tf_summary['total_missing']} | "
                    f"fred_missing={fred_summary['total_missing']}\n"
                )
                print(
                    f"[COMPARE] {tf}: total_missing={tf_summary['total_missing']} | "
                    f"fred_missing={fred_summary['total_missing']}"
                )
            _clear_console_output()
            continue

        if choice == 'b':
            mtf_base_dfs = {tf: {p: df.copy() for p, df in pairs.items()} for tf, pairs in initial_mtf_snapshot.items()}
            pending_imputation = False
            log_stream.write("[INFO] Back dipilih: hasil imputasi dibatalkan, data kembali ke snapshot awal.\n")
            print("[INFO] Back: hasil imputasi dibatalkan.")
            for tf, pairs_dict in mtf_base_dfs.items():
                summary = summarize_dataframe(_combine_mtf_pair_ohlc(pairs_dict), f"mtf_base_dfs[{tf}] (setelah back)")
                _log_dataframe_summary(log_stream, summary)
                _print_dataframe_summary(summary)
            _clear_console_output()
            continue

        if choice == 's':
            base_dir = getattr(parameter, 'BASE_DATA_DIR', '/content/base_data')
            _save_parquet(log_stream, mtf_base_dfs, base_dir, 'mtf_base_dfs', parameter.ASSET_REGISTRY)
            _clear_console_output()
            continue

        if choice == 'k':
            if pending_imputation:
                log_stream.write("[INFO] Konfirmasi imputasi diterima untuk MTF.\n")
                print("[INFO] Konfirmasi: hasil imputasi MTF dipakai.")
            log_stream.write("[INFO] Konfirmasi diterima. Lanjut ke tahap berikutnya (PREPROCESSING MTF).\n")
            for tf, pairs_dict in mtf_base_dfs.items():
                summary = summarize_dataframe(_combine_mtf_pair_ohlc(pairs_dict), f"mtf_base_dfs[{tf}] (setelah menu)")
                _log_dataframe_summary(log_stream, summary)
                _print_dataframe_summary(summary)
            return mtf_base_dfs


def review_and_confirm_fred_data(log_stream, fred_df, mtf_base_dfs, interactive=False):
    """
    Review data FRED sebelum dipakai di PREPROCESSING MTF.

    Opsi: plotting, imputasi FRED, compare ke MTF, save PKL, back, konfirmasi.
    """
    original_fred_df = fred_df.copy() if isinstance(fred_df, pd.DataFrame) else fred_df
    pending_imputation = False
    initial_summary = summarize_dataframe(fred_df, "fred_df (sebelum menu)")
    _log_dataframe_summary(log_stream, initial_summary)
    _print_dataframe_summary(initial_summary)

    if not interactive:
        log_stream.write("[INFO] Interactive review FRED dinonaktifkan. Lanjut otomatis.\n")
        return fred_df

    while True:
        choice = _input_menu(
            "\n[FRED MENU] Pilih: [p]lot, [i]mputasi, [c]ompare_mtf, [s]ave_pkl, [b]ack, [k]onfirmasi : ",
            {'p', 'i', 'c', 's', 'b', 'k'},
            'k',
        )

        if choice == 'p':
            _plot_missing_overview(fred_df, "FRED")
            _clear_console_output()
            continue

        if choice == 'i':
            print("Pilih metode imputasi FRED:")
            print("1) Kalibrasi Hybrid-FRED (Nowcasted Residuals)")
            print("2) MIDAS Simulation (Information Density Weighting)")
            method = _input_menu("Metode [1/2]: ", {'1', '2'}, '1')
            if method == '1':
                log_stream.write("[INFO] Metode 1 dipilih (placeholder): saat ini fallback ke apply_loop_berantai_imputation.\n")
            else:
                log_stream.write("[INFO] Metode 2 dipilih (placeholder): saat ini fallback ke apply_loop_berantai_imputation.\n")

            try:
                imputed_fred_df, stats = apply_loop_berantai_imputation(log_stream, fred_df)
                before_missing = int(fred_df.isna().sum().sum()) if isinstance(fred_df, pd.DataFrame) else 0
                after_missing = int(imputed_fred_df.isna().sum().sum()) if isinstance(imputed_fred_df, pd.DataFrame) else 0
                fred_df = imputed_fred_df
                pending_imputation = True
                log_stream.write(f"[INFO] Imputasi sementara FRED selesai via loop berantai. stats={stats}\n")
                log_stream.write(
                    f"[INFO] Ringkasan imputasi FRED: missing_before={before_missing}, "
                    f"missing_after={after_missing}, reduced={before_missing - after_missing}\n"
                )
                print(
                    f"[IMPUTASI] FRED: missing_before={before_missing}, "
                    f"missing_after={after_missing}, reduced={before_missing - after_missing}"
                )
            except ValueError as err:
                log_stream.write(f"[WARN] Imputasi FRED sementara dilewati: {err}\n")
                print(f"[WARN] Imputasi FRED sementara dilewati: {err}")
            _clear_console_output()
            continue

        if choice == 'c':
            fred_summary = summarize_dataframe(fred_df, 'fred_df')
            _log_dataframe_summary(log_stream, fred_summary)
            _print_dataframe_summary(fred_summary)
            for tf, pairs_dict in mtf_base_dfs.items():
                tf_summary = summarize_dataframe(_combine_mtf_pair_ohlc(pairs_dict), f"mtf_base_dfs[{tf}]")
                _log_dataframe_summary(log_stream, tf_summary)
                _print_dataframe_summary(tf_summary)
                log_stream.write(
                    f"[COMPARE] fred_df vs {tf}: fred_missing={fred_summary['total_missing']} | "
                    f"mtf_missing={tf_summary['total_missing']}\n"
                )
                print(
                    f"[COMPARE] fred_df vs {tf}: fred_missing={fred_summary['total_missing']} | "
                    f"mtf_missing={tf_summary['total_missing']}"
                )
            _clear_console_output()
            continue

        if choice == 'b':
            fred_df = original_fred_df.copy() if isinstance(original_fred_df, pd.DataFrame) else original_fred_df
            pending_imputation = False
            summary = summarize_dataframe(fred_df, "fred_df (setelah back)")
            _log_dataframe_summary(log_stream, summary)
            _print_dataframe_summary(summary)
            _clear_console_output()
            continue

        if choice == 's':
            cache_dir = getattr(parameter, 'PKL_CACHE_DIR', '/content/.pkl')
            fred_pkl_path = os.path.join(cache_dir, getattr(parameter, 'FRED_DF_PKL_NAME', 'fred_df.pkl'))
            _save_pickle(log_stream, fred_df, fred_pkl_path, 'fred_df')
            _clear_console_output()
            continue

        if choice == 'k':
            if pending_imputation:
                log_stream.write("[INFO] Konfirmasi imputasi diterima untuk FRED.\n")
                print("[INFO] Konfirmasi: hasil imputasi FRED dipakai.")
            final_summary = summarize_dataframe(fred_df, "fred_df (setelah menu)")
            _log_dataframe_summary(log_stream, final_summary)
            _print_dataframe_summary(final_summary)
            log_stream.write("[INFO] Konfirmasi FRED diterima. Lanjut ke tahap berikutnya (PREPROCESSING MTF).\n")
            return fred_df
# ============================================================
# 1. LOAD DATA
# ============================================================

# Removed redundant load_base_data wrapper function

# Removed redundant load_fred_data wrapper function

# ============================================================
# 2. PREPROCESSING
# ============================================================


def preprocess_data_tf(log_stream, b_dfs, fred_df, fred_meta, tf_label):
    """
    Preprocessing khusus untuk satu timeframe tertentu dengan penanganan
    Truth Value DataFrame yang aman.
    """
    log_stream.write(f"\n--- Preprocessing Timeframe: {tf_label} ---\n")

    # 1. Hitung Log Return secara mentah
    res_raw = safe_run(f"Log Return {tf_label}", log_stream, apply_log_return_to_price, b_dfs)
    log_returns_raw = res_raw if res_raw is not None else {}
    for pair_name, df in log_returns_raw.items():
        if not df.empty:
            log_stream.write(f"[DEBUG] Columns in log_returns_raw for {pair_name}: {list(df.columns)}\n")

    # 2. Gabungkan hasil ke dalam Dictionary
    # Menggunakan log_returns_raw secara langsung untuk memastikan semua kolom log return tetap ada
    log_returns_dict = log_returns_raw

    for pair_name, df in log_returns_dict.items():
        if not df.empty:
            log_stream.write(f"[DEBUG] Columns in log_returns_dict (after direct assignment) for {pair_name}: {list(df.columns)}\n")


    # 3. Gabungkan hasil ke dalam DataFrame (Cross-Asset Exog Pool)
    # Ini adalah bagian yang paling krusial untuk menghindari ValueError
    combined_df_res = safe_run(f"Combine DF {tf_label}", log_stream, combine_log_returns, log_returns_raw, return_type='df')

    if isinstance(combined_df_res, pd.DataFrame):
        combined_df = combined_df_res
        log_stream.write(f"[DEBUG] Combined log returns (DF) for {tf_label}. Columns: {list(combined_df.columns) if not combined_df.empty else 'Empty'}\n")
    else:
        combined_df = pd.DataFrame()
        log_stream.write(f"[DEBUG] Combined log returns (DF) for {tf_label} is empty.\n")

    cleaned_fred_df = None
    if tf_label == 'D1' and fred_df is not None:
        # 1. Transformasi (Log-Return/Diff)
        fred_transform = safe_run(f"FRED transform {tf_label}", log_stream, apply_fred_transformations, fred_df, parameter.FRED_SERIES, fred_meta)

        if fred_transform is None:
            log_stream.write(f" [WARN] Gagal melakukan transformasi FRED data untuk timeframe {tf_label}\n")
        else:
            # 2. Penanganan Missing Data (Pembersihan Akhir)
            # PASTIKAN log_stream dimasukkan di sini
            cleaned_fred_df = safe_run(
                f"Clean FRED {tf_label}",
                log_stream,
                handle_missing_fred_data,
                fred_transform,
                missing_threshold=parameter.FRED_MISSING_THRESHOLD,
            )

    return log_returns_dict, cleaned_fred_df, combined_df

def setup_kalman_filter_compat(log_stream, df_m1):
    """Kompatibilitas untuk dua signature setup_kalman_filter (baru/lama)."""
    try:
        return setup_kalman_filter(log_stream, df_m1)
    except TypeError:
        # fallback untuk versi lama yang hanya menerima df_m1
        return setup_kalman_filter(df_m1)

# ============================================================
# 3. GRANGER TESTS
# ============================================================

def run_granger_all(log_stream, log_returns, cleaned_fred, timeframe_label="D1"): # Renamed from run_granger_all_mtf
    """
    Versi MTF dari Granger Test.
    D1: Menguji FRED + Pair lain terhadap Pair target.
    H1: Hanya menguji Pair lain (Cross-Asset) terhadap Pair target.
    """
    if not log_returns:
        return pd.DataFrame(), {}

    # 1. Tentukan Target (semua pair dalam timeframe ini)
    granger_targets = {}
    for pair, df in log_returns.items():
        if not df.empty:
             log_return_cols = [col for col in df.columns if col.endswith('_Log_Return')]
             for col in log_return_cols:
                 granger_targets[f"{pair}_{col}"] = df[[col]].dropna()

    # 2. Tentukan Potensi Penyebab (Exogenous)
    granger_exogs = {}

    # HANYA masukkan FRED jika kita di timeframe D1
    if timeframe_label == "D1" and isinstance(cleaned_fred, pd.DataFrame) and not cleaned_fred.empty:
        for col in cleaned_fred.columns:
            if col not in ['release_date', 'effective_until_next_release', 'date']:
                granger_exogs[col] = cleaned_fred[[col]].dropna()

    # 3. Tambahkan Cross-Asset Causality (Pair lain sebagai exog)
    # Ini krusial untuk H1: melihat apakah JPY mempengaruhi EUR, dll.
    all_potential_causes = {**granger_exogs, **granger_targets}

    # Jalankan uji Granger
    # Di dalam main.py bagian 4 (FITTING)
    # -----------------------------------

    # Jalankan uji Granger
    granger_df = safe_run(f"Granger {timeframe_label}", log_stream, run_granger_tests,
                          log_returns, # Corrected: Pass log_returns directly
                          maxlag_test=parameter.maxlag_granger,
                          alpha=parameter.alpha_granger,
                          exogenous_data_dict=cleaned_fred if timeframe_label == 'D1' else None)

    # Identifikasi variabel yang masuk ke model VARX (Exog Map)
    # Inilah yang mencegah error "TypeError: '<' not supported"
    exog_map_tf = safe_run(f"Map Significant Exog {timeframe_label}", log_stream, identify_significant_exog,
                          granger_df, parameter.alpha_granger)

    #mtf_exog_maps[timeframe_label] = exog_map_tf
    if exog_map_tf:
        log_stream.write(f"  [INFO] Exog map for {timeframe_label}: {exog_map_tf}\n")

    return granger_df, exog_map_tf

# ============================================================
# 4\" MODEL FITTING
# ============================================================

def combine_fred_for_model(log_stream, cleaned_fred_df, timeframe_label="D1"):
    """
    Combines cleaned FRED data into a single DataFrame for model fitting.
    Args:
        log_stream (StringIO): Stream for logging.
        cleaned_fred_df (pd.DataFrame): DataFrame containing cleaned FRED data.
        timeframe_label (str): Current timeframe label (e.g., 'D1', 'H1').
    Returns:
        pd.DataFrame: Combined FRED DataFrame for model fitting.
    """
    # Proteksi: Data FRED tidak relevan untuk timeframe intraday
    if timeframe_label != "D1":
        log_stream.write(f"[INFO] Timeframe {timeframe_label} terdeteksi. Melewati penggabungan FRED karena perbedaan frekuensi data.\n")
        return pd.DataFrame()

    if cleaned_fred_df.empty:
        log_stream.write("[WARN] Data FRED bersih kosong. Tidak ada yang digabungkan.\n")
        return pd.DataFrame()

    # Assuming cleaned_fred_df is already a single DataFrame with appropriate columns
    log_stream.write(f"[OK] Data FRED bersih gabungan berhasil dibuat untuk model fitting. Shape: {cleaned_fred_df.shape}\n")

    return cleaned_fred_df


def fit_models(log_stream, log_returns_dict, exog_map, combined_fred_for_model_df, timeframe_label):
    import re
    models = {} # Initialize the models dictionary
    summaries = []

    if not log_returns_dict:
        log_stream.write("[WARN] Data log return kosong. Lewati fitting model.\n")
        return models, pd.DataFrame()

    # Buat pool eksogen gabungan. Ini akan menjadi sumber semua variabel eksogen yang mungkin.
    all_available_model_exogs = pd.DataFrame()
    if combined_fred_for_model_df is not None and not combined_fred_for_model_df.empty:
        all_available_model_exogs = pd.concat([all_available_model_exogs, combined_fred_for_model_df], axis=1)

    # Tambahkan semua log-returns sebagai potensi eksogen cross-asset
    # Pastikan tidak ada duplikasi kolom saat concat
    all_log_returns_df = pd.concat([df for df in log_returns_dict.values()], axis=1, join='outer')
    all_log_returns_df = all_log_returns_df.loc[:,~all_log_returns_df.columns.duplicated()]
    # ffill() sebelum dropna(how='all') agar missing values dari join='outer' ditangani
    all_log_returns_df = all_log_returns_df.ffill().dropna(how='all')
    all_available_model_exogs = pd.concat([all_available_model_exogs, all_log_returns_df], axis=1)
    all_available_model_exogs = all_available_model_exogs.loc[:,~all_available_model_exogs.columns.duplicated()].ffill().dropna(how='all')


    if all_available_model_exogs.empty:
        log_stream.write("[WARN] Tidak ada data eksogen yang tersedia setelah digabungkan dan dibersihkan. Model akan berjalan tanpa eksogen.\n")

    # Iterate over each group defined in VARX_ENDOG_GROUPS
    for group_name, endog_cols_in_group in parameter.VARX_ENDOG_GROUPS.items(): # Use parameter.VARX_ENDOG_GROUPS
        log_stream.write(f"\n[INFO] Melatih model untuk grup: {group_name}\n")

        # 1. Create a combined DataFrame of the endogenous variables specified in the group
        endog_data_frames = []
        for endog_full_name in endog_cols_in_group:
            pair_name_from_endog = None
            match = re.match(r'(.+?)_(Open|High|Low|Close)_Log_Return', endog_full_name)
            if match:
                pair_name_from_endog = match.group(1)

            # Simplified check: directly look for endog_full_name in the columns of the DataFrame for the pair
            if pair_name_from_endog and pair_name_from_endog in log_returns_dict:
                df_log_return_pair = log_returns_dict[pair_name_from_endog]
                # The column name in df_log_return_pair is already the full name, e.g., 'GBPUSD_Close_Log_Return'
                # FIX: Remove incorrect stripping of pair_name prefix. Use endog_full_name directly.
                if not df_log_return_pair.empty and endog_full_name in df_log_return_pair.columns:
                    # FIX: No renaming needed if column is already correct
                    endog_data_frames.append(df_log_return_pair[[endog_full_name]])
                else:
                    log_stream.write(f"[WARN] Kolom '{endog_full_name}' tidak ditemukan di data '{pair_name_from_endog}'. Lewati.\n")
            else:
                log_stream.write(f"[WARN] Data untuk '{endog_full_name}' tidak ditemukan di log_returns_dict. Lewati.\n")

        if not endog_data_frames:
            log_stream.write(f"[WARN] Tidak ada data endogenous valid untuk grup {group_name}. Lewati fitting model.\n")
            continue

        df_endog_group = pd.concat(endog_data_frames, axis=1, join='inner').dropna()

        if df_endog_group.empty:
            log_stream.write(f"[WARN] Data endogenous grup {group_name} kosong setelah penggabungan/dropna. Lewati fitting model.\n")
            continue

        # 2. Identify the relevant exogenous variables for the current group
        exog_cols_to_use = []
        for endog_item in endog_cols_in_group:
            # Get significant exogs for each individual endogenous variable in the group
            current_endog_exogs = exog_map.get(endog_item, [])
            # Add them to the list, avoiding duplicates and excluding if already an endogenous variable in this group
            for exog_cand in current_endog_exogs:
                if exog_cand not in exog_cols_to_use and exog_cand not in endog_cols_in_group:
                    exog_cols_to_use.append(exog_cand)

        # Filter exog_cols_to_use to only include columns actually present in all_available_model_exogs
        final_exog_cols = [col for col in exog_cols_to_use if col in all_available_model_exogs.columns]

        df_combined_for_model = df_endog_group.copy()
        if final_exog_cols:
            log_stream.write(f"[INFO] Menggunakan eksogen signifikan untuk grup {group_name}: {final_exog_cols}\n")
            # Merge endogenous data with selected exogenous variables
            exog_subset = all_available_model_exogs[final_exog_cols]
            # Align indices and ffill
            df_aligned = df_endog_group.join(exog_subset, how='inner')
            df_aligned[final_exog_cols] = df_aligned[final_exog_cols].ffill()
            df_combined_for_model = df_aligned.dropna()
        else:
           log_stream.write("[INFO] Tidak ada eksogen signifikan yang teridentifikasi atau data eksogen lengkap tidak tersedia. Akan mencoba fit AR(X) tanpa eksogen.\n")

        if df_combined_for_model.empty:
            log_stream.write(f"[WARN] Data gabungan kosong setelah dropna untuk grup {group_name}. Lewati fitting model.\n")
            continue

        try:
            log_stream.write(f"[INFO] Fitting model VARX/ARX untuk grup {group_name}, Endog: {endog_cols_in_group}\n")
            # Fit the VARX model (will be VARMAX if multiple endog, SARIMAX if single)
            model_result = fit_varx_or_arx(
                log_stream, df_combined_for_model,
                endog_cols=endog_cols_in_group, # Pass the list of endogenous columns for the group
                exog_cols=final_exog_cols,
                maxlags=parameter.maxlag_test # Use parameter.maxlag_test
            )

            # Store the fitted model result in the dictionary.
            # The key should be the group name.
            model_key = group_name
            models[model_key] = model_result # Use the group name as the key
            models[model_key]['endog_names'] = endog_cols_in_group # Store the endogenous column names
            models[model_key]['exog_names'] = final_exog_cols # Store exog_names here

            # Extract R2 for summary (handle dict for VARMAX or single value for ARX)
            r2_values = model_result.get("R2", np.nan)

            # If VARMAX, R2 will be a dictionary of R2s for each endogenous variable
            if isinstance(r2_values, dict):
                for endog_col_name, r2_val in r2_values.items():
                    summaries.append({
                        "Group": group_name,
                        "Endog_Variable": endog_col_name,
                        "Model_Type": model_result["model_type"],
                        "Lags_Used": model_result["lags_used"],
                        "Num_Exog": len(final_exog_cols),
                        "R2": r2_val
                    })
            else: # If SARIMAX (single endogenous in group, though this path should be mostly for VARMAX now)
                 if len(endog_cols_in_group) == 1:
                      summaries.append({
                          "Group": group_name,
                          "Endog_Variable": endog_cols_in_group[0],
                          "Model_Type": model_result["model_type"],
                          "Lags_Used": model_result["lags_used"],
                          "Num_Exog": len(final_exog_cols),
                          "R2": r2_values # Assuming r2_values is a single float here
                      })
                 else:
                     log_stream.write(f"[WARN] R2 format tidak terduga untuk grup {group_name}. Tidak dapat menambahkan ke ringkasan.\n")

            log_stream.write(f"[OK] Model {model_result['model_type']} fitted for group {group_name}. Lags={model_result['lags_used']}, Num Exog={len(final_exog_cols)}, R\u00b2={r2_values}\n")

        except ValueError as ve:
            log_stream.write(f"[ERROR] Gagal fitting model untuk grup {group_name}: {ve}\n")
            summaries.append({
                "Group": group_name, "Endog_Variable": "N/A", "Model_Type": None,
                "Lags_Used": None, "Num_Exog": len(final_exog_cols), "R2": np.nan
            })
        except RuntimeError as err_e:
            log_stream.write(f"[ERROR] Gagal fitting model untuk grup {group_name}: {err_e}\n")
            summaries.append({
                "Group": group_name, "Endog_Variable": "N/A", "Model_Type": None,
                "Lags_Used": None, "Num_Exog": len(final_exog_cols), "R2": np.nan
            })
        except Exception as e:
            log_stream.write(f"[ERROR] Terjadi error saat fitting model untuk grup {group_name}: {e}\n")
            summaries.append({
                "Group": group_name, "Endog_Variable": "N/A", "Model_Type": None,
                "Lags_Used": None, "Num_Exog": len(final_exog_cols), "R2": np.nan
            })

    # Display summary of fitted models
    varx_fit_df = pd.DataFrame(summaries)

    log_stream.write("\n[INFO] Ringkasan Hasil Fitting Model VARX/ARX (Grup):\n")
    if not varx_fit_df.empty:
        log_stream.write(varx_fit_df.dropna(subset=['R2']).sort_values(by='R2', ascending=False).to_string() + "\n")
    else:
        log_stream.write("[INFO] Tidak ada model yang berhasil di-fit.\n")

    log_stream.write("\n[OK] Proses fitting model selesai. Fitted models stored in 'models' dictionary.\n")

    return models, varx_fit_df

#----------DCC GARCH FITTING-----------
def fit_dcc_garch_models(log_stream, residuals_df):
    """Fits a DCC-GARCH model to a DataFrame of residuals.

    Args:
        log_stream (StringIO): Stream to write log messages.
        residuals_df (pd.DataFrame): DataFrame where each column represents the residuals
                                     of an endogenous variable from a VARX/ARX model.

    Returns:
        DCCGARCH: A fitted DCCGARCH model object.
    """
    log_stream.write("[INFO] Fitting DCC-GARCH model to residuals...")

    if residuals_df.empty:
        log_stream.write("[WARN] Residuals DataFrame is empty. Skipping DCC-GARCH fitting.")
        return None

    try:
        # Ensure residuals are float type
        residuals_df = residuals_df.astype(float)

        # DCCGARCH constructor tidak menerima argumen p/q; parameter diestimasi saat fit.
        dcc_model = DCCGARCH()
        dcc_model.fit(
            eps=residuals_df.to_numpy(),
            column_names=list(residuals_df.columns),
            disp=False,
        )

        log_stream.write("[OK] DCC-GARCH model fitted successfully.")
        return dcc_model

    except Exception as e:
        log_stream.write(f"[ERROR] Failed to fit DCC-GARCH model: {e}")
        return None


# ============================================================
# 5. FORECASTING & RESTORATION
# ============================================================


def forecasting_and_restore(log_stream, log_returns_dict, models, fitted_dcc_garch_models, exog_map, cleaned_fred_data, base_data):
    """Performs forecasting using fitted models and restores forecasts to price scale."""
    log_stream.write(f"\n[INFO] Melakukan peramalan ({parameter.FORECAST_HORIZON} langkah ke depan) dengan interval kepercayaan...\n") # Use parameter.FORECAST_HORIZON

    if not log_returns_dict or not models:
        log_stream.write("[WARN] Data log return atau fitted VARX/ARX models kosong. Melewati peramalan.\n")
        combined_forecasts_with_intervals = {}
    else:
        combined_forecasts_with_intervals = safe_run("Generate Combined Forecasts", log_stream, auto_varx_forecast,
                                                    fitted_models=models, # Corrected: Use 'fitted_models'
                                                    combined_log_returns_dict=log_returns_dict, # Needs to be dict by TF
                                                    final_stationarized_fred_data=cleaned_fred_data, # Only relevant for D1
                                                    significant_pair_exog_map=exog_map, # Renamed exog_map to significant_pair_exog_map
                                                    forecast_horizon=parameter.FORECAST_HORIZON,
                                                    verbose=True)

    log_stream.write("\n[OK] Peramalan gabungan selesai. Hasil disimpan dalam dictionary 'combined_forecasts_with_intervals'.\n")

    log_stream.write("\n[INFO] Hasil Peramalan (Log Return dengan Interval Kepercayaan):\n")
    if combined_forecasts_with_intervals:
        for model_key, forecast_data in combined_forecasts_with_intervals.items():
            if 'interval_forecast' in forecast_data and not forecast_data['interval_forecast'].empty:
                log_stream.write(f"\n--- Peramalan untuk: {model_key} ---\n")
                log_stream.write(forecast_data['interval_forecast'].head().to_string() + "\n")
            else:
                log_stream.write(f"\n--- Peramalan untuk: {model_key} ---\n (DataFrame kosong)\n")
    else:
        log_stream.write("Tidak ada peramalan log return gabungan yang dihasilkan.\n")


    log_stream.write("\n[INFO] Merestorasi peramalan log return ke peramalan harga (OHLC) dengan interval kepercayaan...\n")

    if not combined_forecasts_with_intervals or not base_data:
        log_stream.write("[WARN] Data peramalan gabungan atau data harga base (base_dfs) tidak lengkap. Melewati restorasi peramalan harga.\n")
        restored_price_forecasts_with_intervals = {}
    else:
        restored_price_forecasts_with_intervals = safe_run("Restore Price Forecast with Intervals", log_stream, restore_log_returns_to_price,
                                                            combined_forecasts_with_intervals, base_data, confidence_level=parameter.CONFIDENCE_LEVEL)


    log_stream.write("\n[OK] Restorasi peramalan harga dengan interval kepercayaan selesai. Hasil disimpan dalam dictionary 'restored_price_forecasts_with_intervals'.\n")

    log_stream.write("\n[INFO] Hasil Peramalan Harga (Direstorasi - OHLC dengan Interval Kepercayaan):\n")
    if restored_price_forecasts_with_intervals:
        for pair_name, forecast_df in restored_price_forecasts_with_intervals.items():
            log_stream.write(f"\n--- Restorasi Harga Peramalan untuk Pair: {pair_name} (OHLC) ---\n")
            if not forecast_df.empty:
                log_stream.write(forecast_df.to_string() + "\n")
            else:
                log_stream.write("DataFrame peramalan harga kosong.\n")
    else:
        log_stream.write("Tidak ada peramalan harga OHLC yang berhasil direstorasi.\n")

    return combined_forecasts_with_intervals, restored_price_forecasts_with_intervals

def save_pipeline_outputs_to_file(filepath, execution_log, model_summary_df, combined_log_returns_forecasts, restored_price_forecasts, last_actual_prices_dict):
    """Saves all important pipeline outputs to a specified file."""
    with open(filepath, 'w') as f:
        f.write("==== Pipeline Execution Log ====\n")
        f.write(execution_log if execution_log is not None else "Execution log not available.\n")
        f.write("\n\n")

        f.write("==== Model Summary ====\n")
        if model_summary_df is not None and not model_summary_df.empty:
            f.write(model_summary_df.to_string())
        else:
            f.write("No model summary available.\n")
        f.write("\n\n")

        f.write("==== Combined Log Returns Forecasts ====\n")
        if combined_log_returns_forecasts:
            for model_key, forecast_data in combined_log_returns_forecasts.items():
                f.write(f"--- Forecast for: {model_key} ---\n")
                if 'interval_forecast' in forecast_data and not forecast_data['interval_forecast'].empty:
                    f.write(forecast_data['interval_forecast'].to_string())
                else:
                    f.write("Empty forecast DataFrame.\n")
                f.write("\n")
        else:
            f.write("No combined log returns forecasts generated.\n")
        f.write("\n\n")

        f.write("==== Restored Price Forecasts (OHLC) ====\n")
        if restored_price_forecasts:
            for pair_name, forecast_df in restored_price_forecasts.items():
                f.write(f"--- Restored Price Forecast for Pair: {pair_name} (OHLC) ---\n")
                if not forecast_df.empty:
                    f.write(forecast_df.to_string())
                else:
                    f.write("Empty restored price forecast DataFrame.\n")
                f.write("\n")
        else:
            f.write("No restored price forecasts generated.\n")
        f.write("\n\n")

        f.write("==== Last Actual Prices for Restoration ====\n")
        if last_actual_prices_dict:
            f.write(json.dumps(last_actual_prices_dict, indent=2))
        else:
            f.write("No last actual prices recorded.\n")

    print(f"Output saved to {filepath}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"pipeline_run_{timestamp}"
    log_stream = StringIO()
    log_stream.write(f"[INFO] Starting pipeline run with ID: {run_id}\n")

    # === 1. UNDUH DATA FRED TERLEBIH DAHULU ===
    # Removed local load_fred_data wrapper. Directly call download_macro_data.
    if parameter.FRED_API_KEY == 'YOUR_FRED_API_KEY': # Check for placeholder
        log_stream.write("[WARN] FRED API Key is a placeholder. FRED data download will likely fail.\n")
        fred_df, fred_meta = None, None # Force None if key is invalid
    else:
        fred_data_results = safe_run("Unduh FRED", log_stream, download_macro_data, parameter.FRED_API_KEY, parameter.FRED_SERIES, parameter.fred_lookback_days)
        fred_df, fred_meta = fred_data_results if fred_data_results else (None, None)


    # === 2. LOAD DATA BASE MTF ===
    base_dir = getattr(parameter, 'BASE_DATA_DIR', '/content/base_data')
    msg = f"[DEBUG] Menyiapkan pemuatan data. Base Dir: {base_dir}\n"
    log_stream.write(msg); print(msg, end="")

    # === 1. TRY LOAD FROM PARQUET (LAZY) ===
    mtf_base_dfs = _load_parquet_lazy(base_dir, parameter.ASSET_REGISTRY)

    if mtf_base_dfs:
        total_assets = sum(len(dfs) for dfs in mtf_base_dfs.values() if isinstance(dfs, dict))
        msg = f"[OK] mtf_base_dfs dimuat dari parquet {base_dir}. Total TF: {len(mtf_base_dfs)}, Est. Total Assets: {total_assets}\n"
        log_stream.write(msg); print(msg, end="")
    else:
        msg = f"[INFO] Parquet tidak ditemukan atau kosong di {base_dir}. Mencoba fallback ke Pickle...\n"
        log_stream.write(msg); print(msg, end="")

        # === 2. FALLBACK TO PICKLE ===
        cache_dir = getattr(parameter, 'PKL_CACHE_DIR', '/content/.pkl')
        mtf_pkl_path = os.path.join(cache_dir, getattr(parameter, 'MTF_BASE_DFS_PKL_NAME', 'mtf_base_dfs.pkl'))
        mtf_base_dfs = _load_pickle_if_exists(log_stream, mtf_pkl_path, 'mtf_base_dfs')

        if mtf_base_dfs:
            msg = f"[OK] mtf_base_dfs dimuat dari pickle: {mtf_pkl_path}\n"
            log_stream.write(msg); print(msg, end="")

    if not isinstance(mtf_base_dfs, dict) or not mtf_base_dfs:
        mtf_base_dfs = {}
        for tf in parameter.MTF_INTERVALS.keys(): # Iterate over keys D1, H1, M1
            interval_str = parameter.MTF_INTERVALS[tf] # Interval string untuk loader sumber data saat ini
            lookback_days_for_tf = parameter.LOOKBACK_DAYS[tf] # Get the lookback days for this TF
            log_stream.write(f"[DEBUG] Loading data for TF: {tf}, interval_str: {interval_str}, lookback_days: {lookback_days_for_tf}\n")
            mtf_base_dfs[tf] = safe_run(f"Load Data {tf}", log_stream, load_base_data_mtf,
                                         parameter.PAIRS, lookback_days_for_tf, interval_str,
                                         parameter.USE_LOCAL_CSV_FOR_PAIRS,
                                         parameter.LOCAL_CSV_FILEPATH)
        _save_parquet(log_stream, mtf_base_dfs, base_dir, 'mtf_base_dfs', parameter.ASSET_REGISTRY)

    mtf_imputation_assets = {}
    for tf in parameter.MTF_INTERVALS.keys():
        interval_str = parameter.MTF_INTERVALS[tf]
        lookback_days_for_tf = parameter.LOOKBACK_DAYS[tf]
        tf_pairs = mtf_base_dfs.get(tf, {}) if isinstance(mtf_base_dfs, dict) else {}
        special_assets = safe_run(
            f"Load Imputation Assets {tf}",
            log_stream,
            download_imputation_special_assets,
            parameter.IMPUTATION_SPECIAL_ASSETS,
            lookback_days_for_tf,
            interval_str,
            tf_pairs,
        )
        mtf_imputation_assets[tf] = special_assets or {}

    # Freshness check hanya untuk monitoring, tidak menghentikan proses training
    safe_run("Cek Data Freshness", log_stream, check_data_freshness,
             mtf_base_dfs, datetime.now(timezone.utc))

    # Selaraskan seluruh timeframe ke common close agar horizon observasi konsisten
    mtf_base_dfs = safe_run("Align MTF to Common Close", log_stream, align_mtf_data_to_common_close,
                            mtf_base_dfs) or mtf_base_dfs

    interactive_review = getattr(parameter, 'ENABLE_INTERACTIVE_PREPROCESS_REVIEW', False)
    mtf_base_dfs = safe_run(
        "Review Data MTF",
        log_stream,
        review_and_confirm_mtf_data,
        mtf_base_dfs,
        fred_df,
        interactive=interactive_review,
        imputation_assets_by_tf=mtf_imputation_assets,
    ) or mtf_base_dfs
    if fred_df is not None:
        reviewed_fred_df = safe_run(
            "Review Data FRED",
            log_stream,
            review_and_confirm_fred_data,
            fred_df,
            mtf_base_dfs,
            interactive=interactive_review,
        )
        if reviewed_fred_df is not None:
            fred_df = reviewed_fred_df

    # === 3. PREPROCESSING MTF ===
    mtf_log_returns = {}
    mtf_cleaned_fred = {} # Biasanya FRED hanya diproses sekali di D1
    mtf_exog_pool = {} # This will hold combined log returns from other pairs to be used as exog
    cleaned_fred_combined_df = pd.DataFrame() # To hold the processed FRED data as a single DF

    for tf, b_dfs in mtf_base_dfs.items():
        if not b_dfs: continue
        # Call preprocess_data_tf, which should return (log_returns_dict, cleaned_fred_for_tf, combined_df)
        preprocess_result = safe_run(f"Preprocess {tf}", log_stream, preprocess_data_tf, b_dfs, fred_df, fred_meta, tf)
        if not preprocess_result or not isinstance(preprocess_result, tuple) or len(preprocess_result) != 3:
            log_stream.write(f"[WARN] Preprocess {tf} tidak mengembalikan tuple 3 elemen. Layer {tf} dilewati.\n")
            continue

        log_returns_tf, cleaned_fred_tf, combined_log_returns_tf = preprocess_result

        log_stream.write(f"[DEBUG] Inside main loop for tf={tf}:\n") # New debug

        if log_returns_tf is not None:
            mtf_log_returns[tf] = log_returns_tf
            log_stream.write(f"[DEBUG] mtf_log_returns[{tf}] populated with keys: {list(log_returns_tf.keys())}. Example columns for first pair: {list(log_returns_tf[next(iter(log_returns_tf))].columns) if log_returns_tf else 'N/A'}\n") # MODIFIED DEBUG PRINT
        else:
            log_stream.write(f"[DEBUG] log_returns_tf for {tf} was None.\n")

        if tf == 'D1' and cleaned_fred_tf is not None:
            if isinstance(cleaned_fred_tf, dict) and cleaned_fred_tf:
                # Combine all individual FRED series DataFrames (from the dict) into one DataFrame
                dfs_to_concat = []
                # Keep track of the 'effective_until_next_release' column separately
                effective_until_next_release_df = None

                for series_name, df_series in cleaned_fred_tf.items():
                    if not df_series.empty:
                        # Extract only the value columns
                        value_cols = [col for col in df_series.columns if col != 'effective_until_next_release']
                        if value_cols:
                            dfs_to_concat.append(df_series[value_cols])
                            # Collect effective_until_next_release, if present, from the first non-empty df
                            if effective_until_next_release_df is None and 'effective_until_next_release' in df_series.columns:
                                effective_until_next_release_df = df_series[['effective_until_next_release']].copy()
                        else:
                            log_stream.write(f"[WARN] No value columns found in transformed FRED series '{series_name}'. Skipping.\n")
                    else:
                        log_stream.write(f"[WARN] Transformed FRED series '{series_name}' is empty. Skipping.\n")


                if dfs_to_concat:
                    # Concatenate all value columns
                    combined_fred_values_df = pd.concat(dfs_to_concat, axis=1, join='outer')
                    combined_fred_values_df = combined_fred_values_df.ffill().dropna(how='all')

                    if effective_until_next_release_df is not None:
                        # Merge the combined values with the effective_until_next_release
                        cleaned_fred_combined_df = combined_fred_values_df.merge(
                            effective_until_next_release_df,
                            left_index=True,
                            right_index=True,
                            how='left'
                        ).ffill()
                        # Ensure effective_until_next_release is also UTC localized
                        if 'effective_until_next_release' in cleaned_fred_combined_df.columns:
                            if cleaned_fred_combined_df['effective_until_next_release'].dtype == 'datetime64[ns]':
                                cleaned_fred_combined_df['effective_until_next_release'] = cleaned_fred_combined_df['effective_until_next_release'].dt.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')
                    else:
                        cleaned_fred_combined_df = combined_fred_values_df
                        log_stream.write("[WARN] 'effective_until_next_release' column not found in any FRED series for D1 combination. Will proceed without it in cleaned_fred_combined_df.\n")

                    # Ensure the final index is UTC localized.
                    if cleaned_fred_combined_df.index.tz is None:
                        cleaned_fred_combined_df = cleaned_fred_combined_df.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT')

                    log_stream.write(f"[INFO] FRED data for D1 successfully combined into a single DataFrame. Shape: {cleaned_fred_combined_df.shape}\n")
                else:
                    log_stream.write(f"[WARN] cleaned_fred_tf for D1 was a dict but contained no valid DataFrames to combine. cleaned_fred_combined_df remains empty.\n")
            elif isinstance(cleaned_fred_tf, pd.DataFrame) and not cleaned_fred_tf.empty:
                cleaned_fred_combined_df = cleaned_fred_tf # Fallback if it somehow is already a DataFrame (though unlikely for D1)
                log_stream.write(f"[INFO] FRED data for D1 was already a DataFrame. Shape: {cleaned_fred_combined_df.shape}\n")
            else:
                log_stream.write(f"[WARN] cleaned_fred_tf for D1 is empty or not a dict/DataFrame. cleaned_fred_combined_df remains empty.\n")

        if combined_log_returns_tf is not None:
            mtf_exog_pool[tf] = combined_log_returns_tf # Store combined log returns for potential cross-asset exog

    log_stream.write(f"[DEBUG] Final mtf_log_returns keys: {mtf_log_returns.keys()}\n")

    # === 4. FITTING MULTI-TIMEFRAME ===
    ensemble_results = {}
    all_summaries = []
    mtf_exog_maps = {}

    for tf in ['D1', 'H1']:
        log_stream.write(f"\n[PROCESS] Analisis & Fitting Layer {tf}...\n")

        # Granger Test: Only FRED data should be used as exogenous for D1
        granger_result = safe_run(f"Granger {tf}", log_stream, run_granger_all,
                               mtf_log_returns.get(tf, {}), cleaned_fred_combined_df if tf == 'D1' else pd.DataFrame(), timeframe_label=tf)
        if not granger_result or not isinstance(granger_result, tuple) or len(granger_result) != 2:
            log_stream.write(f"[WARN] Granger {tf} tidak mengembalikan tuple 2 elemen. Eksogen dianggap kosong.\n")
            granger_results, exog_map_tf = pd.DataFrame(), {}
        else:
            granger_results, exog_map_tf = granger_result
        mtf_exog_maps[tf] = exog_map_tf # Store the exog_map for the current timeframe

        # Prepare combined exogenous pool for model fitting for this TF
        import polars as pl

        lazy_frames = []
        seen_columns = set()

        if tf == 'D1' and isinstance(cleaned_fred_combined_df, pd.DataFrame) and not cleaned_fred_combined_df.empty:
            fred_frame = cleaned_fred_combined_df.ffill()
            fred_cols = [col for col in fred_frame.columns if col not in seen_columns]
            if fred_cols:
                seen_columns.update(fred_cols)
                lazy_frames.append(pl.from_pandas(fred_frame[fred_cols]).lazy())

        for other_tf, other_log_returns_df in mtf_exog_pool.items():
            # Example: H1 can use D1 log returns as exog, M1 can use H1 as exog
            # This logic needs to be refined based on actual cross-timeframe exog strategy
            if isinstance(other_log_returns_df, pd.DataFrame) and not other_log_returns_df.empty:
                usable_cols = [col for col in other_log_returns_df.columns if col not in seen_columns]
                if usable_cols:
                    seen_columns.update(usable_cols)
                    lazy_frames.append(pl.from_pandas(other_log_returns_df[usable_cols]).lazy())

        if lazy_frames:
            combined_lazy = pl.concat(lazy_frames, how="horizontal")
            combined_lazy = combined_lazy.unique()
            current_tf_exog_pool = combined_lazy.collect().to_pandas()
            current_tf_exog_pool = current_tf_exog_pool.ffill().dropna(how='all')
        else:
            current_tf_exog_pool = pd.DataFrame()

        # Fit VARX/ARX models
        models_tf, summary_tf = fit_models(
            log_stream,
            mtf_log_returns.get(tf, {}), # Log returns for the current TF
            mtf_exog_maps[tf],             # Exog map from Granger for current TF
            current_tf_exog_pool,        # Combined pool of all potential exogs
            timeframe_label=tf
        )
        ensemble_results[tf] = models_tf
        if summary_tf is not None and not summary_tf.empty:
            summary_tf['TF'] = tf # Tandai timeframe di summary
            all_summaries.append(summary_tf)

    # 5. Setup Lapisan M1 (Kalman)
    if 'M1' in mtf_base_dfs: # Kalman filter is applied directly to M1 raw data
        m1_pairs = mtf_base_dfs.get('M1', {})
        m1_df = next((df for df in m1_pairs.values() if isinstance(df, pd.DataFrame) and not df.empty), None)
        kalman_models_m1 = safe_run("Setup Kalman Filter for M1", log_stream, setup_kalman_filter_compat, m1_df) if m1_df is not None else None
        if kalman_models_m1 is not None:
            ensemble_results['M1'] = kalman_models_m1

    # === 6. VOLATILITY (H1) & SAVE ===
    # DCC-GARCH models will be fitted using residuals from VARX models
    fitted_dcc_garch_models = {}
    if 'H1' in ensemble_results and mtf_log_returns.get('H1'):
        # First, prepare the residuals using the dedicated helper function
        h1_residuals_df = safe_run("Prepare H1 Residuals for DCC-GARCH", log_stream,
                                   prepare_residuals_for_dcc_garch,
                                   ensemble_results,     # Kirim dict model
                                   mtf_log_returns)  # Kirim data return

        if h1_residuals_df is not None and not h1_residuals_df.empty:
            # Then, fit the DCC-GARCH model with the prepared residuals
            fitted_dcc_garch_models_h1 = safe_run("Fit DCC-GARCH for H1", log_stream,
                                                  fit_dcc_garch_models,
                                                  h1_residuals_df)
            if fitted_dcc_garch_models_h1: # Menyimpan objek model DCC-GARCH untuk H1
                fitted_dcc_garch_models['H1'] = fitted_dcc_garch_models_h1
        else:
            log_stream.write("[WARN] H1 residuals not available or empty for DCC-GARCH fitting.\n")
    else:
        log_stream.write("[WARN] H1 VARX models or log returns not available for DCC-GARCH fitting.\n")

    # Ambil harga terakhir dari M1 untuk VPS
    last_actual_prices_dict = {}
    if 'M1' in mtf_base_dfs:
        for p, df in mtf_base_dfs['M1'].items():
            if not df.empty:
                last_actual_prices_dict[p] = {
                    'timestamp': df.index[-1].isoformat(),
                    'open': df['Open'].iloc[-1],
                    'high': df['High'].iloc[-1],
                    'low': df['Low'].iloc[-1],
                    'close': df['Close'].iloc[-1]
                }


    # === 7. PACKAGING UNTUK VPS ===
    try:
        package_to_save = {
            "run_id": run_id,
            "ensemble": ensemble_results, # Dictionary D1, H1, M1 models
            "dcc_garch": fitted_dcc_garch_models, # H1 DCC-GARCH models
            "last_prices": last_actual_prices_dict, # Last M1 prices
            "exog_maps": mtf_exog_maps, # Exog maps by TF
            "timestamp": timestamp
        }
        output_dir = os.path.join(ROOT_DIR, 'vps_sync')
        os.makedirs(output_dir, exist_ok=True)
        fitted_models_path = os.path.join(output_dir, 'fitted_ensemble.pkl')

        with open(fitted_models_path, 'wb') as f:
            pickle.dump(package_to_save, f);
        log_stream.write(f"[OK] Multi-Timeframe Ensemble saved to {fitted_models_path}\n")
    except Exception as e:
        log_stream.write(f"[ERROR] Save to PKL failed: {e}\n")

    # Combine Summaries
    final_summary = pd.concat(all_summaries) if all_summaries else pd.DataFrame()

    # === 8. FORECASTING & RESTORATION (FOR COLAB DISPLAY ONLY) ===
    # This part needs to be adapted to forecast for each timeframe and restore prices appropriately
    # Currently, it passes a single combined_forecasts_with_intervals and base_data (likely only D1 or M1)
    combined_forecasts_with_intervals_output = {}
    restored_price_forecasts_with_intervals = {}

    # Need to iterate through each TF that has models and log_returns
    for tf_forecast in ['D1', 'H1']:
        if tf_forecast in ensemble_results and mtf_log_returns.get(tf_forecast):
            # Auto_varx_forecast likely needs to be updated to handle MTF structure
            # and potentially DCC-GARCH output for interval forecasts.
            # For simplicity here, calling forecasting_and_restore for each TF
            # This function needs significant refactoring in `main.py` to be truly MTF aware.
            tf_forecasts, tf_restored_prices = forecasting_and_restore(
                log_stream,
                mtf_log_returns.get(tf_forecast),
                ensemble_results.get(tf_forecast),
                fitted_dcc_garch_models.get(tf_forecast), # Pass relevant DCC-GARCH for this TF
                mtf_exog_maps.get(tf_forecast),
                cleaned_fred_combined_df, # FRED data is D1 only, pass for D1 forecast
                mtf_base_dfs.get(tf_forecast) # Base data for restoration
            )
            if tf_forecasts: combined_forecasts_with_intervals_output[tf_forecast] = tf_forecasts
            if tf_restored_prices: restored_price_forecasts_with_intervals[tf_forecast] = tf_restored_prices


    # Final return values
    return (run_id, mtf_base_dfs, fred_df, fred_meta, mtf_log_returns, cleaned_fred_combined_df,
            granger_results, mtf_exog_maps, current_tf_exog_pool, ensemble_results, # current_tf_exog_pool is misleading here, should be per TF
            fitted_dcc_garch_models, combined_forecasts_with_intervals_output,
            restored_price_forecasts_with_intervals, log_stream.getvalue(), final_summary, last_actual_prices_dict)

if __name__ == "__main__":
    # To make it available in the interactive environment without explicitly calling 'global'
    # anywhere, we can execute the main function and then assign the results to global variables.
    pass
