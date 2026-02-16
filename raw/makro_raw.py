import pandas as pd
from fredapi import Fred
from datetime import datetime, timedelta
import numpy as np
#import parameters #(assuming parameter.py is executed or imported)
from parameter import FRED_API_KEY, FRED_SERIES

def download_macro_data(log_stream, FRED_API_KEY, FRED_SERIES, lookback_days): # ADDED log_stream
    """
    Downloads macro economic data from FRED API.

    Args:
        log_stream (StringIO): Stream to write log messages.
        FRED_API_KEY (str): Your FRED API key.
        FRED_SERIES (dict): A dictionary where keys are desired names for the series
                            and values are the FRED series IDs.
        lookback_days (int): Number of days to look back from the current date.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with combined FRED data, indexed by release date.
                            Columns are the keys from FRED_SERIES.
            - list: A list of dictionaries containing metadata for each downloaded series.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    log_stream.write(f"[INFO] Mengambil data dari {start_date.date()} hingga {end_date.date()}\n") # Changed print to log_stream.write

    log_stream.write(f"DEBUG: Type of FRED_SERIES is {type(FRED_SERIES)}\n")
    log_stream.write(f"DEBUG: Content: {str(FRED_SERIES)[:100]}\n")

    # Jika FRED_SERIES adalah list, konversi menjadi dict untuk menghindari error .items()
    if isinstance(FRED_SERIES, list):
        log_stream.write("[WARN] FRED_SERIES terdeteksi sebagai LIST. Mengonversi ke dictionary...\n")
        series_dict = {item: item for item in FRED_SERIES}
    elif isinstance(FRED_SERIES, dict):
        series_dict = FRED_SERIES
    else:
        log_stream.write(f"[ERROR] Tipe data FRED_SERIES tidak dikenal: {type(FRED_SERIES)}\n")
        return pd.DataFrame(), []

    try:
        fred = Fred(api_key=FRED_API_KEY)
    except Exception as e:
        log_stream.write(f"[ERROR] Gagal inisialisasi FRED API: {e}\n")
        return pd.DataFrame(), []

    #fred = Fred(api_key=FRED_API_KEY)
    fred_data = []
    fred_metadata = []

    for name, series_id in series_dict.items():
        try:
            # 1′ Ambil metadata
            info = fred.get_series_info(series_id)
            meta = {
                "series_name": name,
                "series_id": series_id,
                "title": getattr(info, "title", None),
                "frequency": getattr(info, "frequency", None),
                "units": getattr(info, "units", None),
                "seasonal_adjustment": getattr(info, "seasonal_adjustment", None),
                "last_updated": getattr(info, "last_updated", None),
                "notes": getattr(info, "notes", None),
            }
            fred_metadata.append(meta)

            # 2′ Ambil data (utamakan realtime, fallback ke simple)
            try:
                df = fred.get_series_all_releases(series_id)
                mode = "all_releases"
            except Exception as e:
                log_stream.write(f"[WARN] Fallback {name} ({series_id}) ke simple mode: {e}\n") # Changed print to log_stream.write
                df = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                mode = "simple_series"

            if df is None or len(df) == 0:
                log_stream.write(f"[WARN] Tidak ada data untuk {name} ({series_id})\n") # Changed print to log_stream.write
                continue

            # 3′ Normalisasi format DataFrame
            if isinstance(df, pd.Series):
                # Simple mode
                df = df.to_frame(name='value')
                df['release_date'] = pd.to_datetime(df.index)
                df['effective_until_next_release'] = pd.NaT
                df['date'] = df['release_date'] # Use release_date as date for simple mode
            else:
                # Realtime mode
                # Use 'realtime_start' as release_date and 'date' as the observed date
                if 'realtime_start' in df.columns:
                     df['release_date'] = pd.to_datetime(df['realtime_start'], errors='coerce')
                else:
                     # Fallback for older pandas/fredapi versions or different ALFRED format
                     df['release_date'] = pd.to_datetime(df['date'], errors='coerce') # Assuming 'date' exists in realtime mode


                if 'realtime_end' in df.columns:
                    df['effective_until_next_release'] = pd.to_datetime(df['realtime_end'], errors='coerce')
                else:
                    # Calculate effective_until_next_release if not provided
                    df = df.sort_values('release_date')
                    df['effective_until_next_release'] = df['release_date'].shift(-1)


                if 'date' not in df.columns:
                     # Attempt to find an observation date column if 'date' is missing
                     # This part might need more specific logic depending on ALFRED's format
                     log_stream.write(f"[WARN] 'date' column not found for {name} in realtime mode.\n") # Changed print to log_stream.write
                     df['date'] = df['release_date'] # Fallback, may not be accurate observed date


                if 'value' not in df.columns:
                    # Assuming the value is the last column if named differently
                    if len(df.columns) > 3: # Exclude standard date columns
                        value_candidate = df.columns[-1]
                        if not value_candidate in ['release_date', 'effective_until_next_release', 'date']:
                             df.rename(columns={value_candidate: 'value'}, inplace=True)
                        else:
                             # If the last column is also a date column, try the second to last
                             value_candidate = df.columns[-2]
                             if not value_candidate in ['release_date', 'effective_until_next_release', 'date']:
                                  df.rename(columns={value_candidate: 'value'}, inplace=True)
                             else:
                                 log_stream.write(f"[WARN] Could not identify value column for {name} in realtime mode.\n") # Changed print to log_stream.write
                                 # Attempt to find a numeric column that is not a date
                                 numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                                 numeric_cols = [c for c in numeric_cols if c not in ['release_date', 'effective_until_next_release', 'date']]
                                 if numeric_cols:
                                     df.rename(columns={numeric_cols[0]: 'value'}, inplace=True)
                                     log_stream.write(f"[INFO] Using first numeric column '{numeric_cols[0]}' as value for {name}.\n") # Changed print to log_stream.write
                                 else:
                                     log_stream.write(f"[ERROR] No numeric value column found for {name}. Skipping.\n") # Changed print to log_stream.write
                                     continue


            # 4′ Konversi tipe dan bersihkan
            for col in ['release_date', 'effective_until_next_release', 'date']:
                if col in df.columns:
                    # Use format to handle mixed date formats if necessary, or let errors='coerce' handle it
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize("UTC", nonexistent='shift_forward', ambiguous='NaT')
                else:
                     # Add the column with NaT if it doesn't exist after renaming/processing
                     df[col] = pd.NaT


            if 'value' not in df.columns:
                 log_stream.write(f"[ERROR] No 'value' column available after processing for {name}. Skipping.\n") # Changed print to log_stream.write
                 continue

            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            # 5′ Filter berdasarkan rentang waktu release_date
            df = df[(df['release_date'] >= pd.Timestamp(start_date, tz='UTC')) &
                    (df['release_date'] <= pd.Timestamp(end_date, tz='UTC'))]

            df = df.dropna(subset=['value']).sort_values(['release_date'])
            # Keep the last release for a given date if there are multiple releases on the same day
            df = df.drop_duplicates(subset=['release_date'], keep='last')
            df = df.rename(columns={'value': name})

            # --- Add inspection here ---
            log_stream.write(f"--- Inspection after download and filter for {name} ({series_id}) ---\n") # Changed print to log_stream.write
            log_stream.write(f"DataFrame shape: {df.shape}\n") # Changed print to log_stream.write
            log_stream.write(f"DataFrame columns: {list(df.columns)}\n") # Changed print to log_stream.write
            if not df.empty:
                index_tz = df.index.tz if isinstance(df.index, pd.DatetimeIndex) else 'N/A'
                log_stream.write(f"Index dtype: {df.index.dtype}, tz: {index_tz}\n")
                release_tz = df['release_date'].dt.tz if hasattr(df['release_date'], 'dt') else 'N/A'
                log_stream.write(f"release_date dtype: {df['release_date'].dtype}, tz: {release_tz}\n")
                if 'effective_until_next_release' in df.columns:
                     eff_tz = df['effective_until_next_release'].dt.tz if hasattr(df['effective_until_next_release'], 'dt') else 'N/A'
                     log_stream.write(f"effective_until_next_release dtype: {df['effective_until_next_release'].dtype}, tz: {eff_tz}\n")
                if 'date' in df.columns:
                     date_tz = df['date'].dt.tz if hasattr(df['date'], 'dt') else 'N/A'
                     log_stream.write(f"date dtype: {df['date'].dtype}, tz: {date_tz}\n")
            else:
                log_stream.write("DataFrame is empty.\n") # Changed print to log_stream.write
            log_stream.write("---------------------------------------------------------\n") # Changed print to log_stream.write
            # --- End inspection ---


            fred_data.append(df[['release_date', 'effective_until_next_release', 'date', name]])
            log_stream.write(f"[OK] {name}: {len(df)} observasi ({mode}), freq={meta['frequency']}\n") # Changed print to log_stream.write

        except Exception as e:
            log_stream.write(f"[WARN] Gagal ambil {name} ({series_id}): {e}\n") # Changed print to log_stream.write

    # (fred_df)
    # ============================================================
    # ↔ Gabungkan semua seri berdasarkan tanggal rilis (versi kausal & aman)
    # ============================================================

    if fred_data and isinstance(fred_data, list) and len(fred_data) > 0:
        clean_data = []

        for i, df in enumerate(fred_data):
            if not isinstance(df, pd.DataFrame) or df.empty:
                log_stream.write(f"[WARN] Seri ke-{i} kosong atau bukan DataFrame -- dilewati.\n") # Changed print to log_stream.write
                continue

            # Pastikan tidak ada duplikasi kolom
            df = df.loc[:, ~df.columns.duplicated()].copy()

            # Pastikan kolom release_date ada
            if "release_date" not in df.columns:
                if "date" in df.columns:
                    df = df.rename(columns={"date": "release_date"})
                else:
                    log_stream.write(f"[WARN] Seri ke-{i} tidak memiliki kolom tanggal -- dilewati.\n") # Changed print to log_stream.write
                    continue

            # Tambahkan kolom batas efektif
            if "effective_until_next_release" not in df.columns:
                df = df.sort_values("release_date")
                df["effective_until_next_release"] = df["release_date"].shift(-1)
                df["effective_until_next_release"].fillna(
                    df["release_date"].max() + pd.Timedelta(days=7), inplace=True
                )

            # Pilih hanya kolom penting: release_date, effective_until_next_release, dan value
            value_cols = [c for c in df.columns if c not in ["release_date", "effective_until_next_release", "date"]]
            if len(value_cols) == 0:
                log_stream.write(f"[WARN] Seri ke-{i} tidak memiliki kolom nilai -- dilewati.\n") # Changed print to log_stream.write
                continue

            value_col = value_cols[0]
            df = df[["release_date", "effective_until_next_release", value_col]]
            clean_data.append(df.rename(columns={value_col: f"series_{i}"}))

        if not clean_data:
            fred_df = pd.DataFrame()
            log_stream.write("[WARN] Tidak ada seri valid untuk digabungkan.\n") # Changed print to log_stream.write
        else:
            # Mulai dengan seri pertama
            fred_df = clean_data[0].copy()

            # Gabungkan seri lainnya secara asof (berdasarkan release_date)
            for subdf in clean_data[1:]:
                fred_df = pd.merge_asof(
                    fred_df.sort_values("release_date"),
                    subdf.sort_values("release_date"),
                    on="release_date",
                    direction="backward",
                    suffixes=("", "_dup")
                )
                # Hapus duplikat kolom hasil merge
                dup_cols = [c for c in fred_df.columns if c.endswith("_dup")]
                if dup_cols:
                    fred_df.drop(columns=dup_cols, inplace=True)

            # Pastikan tidak ada kolom duplikat
            fred_df = fred_df.loc[:, ~fred_df.columns.duplicated()]

            # Sortir dan reset index
            fred_df = fred_df.sort_values("release_date").reset_index(drop=True)

            # Pastikan timezone aman
            fred_df["release_date"] = pd.to_datetime(fred_df["release_date"], errors="coerce")
            fred_df["effective_until_next_release"] = pd.to_datetime(
                fred_df["effective_until_next_release"], errors="coerce"
            )

            # Rapikan nilai NaN pada series (jika ada, isi dengan last known)
            value_cols = [c for c in fred_df.columns if c.startswith("series_")]
            fred_df[value_cols] = fred_df[value_cols].ffill()

            log_stream.write(f"\n[INFO] Total gabungan FRED data: {fred_df.shape}\n") # Changed print to log_stream.write
            log_stream.write(f"[INFO] Periode: {fred_df['release_date'].min().date()} -> {fred_df['release_date'].max().date()}\n") # Changed print to log_stream.write
            log_stream.write(f"[INFO] Kolom efektif: release_date, effective_until_next_release ({len(value_cols)} seri)\n") # Changed print to log_stream.write

    else:
        fred_df = pd.DataFrame()
        log_stream.write("[WARN] Tidak ada data FRED yang valid dari langkah pengunduhan.\n") # Changed print to log_stream.write

    # --- Pastikan index fred_df adalah datetime index valid ---
    if not isinstance(fred_df.index, pd.DatetimeIndex):
        # Coba cari kolom tanggal rilis dari FRED API
        if "release_date" in fred_df.columns:
            fred_df["release_date"] = pd.to_datetime(fred_df["release_date"], errors="coerce", utc=True)
            fred_df.set_index("release_date", inplace=True)
        elif "date" in fred_df.columns:
            fred_df["date"] = pd.to_datetime(fred_df["date"], errors="coerce", utc=True)
            fred_df.set_index("date", inplace=True)
        else:
            # fallback jika tidak ada kolom tanggal
            fred_df.index = pd.to_datetime(fred_df.index, errors="coerce", utc=True)

    # Drop baris invalid (NaT)
    fred_df = fred_df[~fred_df.index.isna()]

    # Pastikan timezone UTC
    if fred_df.index.tz is None:
        fred_df = fred_df.tz_localize("UTC", nonexistent='shift_forward', ambiguous='NaT')

    # --- Rename kolom FRED agar sesuai ID-nya ---
    # Find the columns that are value columns (likely starting with 'series_')
    value_cols_to_rename = [col for col in fred_df.columns if col.startswith('series_')]
    # Ensure FRED_SERIES is accessible
    # Assuming FRED_SERIES is defined in the notebook's global scope after running parameter.py
    if 'FRED_SERIES' in globals():
         fred_series_keys = list(FRED_SERIES.keys())
         if len(value_cols_to_rename) == len(fred_series_keys):
             # Create a renaming map from the current series names to the FRED_SERIES keys
             rename_map = {value_cols_to_rename[i]: fred_series_keys[i] for i in range(len(fred_series_keys))}
             fred_df.rename(columns=rename_map, inplace=True)
             log_stream.write("[OK] Kolom FRED berhasil di-rename.\n") # Changed print to log_stream.write
         elif len(value_cols_to_rename) < len(fred_series_keys):
             log_stream.write(f"[WARN] Jumlah kolom FRED yang terdeteksi ({len(value_cols_to_rename)}) lebih sedikit dari jumlah series di FRED_SERIES ({len(fred_series_keys)}). Cek urutan merge atau data.\n") # Changed print to log_stream.write
             # Attempt to rename based on the detected value columns
             rename_map = {value_cols_to_rename[i]: fred_series_keys[i] for i in range(len(value_cols_to_rename))}
             fred_df.rename(columns=rename_map, inplace=True)
             log_stream.write(f"[INFO] Kolom yang terdeteksi ({value_cols_to_rename}) telah di-rename.\n") # Changed print to log_stream.write

         else:
              log_stream.write(f"[WARN] Jumlah kolom FRED yang terdeteksi ({len(value_cols_to_rename)}) lebih banyak dari jumlah series di FRED_SERIES ({len(fred_series_keys)}). Cek urutan merge atau data.\n") # Changed print to log_stream.write
    else:
        log_stream.write("[WARN] Variabel FRED_SERIES tidak ditemukan di global scope. Tidak dapat melakukan rename kolom FRED secara otomatis.\n") # Changed print to log_stream.write


    log_stream.write("\n[INFO] FRED data final setelah koreksi index dan rename kolom:\n") # Changed print to log_stream.write

    return fred_df, fred_metadata
