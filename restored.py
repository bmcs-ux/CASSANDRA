"""Utilities untuk merestorasi hasil forecast log-return menjadi harga."""


def restore_log_returns_to_price(log_stream, forecast_data_dict, base_dfs, confidence_level=0.95):
    """
    Mengembalikan hasil peramalan log-return (termasuk interval kepercayaan) menjadi
    peramalan harga (OHLC) dalam skala asli, termasuk batas atas dan bawah harga.

    Args:
        log_stream (StringIO): Stream to write log messages.
        forecast_data_dict (dict): Dictionary dengan key model group name (e.g., 'METALS_CLOSE')
                                   dan value berupa dictionary yang berisi:
                                   - 'endog_names': List nama variabel endogen (mis. ['GBPUSD_Close_Log_Return']).
                                   - 'interval_forecast': DataFrame output forecast log-return. Format didukung:
                                     1) summary_frame SARIMAX: ['mean', 'mean_ci_lower', 'mean_ci_upper']
                                     2) summary_frame VARMAX: ['0_mean', '0_mean_ci_lower', '0_mean_ci_upper', ...]
                                     3) custom rename: ['<endog>_Forecast', '<endog>_Lower', '<endog>_Upper']
                                     4) format lama: ['<endog>_Mean', '<endog>_Lower_95CI', '<endog>_Upper_95CI']
        base_dfs (dict): Dictionary dengan key pair (mis. 'XAUUSD') dan value DataFrame harga asli (OHLC).
        confidence_level (float): Tingkat kepercayaan yang digunakan untuk interval (mis. 0.95).

    Returns:
        dict: Dictionary berisi DataFrame harga hasil restorasi untuk tiap pair.
              Kolom berformat '<OHLC>_Mean_Forecast', '<OHLC>_Lower_95CI', '<OHLC>_Upper_95CI'.
    """
    import numpy as np
    import pandas as pd
    import re

    log_stream.write(f"  [INFO] Mengembalikan peramalan log-return ke peramalan harga (OHLC) dengan interval {int(confidence_level*100)}%...\n")
    restored_forecasts = {}

    if not forecast_data_dict or not base_dfs:
        log_stream.write("  [WARN] forecast_data_dict atau base_dfs kosong. Tidak ada peramalan yang direstorasi.\n")
        return restored_forecasts

    # Map hasil parsing interval forecast menjadi:
    # {'PAIR': {'Close': {'mean': Series, 'lower_ci': Series, 'upper_ci': Series}, ...}, ...}
    pair_interval_forecasts_map = {}

    ci_lower_col_suffix = f'_Lower_{int(confidence_level*100)}CI'
    ci_upper_col_suffix = f'_Upper_{int(confidence_level*100)}CI'
    def _parse_endog_name(endog_name):
        """Parse endog name seperti GBPUSD_Close_Log_Return menjadi tuple (pair, price_type)."""
        match = re.match(r'(.+?)_(Open|High|Low|Close)_Log_Return$', endog_name)
        if not match:
            return None, None
        return match.group(1), match.group(2)

    def _extract_endog_intervals(interval_df, endog_names):
        """Normalisasi berbagai format output forecast menjadi map by endog_name."""
        endog_interval_map = {}
        if interval_df is None or interval_df.empty:
            return endog_interval_map

        # Format umum SARIMAX summary_frame: mean, mean_ci_lower, mean_ci_upper
        if {'mean', 'mean_ci_lower', 'mean_ci_upper'}.issubset(interval_df.columns):
            if not endog_names:
                return endog_interval_map
            endog_interval_map[endog_names[0]] = {
                'mean': interval_df['mean'],
                'lower_ci': interval_df['mean_ci_lower'],
                'upper_ci': interval_df['mean_ci_upper'],
            }
            return endog_interval_map

        # Format VARMAX summary_frame: 0_mean, 0_mean_ci_lower, 0_mean_ci_upper
        for i, endog_name in enumerate(endog_names or []):
            mean_col = f'{i}_mean'
            lower_col = f'{i}_mean_ci_lower'
            upper_col = f'{i}_mean_ci_upper'
            if {mean_col, lower_col, upper_col}.issubset(interval_df.columns):
                endog_interval_map[endog_name] = {
                    'mean': interval_df[mean_col],
                    'lower_ci': interval_df[lower_col],
                    'upper_ci': interval_df[upper_col],
                }

        # Format hasil rename custom: <endog>_Forecast, <endog>_Lower, <endog>_Upper
        for endog_name in endog_names or []:
            mean_col = f'{endog_name}_Forecast'
            lower_col = f'{endog_name}_Lower'
            upper_col = f'{endog_name}_Upper'
            if {mean_col, lower_col, upper_col}.issubset(interval_df.columns):
                endog_interval_map[endog_name] = {
                    'mean': interval_df[mean_col],
                    'lower_ci': interval_df[lower_col],
                    'upper_ci': interval_df[upper_col],
                }

        # Format lama (backward compatible): <endog>_Mean, <endog>_Lower_95CI, <endog>_Upper_95CI
        for col_name in interval_df.columns:
            if not col_name.endswith('_Mean'):
                continue
            endog_name = col_name[:-5]
            lower_col = f'{endog_name}{ci_lower_col_suffix}'
            upper_col = f'{endog_name}{ci_upper_col_suffix}'
            if {col_name, lower_col, upper_col}.issubset(interval_df.columns):
                endog_interval_map[endog_name] = {
                    'mean': interval_df[col_name],
                    'lower_ci': interval_df[lower_col],
                    'upper_ci': interval_df[upper_col],
                }

        return endog_interval_map

    for _, model_forecast_results in forecast_data_dict.items():
        interval_forecast_df_group = model_forecast_results.get('interval_forecast')
        endog_names = model_forecast_results.get('endog_names', [])

        if interval_forecast_df_group is None or interval_forecast_df_group.empty:
            continue

        endog_interval_map = _extract_endog_intervals(interval_forecast_df_group, endog_names)
        for endog_name, interval_series in endog_interval_map.items():
            pair_name, price_type_short = _parse_endog_name(endog_name)
            if not pair_name or not price_type_short:
                log_stream.write(f"  [WARN] Could not parse pair/price type from endog '{endog_name}'. Skipping.\n")
                continue

            if pair_name not in base_dfs:
                log_stream.write(f"  [WARN] Pair '{pair_name}' from endog '{endog_name}' not found in base_dfs. Skipping.\n")
                continue

            if pair_name not in pair_interval_forecasts_map:
                pair_interval_forecasts_map[pair_name] = {}

            pair_interval_forecasts_map[pair_name][price_type_short] = interval_series

    if not pair_interval_forecasts_map:
        log_stream.write("  [WARN] Tidak ada forecast valid yang cocok dengan pair base_dfs setelah parsing kolom.\n")
        return restored_forecasts

    # === Lakukan restorasi tiap pair ===
    for pair_name, forecasts_for_pair in pair_interval_forecasts_map.items():
        if pair_name not in base_dfs or base_dfs[pair_name].empty:
            log_stream.write(f"  [WARN] Lewati {pair_name}: base data kosong.\n")
            restored_forecasts[pair_name] = pd.DataFrame() # Store empty DataFrame for this pair
            continue

        base_df_sorted = base_dfs[pair_name].sort_index()
        restored_pair_df_list = [] # List to hold restored series for this pair

        log_stream.write(f"  [INFO] Memproses restorasi untuk {pair_name} ...\n")

        for price_type_short, forecast_series_dict in forecasts_for_pair.items():
            # forecast_series_dict contains 'mean', 'lower_ci', 'upper_ci' Series
            if price_type_short not in base_df_sorted.columns:
                log_stream.write(f"  [WARN] Kolom '{price_type_short}' tidak ditemukan di base data {pair_name}. Melewati restorasi untuk jenis harga ini.\n")
                continue

            # Ambil harga terakhir dari data aktual
            price_series_actual = base_df_sorted[price_type_short].dropna()
            if price_series_actual.empty:
                log_stream.write(f"  [WARN] Data aktual kosong untuk '{price_type_short}' di {pair_name}. Melewati restorasi.\n")
                continue

            # Ensure last_actual_price is a scalar, not a Series/DataFrame
            last_actual_price_raw = price_series_actual.iloc[-1]
            if isinstance(last_actual_price_raw, (pd.Series, pd.DataFrame)):
                if not last_actual_price_raw.empty:
                    last_actual_price = last_actual_price_raw.item() # Extract scalar
                else:
                    log_stream.write(f"  [WARN] last_actual_price (scalar conversion) is empty for {pair_name} {price_type_short}.\n")
                    continue # Skip this price_type if cannot get scalar
            else:
                last_actual_price = last_actual_price_raw # Already a scalar

            if pd.isna(last_actual_price):
                log_stream.write(f"  [WARN] last_actual_price is NaN for {pair_name} {price_type_short}. Cannot restore.\n")
                continue # Skip restoration if last_actual_price is NaN

            # Get forecast series
            mean_forecast_returns = forecast_series_dict['mean'].dropna()
            lower_ci_returns = forecast_series_dict['lower_ci'].dropna()
            upper_ci_returns = forecast_series_dict['upper_ci'].dropna()

            if mean_forecast_returns.empty or lower_ci_returns.empty or upper_ci_returns.empty:
                log_stream.write(f"  [WARN] Forecast log return (mean/CI) kosong setelah dropna untuk '{price_type_short}' di {pair_name}. Melewati restorasi.\n")
                continue

            # Ensure index consistency for all forecast series
            forecast_index = mean_forecast_returns.index
            lower_ci_returns = lower_ci_returns.reindex(forecast_index)
            upper_ci_returns = upper_ci_returns.reindex(forecast_index)

            # Restore mean log return to price
            restored_mean_price = last_actual_price * np.exp(mean_forecast_returns.cumsum())
            restored_mean_price.name = f'{price_type_short}_Mean_Forecast'
            restored_pair_df_list.append(restored_mean_price.to_frame())

            # Restore lower CI log return to price
            restored_lower_ci_price = last_actual_price * np.exp(lower_ci_returns.cumsum())
            restored_lower_ci_price.name = f'{price_type_short}_Lower_{int(confidence_level*100)}CI'
            restored_pair_df_list.append(restored_lower_ci_price.to_frame())

            # Restore upper CI log return to price
            restored_upper_ci_price = last_actual_price * np.exp(upper_ci_returns.cumsum())
            restored_upper_ci_price.name = f'{price_type_short}_Upper_{int(confidence_level*100)}CI'
            restored_pair_df_list.append(restored_upper_ci_price.to_frame())

            log_stream.write(f"  [OK] Restored forecast (mean, CI) untuk {pair_name} ({price_type_short}): {len(mean_forecast_returns)} langkah\n")

        # Combine all restored OHLC forecasts for this pair horizontally based on their datetime index
        if restored_pair_df_list:
            restored_pair_df = pd.concat(restored_pair_df_list, axis=1).sort_index()
            restored_forecasts[pair_name] = restored_pair_df
        else:
            log_stream.write(f"  [WARN] Tidak ada kolom OHLC valid untuk {pair_name} setelah restorasi.\n")
            restored_forecasts[pair_name] = pd.DataFrame() # Store empty DataFrame

    log_stream.write("\n[OK] Restorasi peramalan log-return ke harga selesai.\n")
    return restored_forecasts
