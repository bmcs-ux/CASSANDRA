#
def restore_log_returns_to_price(log_stream, forecast_data_dict, base_dfs, confidence_level=0.95):
    """
    Mengembalikan hasil peramalan log-return (termasuk interval kepercayaan) menjadi
    peramalan harga (OHLC) dalam skala asli, termasuk batas atas dan bawah harga.

    Args:
        log_stream (StringIO): Stream to write log messages.
        forecast_data_dict (dict): Dictionary dengan key model group name (e.g., 'METALS_CLOSE')
                                   dan value berupa dictionary yang berisi:
                                   - 'mean_forecast': DataFrame of mean log return forecasts.
                                   - 'interval_forecast': DataFrame dengan mean, standard deviation,
                                                          dan confidence intervals untuk log returns.
        base_dfs (dict): Dictionary dengan key 'PAIR' dan value DataFrame harga asli (OHLC).
        confidence_level (float): Tingkat kepercayaan yang digunakan untuk interval (mis. 0.95).

    Returns:
        dict: Dictionary berisi DataFrame harga hasil restorasi untuk tiap pair.
              Kolom berformat 'PAIR_PRICE_Mean_Forecast', 'PAIR_PRICE_Lower_CI', dst.
    """
    import numpy as np
    import pandas as pd
    import re # Import regex for more robust parsing

    log_stream.write(f"  [INFO] Mengembalikan peramalan log-return ke peramalan harga (OHLC) dengan interval {int(confidence_level*100)}%...\n")
    restored_forecasts = {}

    if not forecast_data_dict or not base_dfs:
        log_stream.write("  [WARN] forecast_data_dict atau base_dfs kosong. Tidak ada peramalan yang direstorasi.\n")
        return restored_forecasts

    # Temporary map to group interval_forecasts by their actual pair name (e.g., 'XAU/USD')
    # This will hold { 'XAU/USD': { 'Close': interval_forecast_df_single_col_series, ... } }
    pair_interval_forecasts_map = {}

    ci_lower_col_suffix = f'_Lower_{int(confidence_level*100)}CI'
    ci_upper_col_suffix = f'_Upper_{int(confidence_level*100)}CI'
    mean_col_suffix = '_Mean'

    for model_key, model_forecast_results in forecast_data_dict.items():
        interval_forecast_df_group = model_forecast_results.get('interval_forecast')

        if interval_forecast_df_group is None or interval_forecast_df_group.empty:
            continue

        # Iterate through columns to extract pair_name, price_type, and their forecasts
        for col_name in interval_forecast_df_group.columns:
            # Example col_name: 'XAU/USD_Close_Log_Return_Mean'
            # We need to extract 'XAU/USD' and 'Close_Log_Return'
            match = re.match(r'(.+?)_(Open|High|Low|Close)_Log_Return_Mean', col_name)
            if match:
                pair_name = match.group(1)
                price_type_full = match.group(2) + '_Log_Return' # e.g., 'Close_Log_Return'
                price_type_short = match.group(2) # e.g., 'Close'

                # Ensure the extracted pair_name actually exists in base_dfs
                if pair_name not in base_dfs:
                    log_stream.write(f"  [WARN] Pair '{pair_name}' from forecast column '{col_name}' not found in base_dfs. Skipping.\n")
                    continue

                if pair_name not in pair_interval_forecasts_map:
                    pair_interval_forecasts_map[pair_name] = {}

                # We need to collect mean, lower, and upper bounds for each price_type
                mean_col = f'{pair_name}_{price_type_full}{mean_col_suffix}'
                lower_ci_col = f'{pair_name}_{price_type_full}{ci_lower_col_suffix}'
                upper_ci_col = f'{pair_name}_{price_type_full}{ci_upper_col_suffix}'

                # Make sure these columns exist in the interval_forecast_df_group
                if mean_col in interval_forecast_df_group.columns and \
                   lower_ci_col in interval_forecast_df_group.columns and \
                   upper_ci_col in interval_forecast_df_group.columns:

                    # Store the forecast Series for this specific price type
                    pair_interval_forecasts_map[pair_name][price_type_short] = {
                        'mean': interval_forecast_df_group[mean_col],
                        'lower_ci': interval_forecast_df_group[lower_ci_col],
                        'upper_ci': interval_forecast_df_group[upper_ci_col]
                    }
                else:
                    log_stream.write(f"  [WARN] Missing interval columns for {price_type_short} in {pair_name}. Skipping.\n")
            else:
                # This regex captures columns ending in _Mean, _StdDev, _Lower_CI, _Upper_CI.
                # We only want the ones ending in _Mean to avoid double processing.
                if not col_name.endswith(mean_col_suffix):
                     continue

                # Fallback for columns that might not perfectly match the standard pattern
                # This part is a bit tricky, but we need to ensure we parse correctly
                # If the col_name is like 'EndogVar_Mean', try to get 'EndogVar'
                match_fallback = re.match(r'(.+?)_Mean', col_name)
                if match_fallback:
                    endog_var_full_name = match_fallback.group(1) # e.g., 'XAU/USD_Close_Log_Return'
                    # Try to extract pair_name and price_type from this full name
                    match_endog_var = re.match(r'(.+?)_(Open|High|Low|Close)_Log_Return', endog_var_full_name)
                    if match_endog_var:
                        pair_name = match_endog_var.group(1)
                        price_type_short = match_endog_var.group(2)

                        if pair_name not in base_dfs:
                            log_stream.write(f"  [WARN] Fallback: Pair '{pair_name}' not in base_dfs for {endog_var_full_name}. Skipping.\n")
                            continue

                        if pair_name not in pair_interval_forecasts_map:
                            pair_interval_forecasts_map[pair_name] = {}

                        mean_col = f'{endog_var_full_name}{mean_col_suffix}'
                        lower_ci_col = f'{endog_var_full_name}{ci_lower_col_suffix}'
                        upper_ci_col = f'{endog_var_full_name}{ci_upper_col_suffix}'

                        if mean_col in interval_forecast_df_group.columns and \
                           lower_ci_col in interval_forecast_df_group.columns and \
                           upper_ci_col in interval_forecast_df_group.columns:
                            pair_interval_forecasts_map[pair_name][price_type_short] = {
                                'mean': interval_forecast_df_group[mean_col],
                                'lower_ci': interval_forecast_df_group[lower_ci_col],
                                'upper_ci': interval_forecast_df_group[upper_ci_col]
                            }
                        else:
                            log_stream.write(f"  [WARN] Fallback: Missing interval columns for {price_type_short} in {pair_name}. Skipping.\n")
                    else:
                         log_stream.write(f"  [WARN] Could not parse pair/price type from fallback column: {endog_var_full_name}. Skipping.\n")
                else:
                    log_stream.write(f"  [WARN] Could not parse pair and price type from column name: {col_name}. Skipping.\n")

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
