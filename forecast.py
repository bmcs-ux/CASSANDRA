## cell_id: auto_varx_forecast_best_model
# ============================================================
# 🔮 AUTO VARX/SARIMAX MODEL SELECTION & FORECASTING (Module)
# ============================================================
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

def auto_varx_forecast(log_stream, combined_log_returns_dict, fitted_models, significant_pair_exog_map, final_stationarized_fred_data, forecast_horizon=2, verbose=True):
    """
    Performs forecasting using pre-fitted VARX/SARIMAX models for each target series.

    Args:
        log_stream (StringIO): Stream to write log messages.
        combined_log_returns_dict (dict): Dictionary where keys are pair names
                                        and values are DataFrames with log return data.
                                        This dictionary should contain the endogenous
                                        variables that the models were fitted on.
        fitted_models (dict): Dictionary where keys are target series names (pairs) and values are
                              dictionaries containing the fitted model results (e.g., from fit_varx_or_arx).
                              Each result dict is expected to contain a 'fitted_model' key with the statsmodels result object
                              and an 'endog_names' key with a list of endogenous variable names used during fitting.
        significant_pair_exog_map (dict): Dictionary mapping original Granger map keys (e.g., "EUR/USD_Close_Log_Return")
                                           to a list of significant exogenous series names (FRED series).
        final_stationarized_fred_data (pd.DataFrame or None): DataFrame with final stationarized FRED data, or None.
        forecast_horizon (int): The number of steps ahead to forecast.
        verbose (bool): If True, print detailed progress messages.

    Returns:
        dict: Dictionary where keys are the *keys from fitted_models* (e.g., "PAIR_ENDOG_VAR") and values are pandas DataFrames
              containing the forecast values for the endogenous variables of that model.
              Column names will match the endogenous variable names used in that specific model.
    """
    if verbose:
        log_stream.write("\n\u2699\uFE0F Menjalankan Auto VARX/SARIMAX Forecasting...\n")
        log_stream.write(f"  Input combined_log_returns_dict keys: {list(combined_log_returns_dict.keys()) if combined_log_returns_dict else 'Empty'}\n")
        # Added type check for the first value in combined_log_returns_dict for better debugging
        if combined_log_returns_dict:
            first_key = next(iter(combined_log_returns_dict))
            first_value = combined_log_returns_dict[first_key]
            log_stream.write(f"  First value type in combined_log_returns_dict: {type(first_value)}\n")

        # Corrected line to safely check final_stationarized_fred_data
        log_stream.write(f"  Input final_stationarized_fred_data columns: {list(final_stationarized_fred_data.columns) if isinstance(final_stationarized_fred_data, pd.DataFrame) and not final_stationarized_fred_data.empty else 'Empty'}\n")

    auto_forecasts = {}

    # Check if the input dictionaries are empty
    if not combined_log_returns_dict:
        if verbose:
            log_stream.write("\u26A0\uFE0F Input combined_log_returns_dict is empty. Skipping forecasting.\n")
        return auto_forecasts
    if not fitted_models:
         if verbose:
             log_stream.write("\u26A0\uFE0F Input fitted_models is empty. Skipping forecasting.\n")
         return auto_forecasts


    # Iterate through the fitted models dictionary
    # The keys are in the format "GROUP_NAME" (for VARMAX) or "PAIR_ENDOG_VARIABLE" (for SARIMAX)
    for model_key, model_result in fitted_models.items():
        if verbose:
            log_stream.write(f"\n\U0001F50D Forecasting using model for: {model_key}\n")

        model = model_result.get('fitted_model')
        endog_names = model_result.get('endog_names') # Get endogenous names used during fitting

        if model is None:
             if verbose:
                  log_stream.write(f"  \u26A0\uFE0F Fitted model object is None for {model_key}. Skipping forecast.\n")
             auto_forecasts[model_key] = {'interval_forecast': pd.DataFrame(), 'endog_names': endog_names} # Store empty forecast
             continue # Skip if no model object is found

        if not endog_names:
             if verbose:
                  log_stream.write(f"  \u26A0\uFE0F Endogenous variable names not found for model {model_key}. Skipping forecast.\n")
             auto_forecasts[model_key] = {'interval_forecast': pd.DataFrame(), 'endog_names': endog_names} # Store empty forecast
             continue

        # Extract the original pair name(s) from the endog_names for data retrieval
        # For SARIMAX, endog_names will have one item, e.g., ['EUR/USD_Close_Log_Return']
        # For VARMAX, endog_names will have multiple items, e.g., ['XAU/USD_Close_Log_Return', 'XAG/USD_Close_Log_Return']

        # We need the data for all pairs involved in this model's endogenous variables
        data_for_this_model_endog = []
        for endog_full_name in endog_names:
            pair_name_from_endog = None
            for p_name_key in combined_log_returns_dict.keys():
                # This check assumes combined_log_returns_dict has the structure like {'D1': {'GBPUSD': df, ...}}
                # We need to access the correct timeframe's dictionary.
                # The function `auto_varx_forecast` is called with `combined_log_returns_dict=mtf_log_returns.get(tf_forecast)`. So it's already specific to a timeframe.
                # The `combined_log_returns_dict` here is expected to be `mtf_log_returns[tf_forecast]`, which is a dict of DataFrames keyed by pair name.
                if endog_full_name.startswith(f"{p_name_key}_"):
                    found_pair_name = p_name_key
                    break

            # This check for `found_pair_name` seems problematic if `combined_log_returns_dict` is the TF-specific one.
            # Let's simplify and assume the `endog_full_name` is directly present as a column in one of the DataFrames in `combined_log_returns_dict` values.

            # First, check if the full endog_name (e.g., 'GBPUSD_Close_Log_Return') is directly a column in any of the DataFrames in `combined_log_returns_dict`'s values
            found_df_for_endog_col = False # Renamed variable to avoid conflict and be more specific
            for pair_key, df_log_return_pair in combined_log_returns_dict.items():
                # Ensure df_log_return_pair is a DataFrame before accessing .empty or .columns
                if not isinstance(df_log_return_pair, pd.DataFrame):
                    if verbose: log_stream.write(f"  \u26A0\uFE0F Expected a DataFrame for pair_key '{pair_key}' but got type {type(df_log_return_pair)}. Skipping this pair for endogenous data collection.\n")
                    continue # Skip to the next item if it's not a DataFrame

                if not df_log_return_pair.empty and endog_full_name in df_log_return_pair.columns:
                    data_for_this_model_endog.append(df_log_return_pair[[endog_full_name]])
                    found_df_for_endog_col = True
                    break # Found the column, move to next endog_full_name
            if not found_df_for_endog_col:
                if verbose: log_stream.write(f"  \u26A0\uFE0F Endogenous column '{endog_full_name}' not found or its DataFrame was invalid/empty. Cannot proceed with this model_key.\n")
                # If a required endogenous column is not found, we cannot forecast for this model_key.
                # Store an empty forecast and skip to the next model.
                auto_forecasts[model_key] = {'interval_forecast': pd.DataFrame(), 'endog_names': endog_names}
                # Set found_df_for_endog to False here to ensure the outer check correctly skips
                found_df_for_endog = False # This variable was previously 'found_df_for_endog' and should be reset or properly managed.
                break # Break from this inner loop and let the outer check for `found_df_for_endog` handle skipping the model_key.

        # After checking all endogenous variables for the model
        if not data_for_this_model_endog or len(data_for_this_model_endog) != len(endog_names):
            if verbose: log_stream.write(f"  \u26A0\uFE0F Not all endogenous data collected for model '{model_key}' or some were invalid. Skipping forecast.\n")
            auto_forecasts[model_key] = {'interval_forecast': pd.DataFrame(), 'endog_names': endog_names}
            continue # Skip to the next model_key because this one is incomplete.

        # Combine the endogenous data into a single DataFrame, aligning by index
        df_endog_data_for_model = pd.concat(data_for_this_model_endog, axis=1, join='inner').sort_index().dropna()

        if df_endog_data_for_model.empty:
            if verbose: log_stream.write(f"  \u26A0\uFE0F Combined endogenous data for model '{model_key}' is empty after dropna. Skipping forecast.\n")
            auto_forecasts[model_key] = {'interval_forecast': pd.DataFrame(), 'endog_names': endog_names} # Store empty forecast
            continue


        # 🔮 Forecast 'forecast_horizon' steps ahead
        try:
            # prepare future datetime index
            # Get the last time from the ENDOGENOUS data used for fitting
            last_time = df_endog_data_for_model.index[-1]
            # try to infer freq from the endog data index
            inferred = pd.infer_freq(df_endog_data_for_model.index)
            freq_to_use = inferred if inferred is not None else 'D' # Default to Daily if freq cannot be inferred
            future_index = pd.date_range(start=last_time + pd.tseries.frequencies.to_offset(freq_to_use), periods=forecast_horizon, freq=freq_to_use)

            if verbose:
                 log_stream.write(f"  Prepared future_index ({freq_to_use} freq): {future_index[0].date()} to {future_index[-1].date()}\n")


            # Prepare future_exog if the model uses exogenous variables (k_exog > 0)
            future_exog = None
            k_exog = None
            try:
                k_exog = int(getattr(model.model, 'k_exog', 0))
                if verbose: log_stream.write(f"  Model k_exog: {k_exog}\n")
            except Exception:
                k_exog = 0 # Assume no exogenous variables
                if verbose: log_stream.write(f"  Could not determine model k_exog. Assuming 0.\n")


            if k_exog and k_exog > 0:
                # === MODIFICATION START ===
                # Aggregate all unique significant exogenous variables for this group's endogenous variables
                all_exog_series_names = set()
                for endog_var_name in endog_names:
                    # Retrieve exogs for each individual endogenous series from the map
                    individual_exogs = significant_pair_exog_map.get(endog_var_name, [])
                    for exog_cand in individual_exogs:
                        if exog_cand not in endog_names: # Ensure exog isn't one of the endogenous variables in this group
                            all_exog_series_names.add(exog_cand)

                exog_series_names = list(all_exog_series_names)
                # === MODIFICATION END ===

                if verbose: log_stream.write(f"  Significant exog names aggregated for model {model_key}: {exog_series_names}\n")

                if exog_series_names:
                    exog_dict = {}
                    found_all_exogs = True
                    for exn in exog_series_names:
                        # Find the final transformed data for this exogenous series
                        # The `final_stationarized_fred_data` can be an empty DataFrame if no FRED data is used/available
                        df_fred_exog_series = None
                        if isinstance(final_stationarized_fred_data, pd.DataFrame) and not final_stationarized_fred_data.empty:
                            if exn in final_stationarized_fred_data.columns:
                                df_fred_exog_series = final_stationarized_fred_data[[exn]]

                        # If it's not a FRED series or FRED data is missing for this exog, check combined_log_returns_dict
                        if df_fred_exog_series is None or df_fred_exog_series.empty:
                            # Check in log_returns for other pair's OHLC log returns
                            found_log_return_for_exog = False
                            for pair_key, df_log_return_pair in combined_log_returns_dict.items():
                                # Ensure df_log_return_pair is a DataFrame before accessing .empty or .columns
                                if not isinstance(df_log_return_pair, pd.DataFrame):
                                    if verbose: log_stream.write(f"  \u26A0\uFE0F Expected a DataFrame for pair_key '{pair_key}' but got type {type(df_log_return_pair)}. Skipping this pair for exogenous data collection.\n")
                                    continue # Skip to the next item if it's not a DataFrame

                                if not df_log_return_pair.empty and exn in df_log_return_pair.columns:
                                    # The exog data is the actual log return series itself
                                    # We need the last value of this series for persistence forecasting of exog
                                    exog_series_data = df_log_return_pair[[exn]].sort_index().dropna()
                                    if not exog_series_data.empty:
                                        exog_dict[exn] = exog_series_data.iloc[-1, 0]
                                        found_log_return_for_exog = True
                                        break
                                    else:
                                        if verbose: log_stream.write(f"  \u26A0\uFE0F Exog series '{exn}' (log-return) is empty after dropna. Cannot construct future_exog.\n")
                                        found_all_exogs = False
                                        break

                            if not found_log_return_for_exog:
                                if verbose: log_stream.write(f"  \u26A0\uFE0F Exog series '{exn}' not found in FRED or log-returns data. Cannot construct future_exog.\n")
                                found_all_exogs = False
                                break
                        else: # It's a FRED series, and we found it
                             # Get the last value from the FRED series
                             if not df_fred_exog_series.empty:
                                  exog_dict[exn] = df_fred_exog_series.sort_index().iloc[-1, 0]
                                  if verbose: log_stream.write(f"  Last value for '{exn}': {exog_dict[exn]}\n")
                             else:
                                  if verbose: log_stream.write(f"  \u26A0\uFE0F FRED series '{exn}' is empty. Cannot construct future_exog.\n")
                                  found_all_exogs = False
                                  break

                    if found_all_exogs and exog_dict:
                         # Create a DataFrame for future exogenous values by repeating the last known values
                         future_exog = pd.DataFrame([exog_dict] * forecast_horizon, index=future_index)
                         # Ensure column order matches model's exog_names if available
                         model_exog_names = getattr(model.model, 'exog_names', list(exog_dict.keys()))
                         # Filter model_exog_names to only include those found in exog_dict
                         model_exog_names_filtered = [name for name in model_exog_names if name in exog_dict]

                         # Ensure future_exog columns match the filtered model_exog_names and are in the same order
                         if set(future_exog.columns) == set(model_exog_names_filtered):
                              future_exog = future_exog[model_exog_names_filtered]
                              if verbose: log_stream.write(f"  Constructed future_exog DataFrame. Shape: {future_exog.shape}, Columns: {future_exog.columns}\n")
                         else:
                             if verbose: log_stream.write(f"  \u26A0\uFE0F Exog column mismatch between constructed future_exog ({list(future_exog.columns)}) and model exog names ({model_exog_names_filtered}). Cannot use exogenous variables for forecasting.\n")
                             future_exog = None # Cannot proceed with exog


                    else:
                         if verbose: log_stream.write(f"  \u26A0\uFE0F Could not construct valid future_exog data for {model_key} based on significant exogs map.\n")
                         future_exog = None # Cannot proceed with exog

                elif verbose:
                    log_stream.write(f"  \u2139\uFE0F No significant exogenous variables mapped for model {model_key}.\n")
                    future_exog = None # No exog to provide

            elif verbose:
                 log_stream.write(f"  \u2139\uFE0F Model does not have exogenous variables (k_exog=0). Forecasting without exog.\n")
                 future_exog = None # No exog needed


            # Now try to forecast using model's best available method
            try:
                if hasattr(model, 'get_forecast'):
                    # prefer get_forecast
                    if future_exog is not None and not future_exog.empty:
                         if verbose: log_stream.write("  \u2139\uFE0F Using model.get_forecast with future_exog.\n")
                         # Ensure future_exog has the correct index (future_index)
                         future_exog = future_exog.reindex(future_index)
                         res = model.get_forecast(steps=forecast_horizon, exog=future_exog)
                    else:
                         if verbose: log_stream.write("  \u2139\uFE0F Using model.get_forecast without exog.\n")
                         res = model.get_forecast(steps=forecast_horizon)

                    # get_forecast returns a PredictionResultsWrapper, which has a summary_frame method.
                    # The summary_frame will contain columns like 'mean', 'mean_ci_lower', 'mean_ci_upper' for each endogenous variable.
                    # For VARMAX, it usually prefixes with '0_mean', '1_mean', etc. for each endogenous variable.
                    forecast_res_df = res.summary_frame(alpha=0.05)

                    # Rename columns for clarity
                    renaming_map = {}
                    if len(endog_names) == 1:
                        renaming_map['mean'] = f'{endog_names[0]}_Forecast'
                        renaming_map['mean_ci_lower'] = f'{endog_names[0]}_Lower'
                        renaming_map['mean_ci_upper'] = f'{endog_names[0]}_Upper'
                    else:
                        for i, endog_name in enumerate(endog_names):
                            renaming_map[f'{i}_mean'] = f'{endog_name}_Forecast'
                            renaming_map[f'{i}_mean_ci_lower'] = f'{endog_name}_Lower'
                            renaming_map[f'{i}_mean_ci_upper'] = f'{endog_name}_Upper'

                    # Apply renaming, handling cases where columns might not exist if forecast_res_df is empty
                    forecast_df_renamed = forecast_res_df.rename(columns={k: v for k, v in renaming_map.items() if k in forecast_res_df.columns})

                    # Set index to future dates
                    if len(forecast_df_renamed) == len(future_index):
                        forecast_df_renamed.index = future_index
                    else:
                        if verbose: log_stream.write(f"  \u26A0\uFE0F Mismatch in forecast length ({len(forecast_df_renamed)}) and future_index length ({len(future_index)}). Using default index.\n")

                    auto_forecasts[model_key] = {'interval_forecast': forecast_df_renamed, 'endog_names': endog_names}
                    if verbose:
                        log_stream.write(f"  -> OK (get_forecast): forecast shape {forecast_df_renamed.shape}, columns {list(forecast_df_renamed.columns)}\n")
                    continue

                # fallback: try predict with start/end
                if hasattr(model, 'predict'):
                    try:
                        if verbose: log_stream.write("  \u2139\uFE0F Using model.predict.\n")
                        # Need the index of the last observation used for fitting
                        start_idx = df_endog_data_for_model.index[-1] # Use the last index of the sorted data
                        # The predict method often takes index values for start/end
                        end_idx = future_index[-1]

                        if future_exog is not None and not future_exog.empty:
                            # predict method usually takes exog aligned to the forecast index
                            raw = model.predict(start=start_idx, end=end_idx, exog=future_exog)
                        else:
                            raw = model.predict(start=start_idx, end=end_idx)

                        # Ensure the result is a DataFrame with the correct index (future_index)
                        # The predict method might return a Series or DataFrame depending on the model
                        if isinstance(raw, pd.Series):
                             # Ensure column name matches the single endogenous variable name
                            forecast_df = pd.DataFrame(raw, columns=[endog_names[0]]) if endog_names else pd.DataFrame(raw, columns=['forecast'])
                        elif isinstance(raw, pd.DataFrame):
                             # Ensure column names match endogenous names
                             if len(raw.columns) == len(endog_names):
                                  raw.columns = endog_names
                             forecast_df = raw
                        else:
                            # Handle other raw types, try to convert to DataFrame
                            if np.ndim(raw) == 1:
                                forecast_df = pd.DataFrame(raw, columns=[endog_names[0]]) if endog_names else pd.DataFrame(raw, columns=['forecast'])
                            else:
                                forecast_df = pd.DataFrame(raw, columns=endog_names) if len(endog_names) == raw.shape[1] else pd.DataFrame(raw)


                        # Ensure the index is set correctly and slice if the last training point is included
                        if not forecast_df.empty:
                             # Find the first index in forecast_df that is in future_index
                             # This handles cases where predict includes the last training point
                             first_future_idx = future_index[0]
                             if first_future_idx in forecast_df.index:
                                  forecast_df = forecast_df.loc[first_future_idx:]
                                  # Ensure the index is exactly future_index after slicing
                                  if len(forecast_df) == len(future_index):
                                       forecast_df.index = future_index
                                  else:
                                       if verbose: log_stream.write(f"  \u26A0\uFE0F Mismatch after slicing predict output ({len(forecast_df)}) and future_index ({len(future_index)}). Using sliced index.\n")
                             elif verbose:
                                  log_stream.write(f"  \u26A0\uFE0F First future index {first_future_idx} not found in predict output index. Using predict output index.\n")


                        auto_forecasts[model_key] = {'interval_forecast': forecast_df, 'endog_names': endog_names} # Use model_key as the key for the forecast result
                        if verbose:
                            log_stream.write(f"  -> OK (predict): forecast shape {forecast_df.shape}, columns {list(forecast_df.columns)}\n")
                        continue
                    except Exception as e_pred:
                        if verbose:
                            log_stream.write(f"  -> predict() attempt failed: {e_pred}\n")

            except Exception as e_getf:
                if verbose:
                    log_stream.write(f"  -> get_forecast/predict block error: {e_getf}\n")


            # If reached here, forecasting attempts failed -> fallback persistence
            if not df_endog_data_for_model.empty:
                # Take the last values of all endogenous variables
                last_vals = df_endog_data_for_model.iloc[-1]
                # Create a DataFrame by repeating the last values for the forecast horizon
                forecast_data = np.tile(last_vals.values, (forecast_horizon, 1))
                forecast_df = pd.DataFrame(forecast_data, index=future_index, columns=endog_names) # Use endog names as columns

                auto_forecasts[model_key] = {'interval_forecast': forecast_df, 'endog_names': endog_names} # Use model_key as the key for the forecast result
                if verbose:
                    log_stream.write(f"  \u26A0\uFE0F Forecast fallback persistence used: shape {forecast_df.shape}, columns {list(forecast_df.columns)}\n")
            else:
                if verbose:
                    log_stream.write(f"  \u274C No data available for persistence forecast for {model_key}.\n")
                auto_forecasts[model_key] = {'interval_forecast': pd.DataFrame(), 'endog_names': endog_names} # Empty forecast on complete failure


        except Exception as e:
            if verbose:
                log_stream.write(f"  \u26A0\uFE0F General Forecast failed for {model_key}, fallback persistence: {e}\n")
            # absolute fallback
            if not df_endog_data_for_model.empty:
                 last_vals = df_endog_data_for_model.iloc[-1]
                 # Need a fallback future index if the original index inference failed
                 fallback_future_index = pd.date_range(start=pd.Timestamp.now(), periods=forecast_horizon, freq='D') # Default to Daily
                 forecast_data = np.tile(last_vals.values, (forecast_horizon, 1))
                 forecast_df = pd.DataFrame(forecast_data, index=fallback_future_index, columns=endog_names) # Use endog names as columns
                 auto_forecasts[model_key] = {'interval_forecast': forecast_df, 'endog_names': endog_names} # Use model_key as the key for the forecast result
                 if verbose:
                      log_stream.write(f"  \u26A0\uFE0F Forecast fallback persistence used: shape {forecast_df.shape}, columns {list(forecast_df.columns)}\n")
            else:
                 if verbose:
                     log_stream.write(f"  \u274C No data available for fallback persistence forecast for {model_key}.\n")
                 auto_forecasts[model_key] = {'interval_forecast': pd.DataFrame(), 'endog_names': endog_names}


    if verbose:
        log_stream.write("\n\u2705 Auto VARX/SARIMAX Forecasting finished.\n")
        log_stream.write("\n\U0001F4CA Forecasts generated for:\n")
        if auto_forecasts:
             for model_key, forecast_data in auto_forecasts.items(): # Iterate through the keys used in auto_forecasts
                 if 'interval_forecast' in forecast_data and not forecast_data['interval_forecast'].empty:
                      log_stream.write(f"  \u2022 {model_key}: Shape {forecast_data['interval_forecast'].shape}, Columns: {list(forecast_data['interval_forecast'].columns)}\n")
                 else:
                      log_stream.write(f"  \u2022 {model_key}: Empty forecast.\n")
        else:
             log_stream.write("  \u2139\uFE0F No forecasts were generated.\n")


    return auto_forecasts
