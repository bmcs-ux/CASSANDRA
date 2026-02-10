#
## cell_id: auto_varx_forecast_best_model
# ============================================================
# 🪄 AUTO VARX/SARIMAX MODEL SELECTION & FORECASTING (Module)
# ============================================================
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

def auto_varx_forecast(combined_log_returns_dict, fitted_models, significant_pair_exog_map, final_stationarized_fred_data, forecast_horizon=2, verbose=True):
    """
    Performs forecasting using pre-fitted VARX/SARIMAX models for each target series.

    Args:
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
        final_stationarized_fred_data (dict): Dictionary of DataFrames with final stationarized FRED data.
                                             Keys are series names, values are DataFrames.
                                             Expected to have the final transformed value as the first column,
                                             named after the original FRED series name.
        forecast_horizon (int): The number of steps ahead to forecast.
        verbose (bool): If True, print detailed progress messages.

    Returns:
        dict: Dictionary where keys are the *keys from fitted_models* (e.g., "PAIR_ENDOG_VAR") and values are pandas DataFrames
              containing the forecast values for the endogenous variables of that model.
              Column names will match the endogenous variable names used in that specific model.
    """
    if verbose:
        print("\n⚙️ Menjalankan Auto VARX/SARIMAX Forecasting...")
        print(f"  Input combined_log_returns_dict keys: {list(combined_log_returns_dict.keys()) if combined_log_returns_dict else 'Empty'}")
        print(f"  Input fitted_models keys: {list(fitted_models.keys()) if fitted_models else 'Empty'}")
        # print(f"  Input significant_pair_exog_map keys: {list(significant_pair_exog_map.keys()) if significant_pair_exog_map else 'Empty'}") # Can be verbose
        print(f"  Input final_stationarized_fred_data keys: {list(final_stationarized_fred_data.keys()) if final_stationarized_fred_data else 'Empty'}")


    auto_forecasts = {}

    # Check if the input dictionaries are empty
    if not combined_log_returns_dict:
        if verbose:
            print("⚠️ Input combined_log_returns_dict is empty. Skipping forecasting.")
        return auto_forecasts
    if not fitted_models:
         if verbose:
             print("⚠️ Input fitted_models is empty. Skipping forecasting.")
         return auto_forecasts


    # Iterate through the fitted models dictionary
    # The keys are in the format "GROUP_NAME" (for VARMAX) or "PAIR_ENDOG_VARIABLE" (for SARIMAX)
    for model_key, model_result in fitted_models.items():
        if verbose:
            print(f"\n🔍 Forecasting using model for: {model_key}")

        model = model_result.get('fitted_model')
        endog_names = model_result.get('endog_names') # Get endogenous names used during fitting

        if model is None:
             if verbose:
                  print(f"  ⚠️ Fitted model object is None for {model_key}. Skipping forecast.")
             auto_forecasts[model_key] = pd.DataFrame() # Store empty forecast
             continue # Skip if no model object is found

        if not endog_names:
             if verbose:
                  print(f"  ⚠️ Endogenous variable names not found for model {model_key}. Skipping forecast.")
             auto_forecasts[model_key] = pd.DataFrame() # Store empty forecast
             continue

        # Extract the original pair name(s) from the endog_names for data retrieval
        # For SARIMAX, endog_names will have one item, e.g., ['EUR/USD_Close_Log_Return']
        # For VARMAX, endog_names will have multiple items, e.g., ['XAU/USD_Close_Log_Return', 'XAG/USD_Close_Log_Return']

        # We need the data for all pairs involved in this model's endogenous variables
        data_for_this_model_endog = []
        for endog_full_name in endog_names:
            found_pair_name = None
            for p_name_key in combined_log_returns_dict.keys():
                if endog_full_name.startswith(f"{p_name_key}_"):
                    found_pair_name = p_name_key
                    break

            if found_pair_name and found_pair_name in combined_log_returns_dict:
                df_log_return_pair = combined_log_returns_dict[found_pair_name]
                # The actual column name in df_log_return_pair is just the OHLC_Log_Return part
                column_in_pair_df = endog_full_name.replace(f"{found_pair_name}_", "")
                if not df_log_return_pair.empty and column_in_pair_df in df_log_return_pair.columns:
                    data_for_this_model_endog.append(df_log_return_pair[[column_in_pair_df]].rename(columns={column_in_pair_df: endog_full_name}))
                else:
                    if verbose: print(f"  ⚠️ Endogenous column '{column_in_pair_df}' not found in data for pair '{found_pair_name}'. Skipping model_key {model_key}.")
                    auto_forecasts[model_key] = pd.DataFrame()
                    continue
            else:
                if verbose: print(f"  ⚠️ Data for pair '{found_pair_name}' not found in combined_log_returns_dict. Skipping model_key {model_key}.")
                auto_forecasts[model_key] = pd.DataFrame()
                continue

        if not data_for_this_model_endog or len(data_for_this_model_endog) != len(endog_names):
            if verbose: print(f"  ⚠️ Not all endogenous data collected for model '{model_key}'. Skipping forecast.")
            auto_forecasts[model_key] = pd.DataFrame()
            continue

        # Combine the endogenous data into a single DataFrame, aligning by index
        df_endog_data_for_model = pd.concat(data_for_this_model_endog, axis=1, join='inner').sort_index().dropna()

        if df_endog_data_for_model.empty:
            if verbose: print(f"  ⚠️ Combined endogenous data for model '{model_key}' is empty after dropna. Skipping forecast.")
            auto_forecasts[model_key] = pd.DataFrame() # Store empty forecast
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
                 print(f"  Prepared future_index ({freq_to_use} freq): {future_index[0].date()} to {future_index[-1].date()}")


            # Prepare future_exog if the model uses exogenous variables (k_exog > 0)
            future_exog = None
            k_exog = None
            try:
                k_exog = int(getattr(model.model, 'k_exog', 0))
                if verbose: print(f"  Model k_exog: {k_exog}")
            except Exception:
                k_exog = 0 # Assume no exogenous variables
                if verbose: print(f"  Could not determine model k_exog. Assuming 0.")


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

                if verbose: print(f"  Significant exog names aggregated for model {model_key}: {exog_series_names}")

                if exog_series_names:
                    exog_dict = {}
                    found_all_exogs = True
                    for exn in exog_series_names:
                        # Find the final transformed data for this exogenous series
                        df_fred_exog = final_stationarized_fred_data.get(exn) # Use original FRED name as key for FRED exogs

                        # If it's not a FRED series, it must be another log-return series from combined_log_returns_dict
                        if df_fred_exog is None or df_fred_exog.empty:
                            # Check in log_returns for other pair's OHLC log returns
                            found_pair_name_for_exog = None
                            for p_name_key in combined_log_returns_dict.keys():
                                if exn.startswith(f"{p_name_key}_"):
                                    found_pair_name_for_exog = p_name_key
                                    break
                            if found_pair_name_for_exog and found_pair_name_for_exog in combined_log_returns_dict:
                                df_log_return_exog_pair = combined_log_returns_dict[found_pair_name_for_exog]
                                column_in_exog_df = exn.replace(f"{found_pair_name_for_exog}_", "")
                                if not df_log_return_exog_pair.empty and column_in_exog_df in df_log_return_exog_pair.columns:
                                    # The exog data is the actual log return series itself
                                    # We need the last value of this series for persistence forecasting of exog
                                    exog_series_data = df_log_return_exog_pair[[column_in_exog_df]].rename(columns={column_in_exog_df: exn}).sort_index().dropna()
                                    if not exog_series_data.empty:
                                        exog_dict[exn] = exog_series_data.iloc[-1, 0]
                                    else:
                                        if verbose: print(f"  ⚠️ Exog series '{exn}' (log-return) is empty. Cannot construct future_exog.")
                                        found_all_exogs = False
                                        break
                                else:
                                    if verbose: print(f"  ⚠️ Exog column '{column_in_exog_df}' not found in log-return data for pair '{found_pair_name_for_exog}'. Cannot construct future_exog.")
                                    found_all_exogs = False
                                    break
                            else:
                                if verbose: print(f"  ⚠️ Exog series '{exn}' not found in FRED or log-returns data. Cannot construct future_exog.")
                                found_all_exogs = False
                                break
                        else: # It's a FRED series
                             # Assume the first column is the final transformed value
                             if len(df_fred_exog.columns) > 0:
                                  # Get the last value from the first column (which should be the transformed value)
                                  exog_dict[exn] = df_fred_exog.sort_index().iloc[-1, 0]
                                  if verbose: print(f"  Last value for '{exn}': {exog_dict[exn]}")
                             else:
                                  if verbose: print(f"  ⚠️ No value column found for FRED series '{exn}' in final_stationarized_fred_data.")
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
                              if verbose: print(f"  Constructed future_exog DataFrame. Shape: {future_exog.shape}, Columns: {future_exog.columns}")
                         else:
                             if verbose: print(f"  ⚠️ Exog column mismatch between constructed future_exog ({list(future_exog.columns)}) and model exog names ({model_exog_names_filtered}). Cannot use exogenous variables for forecasting.")
                             future_exog = None # Cannot proceed with exog


                    else:
                         if verbose: print(f"  ⚠️ Could not construct valid future_exog data for {model_key} based on significant exogs map.")
                         future_exog = None # Cannot proceed with exog

                elif verbose:
                    print(f"  ℹ️ No significant exogenous variables mapped for model {model_key}.")
                    future_exog = None # No exog to provide

            elif verbose:
                 print(f"  ℹ️ Model does not have exogenous variables (k_exog=0). Forecasting without exog.")
                 future_exog = None # No exog needed


            # Now try to forecast using model's best available method
            try:
                if hasattr(model, 'get_forecast'):
                    # prefer get_forecast
                    if future_exog is not None and not future_exog.empty:
                         if verbose: print("  ℹ️ Using model.get_forecast with future_exog.")
                         # Ensure future_exog has the correct index (future_index)
                         future_exog = future_exog.reindex(future_index)
                         res = model.get_forecast(steps=forecast_horizon, exog=future_exog)
                    else:
                         if verbose: print("  ℹ️ Using model.get_forecast without exog.")
                         res = model.get_forecast(steps=forecast_horizon)

                    pm = getattr(res, 'predicted_mean', None)

                    if pm is None:
                        # fallback to summary_frame mean column(s)
                        try:
                            sf = res.summary_frame()
                            # Find columns containing 'mean' (case-insensitive) or just take the first columns
                            mean_cols = [c for c in sf.columns if 'mean' in str(c).lower()]
                            if mean_cols:
                                pm = sf[mean_cols]
                            else:
                                # Take the first N columns where N is the number of endogenous variables
                                if not sf.empty and len(sf.columns) >= len(endog_names):
                                    pm = sf.iloc[:, :len(endog_names)]
                                else:
                                    pm = None # DataFrame is empty or not enough columns
                        except Exception:
                            pm = None # Cannot get summary frame

                    if pm is not None:
                        # ensure shape -> DataFrame with index future_index
                        pred_df = pd.DataFrame(pm)
                        # Ensure the index of pred_df is set correctly to future_index
                        if len(pred_df) == len(future_index):
                            pred_df.index = future_index
                        else:
                            if verbose: print(f"  ⚠️ Mismatch in forecast length ({len(pred_df)}) and future_index length ({len(future_index)}). Using default index.")

                        # Ensure column names match the endogenous names used during fitting
                        if len(pred_df.columns) == len(endog_names):
                             pred_df.columns = endog_names
                        else:
                             if verbose: print(f"  ⚠️ Mismatch in forecast columns ({len(pred_df.columns)}) and model endog names ({len(endog_names)}). Using default column names.")

                        forecast_df = pred_df
                        auto_forecasts[model_key] = forecast_df # Use model_key as the key for the forecast result
                        if verbose:
                            print(f"  -> OK (get_forecast): forecast shape {forecast_df.shape}, columns {list(forecast_df.columns)}")
                        continue

                # fallback: try predict with start/end
                if hasattr(model, 'predict'):
                    try:
                        if verbose: print("  ℹ️ Using model.predict.")
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
                                       if verbose: print(f"  ⚠️ Mismatch after slicing predict output ({len(forecast_df)}) and future_index ({len(future_index)}). Using sliced index.")
                             elif verbose:
                                  print(f"  ⚠️ First future index {first_future_idx} not found in predict output index. Using predict output index.")


                        auto_forecasts[model_key] = forecast_df # Use model_key as the key for the forecast result
                        if verbose:
                            print(f"  -> OK (predict): forecast shape {forecast_df.shape}, columns {list(forecast_df.columns)}")
                        continue
                    except Exception as e_pred:
                        if verbose:
                            print(f"  -> predict() attempt failed: {e_pred}")

            except Exception as e_getf:
                if verbose:
                    print(f"  -> get_forecast/predict block error: {e_getf}")


            # If reached here, forecasting attempts failed -> fallback persistence
            if not df_endog_data_for_model.empty:
                # Take the last values of all endogenous variables
                last_vals = df_endog_data_for_model.iloc[-1]
                # Create a DataFrame by repeating the last values for the forecast horizon
                forecast_data = np.tile(last_vals.values, (forecast_horizon, 1))
                forecast_df = pd.DataFrame(forecast_data, index=future_index, columns=endog_names) # Use endog names as columns

                auto_forecasts[model_key] = forecast_df # Use model_key as the key for the forecast result
                if verbose:
                    print(f"  ⚠️ Forecast fallback persistence used: shape {forecast_df.shape}, columns {list(forecast_df.columns)}")
            else:
                if verbose:
                    print(f"  ❌ No data available for persistence forecast for {model_key}.")
                auto_forecasts[model_key] = pd.DataFrame() # Empty forecast on complete failure


        except Exception as e:
            if verbose:
                print(f"  ⚠️ General Forecast failed for {model_key}, fallback persistence: {e}")
            # absolute fallback
            if not df_endog_data_for_model.empty:
                 last_vals = df_endog_data_for_model.iloc[-1]
                 # Need a fallback future index if the original index inference failed
                 fallback_future_index = pd.date_range(start=pd.Timestamp.now(), periods=forecast_horizon, freq='D') # Default to Daily
                 forecast_data = np.tile(last_vals.values, (forecast_horizon, 1))
                 forecast_df = pd.DataFrame(forecast_data, index=fallback_future_index, columns=endog_names) # Use endog names as columns
                 auto_forecasts[model_key] = forecast_df # Use model_key as the key for the forecast result
                 if verbose:
                      print(f"  ⚠️ Forecast fallback persistence used: shape {forecast_df.shape}, columns {list(forecast_df.columns)}")
            else:
                 if verbose:
                     print(f"  ❌ No data available for fallback persistence forecast for {model_key}.")
                 auto_forecasts[model_key] = pd.DataFrame()


    if verbose:
        print("\n✅ Auto VARX/SARIMAX Forecasting finished.")
        print("\n📊 Forecasts generated for:")
        if auto_forecasts:
             for model_key, forecast_df in auto_forecasts.items(): # Iterate through the keys used in auto_forecasts
                 if not forecast_df.empty:
                      print(f"  • {model_key}: Shape {forecast_df.shape}, Columns: {list(forecast_df.columns)}")
                 else:
                      print(f"  • {model_key}: Empty forecast.")
        else:
             print("  ℹ️ No forecasts were generated.")


    return auto_forecasts
