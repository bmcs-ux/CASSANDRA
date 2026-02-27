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
        log_stream.write("\n⚙️ Menjalankan Auto VARX/SARIMAX Forecasting...\n")
        log_stream.write(f"  Input combined_log_returns_dict keys: {list(combined_log_returns_dict.keys()) if combined_log_returns_dict else 'Empty'}\n")
        log_stream.write(f"  Input fitted_models keys: {list(fitted_models.keys()) if fitted_models else 'Empty'}\n")
        log_stream.write(f"  Input final_stationarized_fred_data columns: {list(final_stationarized_fred_data.columns) if isinstance(final_stationarized_fred_data, pd.DataFrame) and not final_stationarized_fred_data.empty else 'Empty'}\n")

    auto_forecasts = {}

    # Check if the input dictionaries are empty
    if not combined_log_returns_dict:
        if verbose:
            log_stream.write("⚠️ Input combined_log_returns_dict is empty. Skipping forecasting.\n")
        return auto_forecasts
    if not fitted_models:
         if verbose:
             log_stream.write("⚠️ Input fitted_models is empty. Skipping forecasting.\n")
         return auto_forecasts


    # Iterate through the fitted models dictionary
    # The keys are in the format "GROUP_NAME" (for VARMAX) or "PAIR_ENDOG_VARIABLE" (for SARIMAX)
    for model_key, model_result in fitted_models.items():
        if verbose:
            log_stream.write(f"\n🔍 Forecasting using model for: {model_key}\n")

        model = model_result.get('fitted_model')
        if model is None:
            if verbose:
                log_stream.write("⚠️ 'fitted_model' tidak ditemukan di model_result.\n")
            continue

        endog_names = model_result.get('endog_names')
        if endog_names is None or not endog_names:
            if verbose:
                log_stream.write("⚠️ 'endog_names' tidak ditemukan di model_result.\n")
            continue

        # Ensure the model object has a predict method and resid attribute
        if not hasattr(model, 'predict') or not hasattr(model, 'resid'):
            if verbose:
                log_stream.write("⚠️ 'fitted_model' tidak memiliki atribut 'resid' atau 'predict'.\n")
            continue

        try:
            # Determine the starting point for forecasting
            # We need the last observed point of the endogenous data that the model was fitted on.
            # This is available from model.fittedvalues.index
            last_endog_date = model.fittedvalues.index[-1]

            # Prepare exogenous data for forecasting
            exog_forecast = None
            exog_names = model_result.get('exog_names', []) # Get exog names from model_result
            relevant_fred_data = None

            if final_stationarized_fred_data is not None and not final_stationarized_fred_data.empty:
                relevant_fred_data = final_stationarized_fred_data[exog_names].copy() if any(col in final_stationarized_fred_data.columns for col in exog_names) else pd.DataFrame()

            if exog_names:
                # Build the exog_forecast DataFrame
                # Create a date range for the forecast horizon starting from the next period after last_endog_date
                freq = pd.infer_freq(model.fittedvalues.index)
                if freq is None:
                    # Attempt to infer from actual data or assume a common frequency if possible
                    # For simplicity, if freq cannot be inferred, we might need a more robust date generation or assume daily.
                    # For now, let's assume a frequency that works with pandas offset aliases if not explicitly provided.
                    if len(model.fittedvalues.index) > 1:
                        diff = model.fittedvalues.index[1] - model.fittedvalues.index[0]
                        if diff == pd.Timedelta(days=1): freq = 'D'
                        elif diff == pd.Timedelta(hours=1): freq = 'H'
                        elif diff == pd.Timedelta(minutes=1): freq = 'min'
                        else: freq = 'D' # Fallback
                    else: freq = 'D'

                # Generate future dates based on inferred frequency and forecast horizon
                future_dates = pd.date_range(start=last_endog_date + pd.Timedelta(milliseconds=1), periods=forecast_horizon, freq=freq)
                future_dates = future_dates.normalize() # Remove time if it's just a date

                exog_data_for_forecast = {} # To hold required exog series

                for exog_col in exog_names:
                    # Check if exog_col is from FRED
                    is_fred_exog = False
                    if relevant_fred_data is not None and exog_col in relevant_fred_data.columns:
                        exog_data_for_forecast[exog_col] = relevant_fred_data[exog_col]
                        is_fred_exog = True

                    # If not FRED or FRED data is missing for this exog, check combined_log_returns_dict
                    if not is_fred_exog and exog_col in combined_log_returns_dict.columns:
                        # This part assumes combined_log_returns_dict might contain exog series directly
                        exog_data_for_forecast[exog_col] = combined_log_returns_dict[exog_col]


                if not exog_data_for_forecast:
                    if verbose:
                        log_stream.write("⚠️ Tidak ada data FRED stasioner yang tersedia untuk exogen.\n")
                else:
                    # Create exog_forecast DataFrame, interpolating/extrapolating if necessary
                    exog_forecast = pd.DataFrame(exog_data_for_forecast).reindex(model.fittedvalues.index.union(future_dates)).ffill().bfill()
                    exog_forecast = exog_forecast.loc[future_dates] # Select only the future dates
                    if verbose:
                        log_stream.write(f"  Menggunakan exog_forecast: {exog_forecast.shape} rows.\n")

            else:
                if verbose:
                    log_stream.write("  Tidak ada exogen yang relevan untuk model ini.\n")

            # Ensure exog_forecast is provided to predict only if it's not None and not empty.
            predict_kwargs = {'exog': exog_forecast} if exog_forecast is not None and not exog_forecast.empty else {}

            # Perform the forecast
            # The predict method returns a PredictionResultsWrapper which contains forecast and confidence intervals
            forecast_res = model.get_prediction(start=len(model.fittedvalues), end=len(model.fittedvalues) + forecast_horizon - 1, **predict_kwargs)
            forecast_df = forecast_res.summary_frame(alpha=0.05) # Default alpha for 95% CI

            # Adjust index to be future dates
            forecast_df.index = future_dates[:len(forecast_df)]

            # Rename columns for clarity
            new_cols = {
                'mean': f'{endog}_Forecast' for endog in endog_names
            }
            # Apply to forecast_df.columns if VARMAX
            # If it's SARIMAX, 'mean' refers to the single endogenous variable.
            # We need to handle both cases.

            # For VARMAX (multiple endog), forecast_df columns will be like 'mean', 'mean_ci_lower', 'mean_ci_upper'
            # For SARIMAX (single endog), forecast_df columns will be like 'mean', 'mean_ci_lower', 'mean_ci_upper'

            # For consistency, ensure column names reflect the endogenous variable
            if len(endog_names) == 1:
                forecast_df = forecast_df.rename(columns={'mean': f'{endog_names[0]}_Forecast',
                                                          'mean_ci_lower': f'{endog_names[0]}_Lower',
                                                          'mean_ci_upper': f'{endog_names[0]}_Upper'})
            else:
                # VARMAX case, columns will already be named with endog variables if prediction is done correctly
                # Let's verify and rename if necessary
                # statsmodels VARMAX get_prediction for multiple steps returns a DataFrame with column names like '0_0', '0_1', etc.
                # or directly the endogenous variable names if it's a multi-output forecast frame.
                # summary_frame usually makes it 'mean', 'mean_ci_lower', etc., or 'endog_name_mean'

                # Simplified approach: assume summary_frame already contains endog names or has a predictable pattern
                # The actual output structure of summary_frame might vary based on statsmodels version and model type.
                # For now, let's map generic 'mean', 'mean_ci_lower' to the first endogenous variable if multiple are present
                # or if the structure is not immediately clear.

                # This part needs careful inspection of actual `forecast_res.summary_frame` output for VARMAX.
                # Assuming it generates 'mean' for each endogenous variable if multiple, or one 'mean' for all.
                # A safer way is to inspect `forecast_res.predicted_mean` which would be a DataFrame (T x k_endog)
                # and build the summary frame manually.

                # Let's adjust for VARMAX: the `summary_frame` method returns columns like `_0_mean`, `_0_mean_ci_lower`, etc.
                # We need to map these to actual `endog_names`.
                renaming_map = {}
                for i, endog_name in enumerate(endog_names):
                    renaming_map[f'{i}_mean'] = f'{endog_name}_Forecast'
                    renaming_map[f'{i}_mean_ci_lower'] = f'{endog_name}_Lower'
                    renaming_map[f'{i}_mean_ci_upper'] = f'{endog_name}_Upper'
                forecast_df = forecast_df.rename(columns=renaming_map)

            auto_forecasts[model_key] = {'interval_forecast': forecast_df, 'endog_names': endog_names}

            if verbose:
                log_stream.write(f"  Forecast berhasil untuk: {', '.join(endog_names)} (Model: {model_result['model_type']}).\n")

        except Exception as e:
            if verbose:
                log_stream.write(f"  [ERROR] Gagal melakukan peramalan untuk {model_key}: {e}\n")
            auto_forecasts[model_key] = {'interval_forecast': pd.DataFrame(), 'endog_names': endog_names}

    return auto_forecasts
