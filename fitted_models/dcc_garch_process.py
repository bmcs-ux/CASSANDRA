import pandas as pd
import numpy as np
from arch.univariate import GARCH, arch_model
from .dcc_garch import DCCGARCH # Use the custom DCCGARCH class, corrected import
import warnings
warnings.filterwarnings("ignore")

def fit_dcc_garch_to_residuals(log_stream, residuals_df):
    """
    Fits a DCC-GARCH model to a DataFrame of residuals.

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
        
        # Initialize and fit the DCC-GARCH model
        # You might need to adjust the model parameters (e.g., p, o, q for GARCH, P, Q for DCC)
        # based on your specific requirements or model selection criteria.
        # For demonstration, using a simple (1,1) GARCH for univariate and (1,1) for DCC
        
        # Define univariate GARCH specifications for each series
        garch_specs = {}
        for col in residuals_df.columns:
            # Fit a GARCH(1,1) to each series to get starting values or just use default
            # For simplicity, let's use a standard GARCH(1,1) spec
            garch_specs[col] = {'power': 2, 'p': 1, 'o': 0, 'q': 1}

        dcc_model = DCCGARCH(residuals_df, p=1, q=1)
        # dcc_model.resids_initial = residuals_df # Set initial residuals if the class expects it
        # You might need to manually set `resids_initial` or similar properties
        # if the DCCGARCH class requires them to be set before `fit`.
        
        # If DCCGARCH directly takes `resids` argument in fit or __init__
        # then the above line `dcc_model = DCCGARCH(residuals_df, p=1, q=1)` should be enough.
        
        # The `fit` method might take optimization options
        res = dcc_model.fit(disp='off')
        
        log_stream.write("[OK] DCC-GARCH model fitted successfully.")
        return res

    except Exception as e:
        log_stream.write(f"[ERROR] Failed to fit DCC-GARCH model: {e}")
        return None

# Helper function to extract residuals and ensure proper format for DCC-GARCH
def prepare_residuals_for_dcc_garch(log_stream, ensemble_results, log_returns_dict):
    """
    Extracts and prepares residuals from VARX/ARX models for DCC-GARCH fitting.
    Consolidates residuals from D1 and H1 timeframes.

    Args:
        log_stream (StringIO): Stream for logging.
        ensemble_results (dict): Dictionary of fitted VARX/ARX models by timeframe.
        log_returns_dict (dict): Dictionary of original log returns by timeframe.

    Returns:
        pd.DataFrame: DataFrame of combined residuals, or None if no valid residuals.
    """
    all_residuals = {} # Stores residuals as {endog_name_TF: Series}

    for tf, models_tf in ensemble_results.items():
        if tf in ['D1', 'H1']: # Focus on D1 and H1 for GARCH residuals
            if models_tf:
                for model_key, model_result in models_tf.items():
                    if 'fitted_model' in model_result and hasattr(model_result['fitted_model'], 'resid'):
                        fitted_model_obj = model_result['fitted_model']
                        endog_names_in_model = model_result.get('endog_names', [])

                        # Extract residuals for each endogenous variable in the model
                        for endog_name in endog_names_in_model:
                            if endog_name in fitted_model_obj.resid.columns:
                                resid_series = fitted_model_obj.resid[endog_name].dropna()
                                if not resid_series.empty:
                                    # Use a unique key like 'PAIR_Close_Log_Return_D1'
                                    all_residuals[f"{endog_name}_{tf}"] = resid_series
                                else:
                                    log_stream.write(f"[WARN] Residual series for {endog_name} in TF {tf} is empty after dropna. Skipping.")
                            else:
                                log_stream.write(f"[WARN] Column {endog_name} not found in residuals of {model_key} for TF {tf}. Skipping.")
                    else:
                        log_stream.write(f"[WARN] No fitted model or residuals attribute for {model_key} in TF {tf}. Skipping for DCC-GARCH preparation.")
            else:
                log_stream.write(f"[WARN] No models found for TF {tf} in ensemble_results. Skipping for DCC-GARCH preparation.")

    if not all_residuals:
        log_stream.write("[WARN] No valid residuals collected for DCC-GARCH fitting.")
        return None

    # Convert collected residuals into a single DataFrame
    residuals_df = pd.DataFrame(all_residuals)
    residuals_df = residuals_df.dropna() # Drop rows with any NaN after combining

    if residuals_df.empty:
        log_stream.write("[WARN] Combined residuals DataFrame is empty after dropping NaNs. Cannot fit DCC-GARCH.")
        return None
    
    log_stream.write(f"[INFO] Prepared residuals DataFrame for DCC-GARCH with shape: {residuals_df.shape}")
    return residuals_df

