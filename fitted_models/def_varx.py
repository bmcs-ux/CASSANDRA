
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

def fit_multi_timeframe_ensemble(all_data_dict):
    """
    all_data_dict berisi: {'D1': df_d1, 'H1': df_h1, 'M1': df_m1}
    """
    ensemble = {}
    
    # 1. Fit D1 (VARX 1 + FRED)
    ensemble['D1'] = fit_varx_logic(all_data_dict['D1'], lags=1, exog=fred_data)
    
    # 2. Fit H1 (VARX 5 + Cross Asset)
    ensemble['H1'] = fit_varx_logic(all_data_dict['H1'], lags=5, exog=cross_asset_data)
    
    # 3. Fit M1 (Kalman Setup)
    ensemble['M1'] = setup_kalman_filter(all_data_dict['M1'])
    
    return ensemble

def fit_varx_or_arx(log_stream, df_pair,
                    endog_cols = ["Close"],
                    exog_cols = None,
                    maxlags = 4,
                    criterion = "aic"):
    """
    Fit VARX if endog multivariate, otherwise fit SARIMAX (ARX) for single endog.
    Returns dict with model results and diagnostics.
    """
    # Debugging print statement to see what maxlags is received
    log_stream.write(f"[DEBUG] fit_varx_or_arx received maxlags: {maxlags}\n")

    result = {"model_type": None, "fitted_model": None, "summary": None, "R2": None, "lags_used": None}

    if exog_cols is None: exog_cols = []
    df_work = df_pair.copy()

    # 2. SHIFT EXOGENOUS (Mencegah Mencontek Masa Depan)
    if exog_cols:
        df_work[exog_cols] = df_work[exog_cols].shift(1)

    # 3. Handling data setelah shift
    use_cols = list(set(endog_cols + exog_cols))
    sub = df_work[use_cols].dropna()

    if len(sub) < maxlags + 10:
        log_stream.write(f"[WARN] Data terlalu sedikit setelah shift.\n")
        raise ValueError("Not enough observations.")

    endog = sub[endog_cols]
    exog = sub[exog_cols] if len(exog_cols) > 0 else None

    # VARMAX jika multi endog
    if endog.shape[1] > 1:
        best_p = None
        best_ic = np.inf
        for p in range(1, maxlags+1):
            try:
                m = VARMAX(endog, exog=exog, order=(p,0), trend="c")
                r = m.fit(disp=False, maxiter=300)
                lb = []
                for col in r.resid.columns:
                    lb.append(acorr_ljungbox(r.resid[col], lags=[10]).lb_pvalue.iloc[-1])
                lb_pvalue = np.mean(lb)

                current_ic = getattr(r, criterion)
                if lb_pvalue < 0.05:
                    current_ic += 1000 # Penalti agar model yang tidak valid secara statistik tidak terpilih

                if current_ic < best_ic:
                    best_ic = current_ic
                    best_res = r
                    best_p = p
            except Exception as e:
                log_stream.write(f"[VARMAX FAIL p={p}] {e}\n")
                continue

        if best_p is None:
            log_stream.write("[ERROR] Gagal fitting VARMAX.\n")
            raise RuntimeError("Gagal fitting VARMAX.")
        result["model_type"] = "VARMAX"
        result["fitted_model"] = best_res
        result["summary"] = best_res.summary()
        result["lags_used"] = best_p
        resid = best_res.resid
        r2s = {}
        for col in endog.columns:
            r2s[col] = 1 - np.nanvar(resid[col]) / np.nanvar(endog[col])
        result["R2"] = r2s
        return result

    # SARIMAX jika single endog
    else:
        y = endog.iloc[:,0]
        best_p = None
        best_ic = np.inf
        best_res = None
        for p in range(1, maxlags + 1):
            try:
                m = SARIMAX(y, exog=exog, order=(p,0,0))
                r = m.fit(disp=False)

                # 4. VALIDASI STATISTIK (Ljung-Box)
                # Model yang bagus harus memiliki p-value > 0.05 (artinya residu sudah acak)
                lb_test = acorr_ljungbox(r.resid, lags=[10])
                lb_pvalue = lb_test.lb_pvalue.iloc[-1]

                ic = getattr(r, criterion)
                # Beri pinalti berat jika residu masih berpola (Ljung-Box < 0.05)
                if lb_pvalue < 0.05:
                    ic += 2000

                if ic < best_ic:
                    best_ic = ic
                    best_p = p
                    best_res = r
            except Exception as e:
                log_stream.write(f"[VARMAX ERROR p={p}] {e}\n")
                continue

        if best_res is None:
            log_stream.write("[ERROR] Gagal fitting SARIMAX.\n")
            raise RuntimeError("Gagal fitting SARIMAX.")
        result["model_type"] = "SARIMAX(ARX)"
        result["fitted_model"] = best_res
        result["summary"] = best_res.summary()
        result["lags_used"] = best_p
        resid = best_res.resid
        result["R2"] = {col: 1 - (np.var(resid[col]) / np.var(endog[col])) for col in endog.columns}
        
        # Tambahkan ini di bagian akhir pengembalian result (sebelum return result)
        if result["fitted_model"] is not None:
            # Mengambil parameter (theta) awal untuk RLS
            result["initial_theta"] = result["fitted_model"].params.values
            # Mengambil matriks kovariansi awal (P)
            result["initial_P"] = result["fitted_model"].cov_params().values
    
            # Metadata penting untuk membangun Phi (matriks regressor) di VPS
            result["endog_names"] = endog_cols
            result["exog_names"] = exog_cols
            result["k_regressors"] = len(result["initial_theta"]) // len(endog_cols)

        return result


def setup_kalman_filter(df_m1):
    """
    Menyiapkan State Space Model untuk Kalman Filter M1.
    Fokus: Mengestimasi harga 'bersih' (Hidden State) dari observasi M1 yang noise.
    """
    # Kita asumsikan observasi adalah 'Close' price
    # State Vector [x] = [Price, Velocity]
    
    # Transition Matrix (F): Price_t = Price_t-1 + Velocity_t-1
    F = np.array([[1, 1],
                  [0, 1]])
    
    # Observation Matrix (H): Kita hanya melihat Price
    H = np.array([[1, 0]])
    
    # Process Noise (Q): Seberapa cepat trend bisa berubah
    Q = np.eye(2) * 0.0001 
    
    # Measurement Noise (R): Seberapa besar noise bid-ask di M1
    R = np.array([[np.var(df_m1['Close'].diff().dropna()) * 0.5]])
    
    # Initial State
    initial_state = np.array([df_m1['Close'].iloc[-1], 0])
    initial_P = np.eye(2) * 0.1

    return {
        "model_type": "KALMAN_STATEDRIVEN",
        "F": F,
        "H": H,
        "Q": Q,
        "R": R,
        "initial_state": initial_state,
        "initial_P": initial_P
    }

def run_ensemble_factory(log_stream, data_dict, fred_df):
    """
    Pabrik utama untuk membangun ensemble D1, H1, dan M1.
    """
    ensemble = {}
    
    # 1. D1 - Global Compass (Daily + Macro)
    # Gunakan lag kecil (1) agar tidak overfitting pada data harian yang sedikit
    log_stream.write("[PROCESS] Fitting D1 Model (VARX-RLS)...\n")
    ensemble['D1'] = fit_varx_or_arx(log_stream, data_dict['D1'], 
                                     exog_cols=fred_df.columns.tolist(), 
                                     maxlags=1)
    
    # 2. H1 - Tactical Radar (Hourly + Cross Asset)
    # Gunakan lag lebih dalam (5) untuk menangkap momentum intraday
    log_stream.write("[PROCESS] Fitting H1 Model (VARX-RLS)...\n")
    # Asumsikan exog_cols_h1 adalah list aset lain (XAU, DXY, dll)
    ensemble['H1'] = fit_varx_or_arx(log_stream, data_dict['H1'], 
                                     exog_cols=data_dict['H1'].columns.difference(['Close']).tolist(), 
                                     maxlags=5)
    
    # 3. M1 - Execution Sniper (1-Minute Kalman)
    log_stream.write("[PROCESS] Setting up M1 Kalman Filter...\n")
    ensemble['M1'] = setup_kalman_filter(data_dict['M1'])
    
    return ensemble
