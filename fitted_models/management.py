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