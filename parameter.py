import os
from datetime import timedelta

# --- General ---
ROOT_DIR = '/content/drive/MyDrive/books/CASSANDRA/'
VPS_SYNC_PATH = os.path.join(ROOT_DIR, 'vps_sync')

# --- Data Acquisition & Preprocessing ---
ASSET_REGISTRY = {
    'GBPUSD': {'symbol': 'GBPUSD', 'asset_class': 'forex', 'source': 'exness'},
    'AUDUSD': {'symbol': 'AUDUSD', 'asset_class': 'forex', 'source': 'exness'},
    'USDCAD': {'symbol': 'USDCAD', 'asset_class': 'forex', 'source': 'exness'},
    'USDCHF': {'symbol': 'USDCHF', 'asset_class': 'forex', 'source': 'exness'},
    'USDJPY': {'symbol': 'USDJPY', 'asset_class': 'forex', 'source': 'exness'},
    'NZDUSD': {'symbol': 'NZDUSD', 'asset_class': 'forex', 'source': 'exness'},
    'XAUUSD': {'symbol': 'XAUUSD', 'asset_class': 'commodities', 'source': 'exness'},
    'XAGUSD': {'symbol': 'XAGUSD', 'asset_class': 'commodities', 'source': 'exness'},
    'US500': {'symbol': 'US500', 'asset_class': 'index', 'source': 'exness'},
    'DXY': {'symbol': 'DX-Y.NYB', 'asset_class': 'index', 'source': 'yfinance'},
    'EFFRVOL': {'symbol': 'EFFRVOL', 'asset_class': 'macro', 'source': 'fred'},
    'T5YIE': {'symbol': 'T5YIE', 'asset_class': 'macro', 'source': 'fred'},
}

PAIRS = {k: v['symbol'] for k, v in ASSET_REGISTRY.items() if v['source'] != 'fred'}
ALL_SYMBOLS = list(PAIRS.values())
BASE_DATA_DIR = '/content/drive/MyDrive/books/CASSANDRA/data_base'
# Instrumen tambahan khusus untuk workflow imputasi loop berantai.
# Disimpan dalam format func_pair_name/url_segment agar kompatibel dengan downloader berbasis Exness.
IMPUTATION_SPECIAL_ASSETS = [
    {"func_pair_name": "XAU/GBP", "url_segment": "XAUGBP"},
    {"func_pair_name": "XAU/AUD", "url_segment": "XAUAUD"},
    {"func_pair_name": "GBP/USD", "url_segment": "GBPUSD"},
    {"func_pair_name": "XNI/USD", "url_segment": "XNIUSD"},
    {"func_pair_name": "XAG/AUD", "url_segment": "XAGAUD"},
    {"func_pair_name": "XAU/EUR", "url_segment": "XAUEUR"},
    {"func_pair_name": "XAG/EUR", "url_segment": "XAGEUR"},
    {"func_pair_name": "BTC/USD", "url_segment": "BTCUSD"},
    {"func_pair_name": "BTC/XAU", "url_segment": "BTCXAU"},
    {"func_pair_name": "BTC/XAG", "url_segment": "BTCXAG"},
    {"func_pair_name": "EUR/AUD", "url_segment": "EURAUD"},
]

# Direktori cache/save artefak data. Base data MTF kini dipersist ke Parquet, sedangkan cache pickle lama tetap tersedia untuk kompatibilitas.
PKL_CACHE_DIR = '/content/.pkl'
MTF_BASE_DFS_PKL_NAME = 'mtf_base_dfs.pkl'
FRED_DF_PKL_NAME = 'fred_df.pkl'

# Legacy-style defaults used by several modules
lookback_days = 2200
base_interval = '1d'

# FRED API
# Catatan: tetap bisa dioverride dari environment agar aman di deployment.
# Ganti 'YOUR_ACTUAL_FRED_API_KEY_HERE' dengan kunci FRED API Anda yang sebenarnya.
# Anda bisa mendapatkannya dari https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.getenv('FRED_API_KEY', '987d18495a386165f0be970f8a733562')
FRED_SERIES = {k: v['symbol'] for k, v in ASSET_REGISTRY.items() if v['source'] == 'fred'}

FRED_TRANSFORM_POLICY = {
    #'BAMLH0A0HYM2SYTW': 'log_return',
    #'BAMLEMRECRPIEMEASYTW': 'log_return',
    'EFFRVOL': 'log_diff',
    #'RRPONTSYD': 'log_diff',
    #'GDP': 'level_and_diff', # Changed from 'DGS10': 'level_and_diff'
    'T5YIE': 'level_and_diff',
    #'DFF': 'level_and_diff',
    #'DGS10': 'level_and_diff', # Added policy for DGS10
}

# Local CSV
USE_LOCAL_CSV_FOR_PAIRS = False
LOCAL_CSV_FILEPATH = os.path.join(ROOT_DIR, 'data_base', 'combined_data_final_complete.csv')

# Multi-timeframe
MTF_INTERVALS = {
    'D1': '1d',
    'H1': '1h',
    'M1': '1m',
}

LOOKBACK_DAYS = {
    'D1': 600,
    'H1': 600,
    'M1': 600,
    'M5': 7,
    'M15': 7,
}

# FRED lookback
fred_lookback_days = 10 * 365
FRED_MISSING_THRESHOLD = 0.3

# Interactive review sebelum PREPROCESSING MTF.
# False sebagai default agar eksekusi CI/non-interaktif tidak terblokir input().
ENABLE_INTERACTIVE_PREPROCESS_REVIEW = True

# Placeholder konfigurasi untuk metode imputasi FRED yang diusulkan.
FRED_HYBRID_ALPHA = 0.5

# --- Model Parameters ---
maxlag_granger = 5
alpha_granger = 0.05

VARX_ENDOG_GROUPS = {
    'FX_Majors': ['GBPUSD_Close_Log_Return', 'AUDUSD_Close_Log_Return', 'USDJPY_Close_Log_Return'],
    'Commodities': ['XAUUSD_Close_Log_Return', 'XAGUSD_Close_Log_Return'],
    'Risk': ['US500_Close_Log_Return', 'DXY_Close_Log_Return'],
}
maxlag_varx = 5
MIN_OBS_FOR_GARCH = 100

KALMAN_CONFIG = {
    'n_dim_obs': 1,
    'n_dim_state': 1,
    'initial_state_mean': 0.0,
    'initial_state_covariance': 1.0,
    'transition_matrices': 1.0,
    'observation_matrices': 1.0,
    'transition_covariance': 0.01,
    'observation_covariance': 0.1,
}

# --- Forecasting & Risk Management ---
FORECAST_HORIZON = 5
CONFIDENCE_LEVEL = 0.95
VOLATILITY_RISK_THRESHOLD = 0.02

# --- Compatibility aliases for modules that still use legacy names ---
maxlag_test = maxlag_granger
alpha = alpha_granger
