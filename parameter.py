import os
import pandas as pd
from datetime import timedelta

# ---------------------------------------------------------
# RUNTIME & DIRECTORY CONFIGURATION
# ---------------------------------------------------------
# Menggabungkan logika ROOT_DIR dari VPS dan Colab
ROOT_DIR = os.getenv("HF_ROOT_DIR", os.getcwd())
if "/home/bimachasin86/VARX_REGRESION" in ROOT_DIR: # VPS Specific logic preservation
    ROOT_DIR = "/home/bimachasin86/VARX_REGRESION"

VPS_PARAM_DIR = ROOT_DIR
VPS_DATA_DIR = ROOT_DIR
VPS_SYNC_PATH = os.path.join(ROOT_DIR, 'vps_sync')
BASE_DATA_DIR = '/content/drive/MyDrive/books/CASSANDRA/base_data'
PKL_CACHE_DIR = '/content/.pkl'
DEBUG_MODE = True

# Storage Paths
FORECAST_OUTPUT_PATH = os.path.join(ROOT_DIR, 'vps_sync', 'restored_forecasts.pkl')
FRED_DATA_PATH = os.path.join(ROOT_DIR, 'vps_sync', 'final_fred_data.pkl')
FITTED_MODELS_PATH = os.path.join(ROOT_DIR, 'vps_sync', 'fitted_ensemble.pkl')
COLAB_URL_FILE_PATH = os.path.join(VPS_DATA_DIR, "colab_ngrok_url.txt")
LOCAL_CSV_FILEPATH = os.path.join(ROOT_DIR, 'data_base', 'combined_data_final_complete.csv')

# Cache filenames
MTF_BASE_DFS_PKL_NAME = 'mtf_base_dfs.pkl'
FRED_DF_PKL_NAME = 'fred_df.pkl'

# ---------------------------------------------------------
# CREDENTIALS & API KEYS
# ---------------------------------------------------------
MT5_LOGIN = os.getenv("MT5_LOGIN", 206905748)
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "Bima12345#")
MT5_SERVER = os.getenv("MT5_SERVER", "Exness-MT5Trial7")

FRED_API_KEY = os.getenv('FRED_API_KEY', '987d18495a386165f0be970f8a733562')
# VPS Placeholder reminder: "YOUR_FRED_API_KEY"

TRADE_ENGINE_API_KEY = os.getenv("TRADE_ENGINE_API_KEY", "bima_12345678")
COLAB_API_KEY_FOR_TRADE_ENGINE = os.getenv("COLAB_API_KEY_FOR_TRADE_ENGINE", TRADE_ENGINE_API_KEY)
COLAB_API_KEY_FOR_MONITOR = os.getenv("COLAB_API_KEY_FOR_MONITOR", TRADE_ENGINE_API_KEY)
TRADE_ENGINE_API_URL = os.getenv("TRADE_ENGINE_API_URL", "http://127.0.0.1:8081/receive_signal")

# ---------------------------------------------------------
# ASSET REGISTRY & PAIRS
# ---------------------------------------------------------
ASSET_REGISTRY = {
    'XAUGBP': {'symbol': 'XAUGBP', 'asset_class': 'forex', 'source': 'exness'},
    'GBPUSD': {'symbol': 'GBPUSD', 'asset_class': 'forex', 'source': 'exness'},
    'AUDUSD': {'symbol': 'AUDUSD', 'asset_class': 'forex', 'source': 'exness'},
    'USDCAD': {'symbol': 'USDCAD', 'asset_class': 'forex', 'source': 'exness'},
    'USDCHF': {'symbol': 'USDCHF', 'asset_class': 'forex', 'source': 'exness'},
    'USDJPY': {'symbol': 'USDJPY', 'asset_class': 'forex', 'source': 'exness'},
    'NZDUSD': {'symbol': 'NZDUSD', 'asset_class': 'forex', 'source': 'exness'},
    'EURAUD': {'symbol': 'EURAUD', 'asset_class': 'forex', 'source': 'exness'},
    'XAUUSD': {'symbol': 'XAUUSD', 'asset_class': 'commodities', 'source': 'exness'},
    'XAGUSD': {'symbol': 'XAGUSD', 'asset_class': 'commodities', 'source': 'exness'},
    'XAUAUD': {'symbol': 'XAUAUD', 'asset_class': 'commodities', 'source': 'exness'},
    'XNIUSD': {'symbol': 'XNIUSD', 'asset_class': 'commodities', 'source': 'exness'},
    'XAGAUD': {'symbol': 'XAGAUD', 'asset_class': 'commodities', 'source': 'exness'},
    'XAUEUR': {'symbol': 'XAUEUR', 'asset_class': 'commodities', 'source': 'exness'},
    'XAGEUR': {'symbol': 'XAGEUR', 'asset_class': 'commodities', 'source': 'exness'},
    'BTCUSD': {'symbol': 'BTCUSD', 'asset_class': 'crypto', 'source': 'exness'},
    'BTCXAU': {'symbol': 'BTCXAU', 'asset_class': 'crypto', 'source': 'exness'},
    'BTCXAG': {'symbol': 'BTCXAG', 'asset_class': 'crypto', 'source': 'exness'},
    'US500':  {'symbol': 'US500', 'asset_class': 'index', 'source': 'exness'},
    'DXY':     {'symbol': 'DX-Y.NYB', 'asset_class': 'index', 'source': 'yfinance'},
    'EFFRVOL': {'symbol': 'EFFRVOL', 'asset_class': 'macro', 'source': 'fred'},
    'T5YIE':   {'symbol': 'T5YIE', 'asset_class': 'macro', 'source': 'fred'},
}
def _derive_vps_mt5_symbol(asset_key, meta):
    source = meta.get('source')
    if source == 'fred':
        return None
    # Semua instrumen trading pada VPS menggunakan suffix broker `m`.
    # DXY tetap dipaksa ke simbol terminal VPS (DXYm), bukan simbol yfinance.
    if asset_key == 'DXY':
        return 'DXYm'
    return f"{asset_key}m"


PAIRS = {
    k: sym
    for k, v in ASSET_REGISTRY.items()
    for sym in [_derive_vps_mt5_symbol(k, v)]
    if sym is not None
}
ALL_SYMBOLS = list(PAIRS.values())

# Colab PAIRS derivation & list
COLAB_PAIRS_MAP = {k: v['symbol'] for k, v in ASSET_REGISTRY.items() if v['source'] != 'fred'}
ALL_SYMBOLS = list(COLAB_PAIRS_MAP.values())

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

# ---------------------------------------------------------
# DATA WINDOWS & TIMEFRAMES
# ---------------------------------------------------------
lookback_days = 2200
base_interval = '1d'
HF_LOOKBACK_DAYS = 3
HF_BASE_INTERVAL = "1m"
fred_lookback_days = 10 * 365

LOOKBACK_DAYS = {
    'D1': 600,
    'H1': 600,
    'M1': 600,
    'M5': 7,
    'M15': 7,
}

TF_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "60min",
    "H4": "240min"
}

MTF_INTERVALS = {
    'D1': '1d',
    'H1': '1h',
    'M1': '1m',
}

# ---------------------------------------------------------
# FRED SPECIFIC CONFIGURATION
# ---------------------------------------------------------
FRED_SERIES = {
    "EFFRVOL": "EFFRVOL",
    "T5YIE": "T5YIE",
    "S&P 500": "SP500",
    "Index Semi-Annual" :  "BAMLH0A0HYM2SYTW",
    "Markets Corporate Plus" : "BAMLEMRECRPIEMEASYTW",
    "Effective Federal Funds Volume" : "EFFRVOL",
    "Overnight Reverse Repurchase Agreements: Treas.." : "RRPONTSYD",
    "Market Yield on U.S. Treasury Securities at 10" : "DGS10",
    "5-Year Breakeven Inflation Rate: Treas..": "T5YIE",
    "Effective Federal Funds Rate" : "DFF"
}
# Colab derivation for FRED_SERIES
COLAB_FRED_SERIES = {k: v['symbol'] for k, v in ASSET_REGISTRY.items() if v['source'] == 'fred'}

FRED_TRANSFORM_POLICY = {
    'EFFRVOL': 'log_diff',
    'T5YIE': 'level_and_diff',
}

FRED_MISSING_THRESHOLD = 0.3
FRED_HYBRID_ALPHA = 0.5

# ---------------------------------------------------------
# MODEL PARAMETERS & GRANGER CAUSALITY
# ---------------------------------------------------------
maxlag_test = 5
alpha = 0.05
min_obs_for_granger = 50

maxlag_granger = 5
alpha_granger = 0.05
maxlag_varx = 5
MIN_OBS_FOR_GARCH = 100

VARX_ENDOG_GROUPS = {
    'FX_Majors': [
        'GBPUSD_Close_Log_Return',
        'AUDUSD_Close_Log_Return',
        'USDJPY_Close_Log_Return',
        'USDCAD_Close_Log_Return',
        'USDCHF_Close_Log_Return',
        'NZDUSD_Close_Log_Return'
    ],
    'FX_Crosses': [
        'EURAUD_Close_Log_Return'
    ],
    'Commodities': [
        'XAUUSD_Close_Log_Return',
        'XAGUSD_Close_Log_Return',
        'XAUGBP_Close_Log_Return',
        'XAUAUD_Close_Log_Return',
        'XAUEUR_Close_Log_Return',
        'XAGAUD_Close_Log_Return',
        'XAGEUR_Close_Log_Return',
        'XNIUSD_Close_Log_Return'
    ],
    'Crypto': [
        'BTCUSD_Close_Log_Return',
        'BTCXAU_Close_Log_Return',
        'BTCXAG_Close_Log_Return'
    ],
    'Risk_Index': [
        'US500_Close_Log_Return',
        'DXY_Close_Log_Return'
    ]
}
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

# ---------------------------------------------------------
# RLS PARAMETERS (HIGH-FREQUENCY MONITORING)
# ---------------------------------------------------------
FORGETTING_FACTOR = 0.999
RLS_INITIAL_P_DIAG = 1e2
RLS_INITIAL_THETA = 0.0

_RLS_DEVIATION_THRESHOLD = True
RLS_DEVIATION_THRESHOLD = 6.90
RLS_DEVIATION_CLOSE_ALL_THRESHOLD = 7.1

RLS_SCALING_FACTOR_SL = 0.25
RLS_SCALING_FACTOR_TP = 0.35
RLS_SNR_INCREASE_FACTOR = 0.05
RLS_TP_RR_MIN = 0.3
RLS_SL_MAX_MULTIPLIER = 2.2

_RLS_CONFIDENCE = False
RLS_MIN_UPDATES_FOR_CONFIDENCE = 40
RLS_CONFIDENCE_ALPHA = 0.4
RLS_CONFIDENCE_ENTRY_THRESHOLD = 0.40
RLS_MAX_PRED_VARIANCE_FOR_ENTRY = 25.0

RLS_RETURN_EMA_ALPHA = 0.35
RLS_RETURN_DEADBAND = 5e-5
RLS_RETURN_DIRECTION_EPSILON = 1e-5

RLS_VOLATILITY_WINDOW = 196
RLS_MIN_INNOVATION_SCALE = 0.5
RLS_DEVIATION_ADAPTIVE_STD_MULTIPLIER = 0.5

# ---------------------------------------------------------
# FORECASTING, RISK & EXECUTION
# ---------------------------------------------------------
forecast_horizon = 2
FORECAST_HORIZON = 5
CONFIDENCE_LEVEL = 0.95
VOLATILITY_RISK_THRESHOLD = 0.02

MAGIC_NUMBER = 202401
EQUITY = 1000
RISK_PER_TRADE_PCT = 0.1
K_ATR_STOP = 1.8
K_MODEL_STOP = 1.5
SNR_THRESHOLD = 0.1
TP_RR_RATIO = 1.5

CONSENSUS_WEIGHT_D1 = 0.4
CONSENSUS_WEIGHT_H1 = 0.5
CONSENSUS_WEIGHT_M15 = 0.2
CONSENSUS_THRESHOLD = 0.75

DCC_RISK_MULTIPLIER = 0.7
DCC_FLIP_EPS_MULTIPLIER = 0.2

MEAN_REVERSION_HIGH_Z = 1.5
MEAN_REVERSION_LOW_VOL_PREDVAR = 0.002

# Kalman execution filter defaults (M1)
KALMAN_F = [[1, 1], [0, 1]]
KALMAN_H = [[1, 0]]
KALMAN_Q = [[0.0001, 0.0], [0.0, 0.0001]]
KALMAN_R = [[0.001]]
KALMAN_INITIAL_STATE = [1.0, 0.0]
KALMAN_INITIAL_P = [[0.1, 0.0], [0.0, 0.1]]
KALMAN_ZSCORE_WINDOW = 120
KALMAN_FLIP_ZSCORE = 1.0

# ---------------------------------------------------------
# SAFEGUARDS & MISC
# ---------------------------------------------------------
NEWS = True
BLOCK_SIGNAL_FOR = {"US500", "US30", "DXY", "USDCAD", "USDJPY", "AUDUSD", "GBPUSD"}
BLOK_SIGNAL_FOR = BLOCK_SIGNAL_FOR

USE_LOCAL_CSV_FOR_PAIRS = False
ENABLE_INTERACTIVE_PREPROCESS_REVIEW = True

# Compatibility aliases
maxlag_test = maxlag_granger
alpha = alpha_granger
