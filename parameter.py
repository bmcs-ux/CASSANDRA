import os
from datetime import timedelta

# --- General ---
ROOT_DIR = '/content/drive/MyDrive/books/CASSANDRA/'
# Ensure VPS_SYNC_PATH uses ROOT_DIR
VPS_SYNC_PATH = os.path.join(ROOT_DIR, 'vps_sync')

# --- Data Acquisition & Preprocessing ---
# PAIRS for Yahoo Finance. For FX, typically use 'CURRENCYPAIR=X' (e.g., 'EURUSD=X')
# For Commodities: 'GC=F' (Gold), 'SI=F' (Silver), 'HG=F' (Copper), 'CL=F' (Crude Oil)
# For Indices: '^GSPC' (S&P 500), '^DJI' (Dow Jones), '^IXIC' (NASDAQ), 'DX-Y.NYB' (DXY)
PAIRS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'USDCAD=X',
    'USDCHF': 'USDCHF=X',
    'USDJPY': 'USDJPY=X',
    'NZDUSD': 'NZDUSD=X',
    'XAUUSD': 'GC=F', # Gold
    'XAGUSD': 'SI=F', # Silver
    'USOUSD': 'CL=F', # Crude Oil
    'SPX500': '^GSPC', # S&P 500
    'DJI': '^DJI', # Dow Jones
    'NDX': '^IXIC', # Nasdaq
    'DXY': 'DX-Y.NYB', # Dollar Index
    'CADJPY': 'CADJPY=X' # Example of a cross pair
}

# List of all symbols to check for uniqueness in data fetching
ALL_SYMBOLS = list(PAIRS.values())

# FRED API
FRED_API_KEY = os.getenv('FRED_API_KEY', 'YOUR_FRED_API_KEY') # Ambil dari environment variable
FRED_SERIES = {
    'FEDFUNDS': 'FEDFUNDS',             # Fed Funds Rate
    'CPALTT01USM657N': 'CPALTT01USM657N', # CPI All Items (Monthly)
    'PCEPILFE': 'PCEPILFE',             # Core PCE Price Index
    'UNRATE': 'UNRATE',                 # Unemployment Rate
    'GDP': 'GDP',                       # Gross Domestic Product
    'SP500': 'SP500'                    # S&P 500 (can be redundant if using ^GSPC)
}

# New parameters for local CSV usage
USE_LOCAL_CSV_FOR_PAIRS = False # Set to False to ensure download is attempted
LOCAL_CSV_FILEPATH = os.path.join(ROOT_DIR, 'data_base', 'combined_data_final_complete.csv') # Corrected path


# Timeframes and Lookback Periods
# YFinance uses '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
MTF_INTERVALS = {'D1': '1d', 'H1': '1h', 'M1': '1m', 'M5': '5m', 'M15': '15m'} # Added M5 and M15

# Lookback days for each timeframe (rough estimate for data fetching)
# Actual data used will depend on available history and stationarity requirements
LOOKBACK_DAYS = {
    'D1': 30,  # Reduced for testing, was 5*365
    'H1': 7,     # Reduced for testing, was 90
    'M1': 2,       # Reduced for testing, was 7 (yfinance max 7 days for 1m interval)
    'M5': 7,     # Added M5 lookback
    'M15': 7     # Added M15 lookback
}

# General Lookback for FRED (usually long-term)
fred_lookback_days = 10*365 # 10 years for FRED data

# Data Cleaning
# Missing data threshold for FRED. If a series has more than this % missing, it might be dropped.
FRED_MISSING_THRESHOLD = 0.3 # 30%

# --- Model Parameters ---
# Granger Causality Test
maxlag_granger = 5  # Max lags to test for Granger Causality
alpha_granger = 0.05 # Significance level for Granger test

# VARX Model
VARX_ENDOG_GROUPS = {
    'FX_Majors_D1': ['EURUSD_Close_Log_Return', 'GBPUSD_Close_Log_Return', 'AUDUSD_Close_Log_Return'],
    'Commodities_H1': ['XAUUSD_Close_Log_Return', 'XAGUSD_Close_Log_Return'],
    'Indices_D1': ['SPX500_Close_Log_Return', 'DJI_Close_Log_Return']
}
# Default maxlags for VARX/ARX models if not specified per TF
maxlag_varx = 5

# DCC-GARCH Model
MIN_OBS_FOR_GARCH = 100 # Minimum observations required to fit GARCH models

# Kalman Filter
KALMAN_CONFIG = {
    'n_dim_obs': 1, # Observed variable (e.g., price)
    'n_dim_state': 1, # Latent state (e.g., true price)
    'initial_state_mean': 0.0,
    'initial_state_covariance': 1.0,
    'transition_matrices': 1.0,
    'observation_matrices': 1.0,
    'transition_covariance': 0.01, # Process noise
    'observation_covariance': 0.1   # Measurement noise
}

# --- Forecasting & Risk Management ---
FORECAST_HORIZON = 5 # Number of steps to forecast ahead
CONFIDENCE_LEVEL = 0.95 # For confidence intervals (95%)
VOLATILITY_RISK_THRESHOLD = 0.02 # If forecast interval width exceeds this, trigger warning/deferral

# --- VPS Sync ---
# This is already defined at the top using ROOT_DIR
# FITTED_MODELS_PATH = os.path.join(VPS_SYNC_PATH, 'fitted_ensemble.pkl')

# --- Other --- # Add any other parameters

# Example of how parameters might be structured and accessed:
# from parameter import PAIRS, FRED_API_KEY
# print(PAIRS['EURUSD'])
