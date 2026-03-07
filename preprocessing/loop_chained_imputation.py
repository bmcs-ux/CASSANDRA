import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that converts infinite values to NaN."""
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _fill_ohlc_from_close(df: pd.DataFrame, prefix: str) -> None:
    """Fill Open/High/Low for a symbol prefix using its Close when values are missing."""
    close_col = f"{prefix}_Close"
    for suffix in ("Open", "High", "Low"):
        col = f"{prefix}_{suffix}"
        if col in df.columns and close_col in df.columns:
            df[col] = df[col].fillna(df[close_col])


def impute_btc_cross_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Impute BTC/XAU and BTC/XAG with triangular relation:
    BTC/XAU = BTC/USD / XAU/USD
    BTC/XAG = BTC/USD / XAG/USD
    """
    imputed_df = df.copy()

    required = [
        "BTC/USD_Close",
        "XAU/USD_Close",
        "XAG/USD_Close",
        "BTC/XAU_Close",
        "BTC/XAG_Close",
    ]
    missing_cols = [c for c in required if c not in imputed_df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan untuk imputasi BTC cross: {missing_cols}")

    initial_btcxau = imputed_df["BTC/XAU_Close"].count()
    initial_btcxag = imputed_df["BTC/XAG_Close"].count()

    btcxau_synthetic = _safe_divide(imputed_df["BTC/USD_Close"], imputed_df["XAU/USD_Close"])
    btcxag_synthetic = _safe_divide(imputed_df["BTC/USD_Close"], imputed_df["XAG/USD_Close"])

    imputed_df["BTC/XAU_Close"] = imputed_df["BTC/XAU_Close"].fillna(btcxau_synthetic)
    imputed_df["BTC/XAG_Close"] = imputed_df["BTC/XAG_Close"].fillna(btcxag_synthetic)

    _fill_ohlc_from_close(imputed_df, "BTC/XAU")
    _fill_ohlc_from_close(imputed_df, "BTC/XAG")

    final_btcxau = imputed_df["BTC/XAU_Close"].count()
    final_btcxag = imputed_df["BTC/XAG_Close"].count()

    return imputed_df, {
        "BTC/XAU": {
            "initial": int(initial_btcxau),
            "final": int(final_btcxau),
            "added": int(final_btcxau - initial_btcxau),
        },
        "BTC/XAG": {
            "initial": int(initial_btcxag),
            "final": int(final_btcxag),
            "added": int(final_btcxag - initial_btcxag),
        },
    }


def impute_metals_from_btc_cross(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Impute XAU/USD and XAG/USD with inverse triangular relation:
    XAU/USD = BTC/USD / BTC/XAU
    XAG/USD = BTC/USD / BTC/XAG
    """
    imputed_df = df.copy()

    required = [
        "BTC/USD_Close",
        "BTC/XAU_Close",
        "BTC/XAG_Close",
        "XAU/USD_Close",
        "XAG/USD_Close",
    ]
    missing_cols = [c for c in required if c not in imputed_df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan untuk imputasi logam: {missing_cols}")

    initial_xau = imputed_df["XAU/USD_Close"].count()
    initial_xag = imputed_df["XAG/USD_Close"].count()

    xau_synthetic = _safe_divide(imputed_df["BTC/USD_Close"], imputed_df["BTC/XAU_Close"])
    xag_synthetic = _safe_divide(imputed_df["BTC/USD_Close"], imputed_df["BTC/XAG_Close"])

    imputed_df["XAU/USD_Close"] = imputed_df["XAU/USD_Close"].fillna(xau_synthetic)
    imputed_df["XAG/USD_Close"] = imputed_df["XAG/USD_Close"].fillna(xag_synthetic)

    _fill_ohlc_from_close(imputed_df, "XAU/USD")
    _fill_ohlc_from_close(imputed_df, "XAG/USD")

    final_xau = imputed_df["XAU/USD_Close"].count()
    final_xag = imputed_df["XAG/USD_Close"].count()

    return imputed_df, {
        "XAU/USD": {
            "initial": int(initial_xau),
            "final": int(final_xau),
            "added": int(final_xau - initial_xau),
        },
        "XAG/USD": {
            "initial": int(initial_xag),
            "final": int(final_xag),
            "added": int(final_xag - initial_xag),
        },
    }


def apply_loop_berantai_imputation(log_stream, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply chained-loop imputation for BTC cross and precious metals series."""
    log_stream.write("\n[INFO] Menjalankan imputasi Loop Berantai (BTC cross & logam mulia)...\n")

    imputed_df, btc_stats = impute_btc_cross_pairs(df)
    for symbol, stats in btc_stats.items():
        log_stream.write(
            f"  [INFO] {symbol}: Awal {stats['initial']} -> Akhir {stats['final']} (Tambah {stats['added']})\n"
        )

    imputed_df, metal_stats = impute_metals_from_btc_cross(imputed_df)
    for symbol, stats in metal_stats.items():
        log_stream.write(
            f"  [INFO] {symbol}: Awal {stats['initial']} -> Akhir {stats['final']} (Tambah {stats['added']})\n"
        )

    log_stream.write("[OK] Imputasi Loop Berantai selesai.\n")
    return imputed_df, {"btc_cross": btc_stats, "metals": metal_stats}
