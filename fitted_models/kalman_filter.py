import numpy as np


def setup_kalman_filter(log_stream, df_m1):
    """
    Menyiapkan State Space Model untuk Kalman Filter M1.
    Fokus: Mengestimasi harga 'bersih' (hidden state) dari observasi M1 yang noisy.
    """
    if df_m1 is None or df_m1.empty or 'Close' not in df_m1.columns:
        if log_stream is not None:
            log_stream.write("[WARN] Data M1 tidak valid untuk setup Kalman filter.\n")
        return None

    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.eye(2) * 0.0001

    close_diff = df_m1['Close'].diff().dropna()
    measurement_var = np.var(close_diff) if not close_diff.empty else 1e-6
    R = np.array([[measurement_var * 0.5]])

    initial_state = np.array([df_m1['Close'].iloc[-1], 0])
    initial_P = np.eye(2) * 0.1

    return {
        "model_type": "KALMAN_STATEDRIVEN",
        "F": F,
        "H": H,
        "Q": Q,
        "R": R,
        "initial_state": initial_state,
        "initial_P": initial_P,
    }
