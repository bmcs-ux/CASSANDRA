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
