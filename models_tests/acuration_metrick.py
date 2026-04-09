import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_forecast_metrics(y_true, y_pred):
    """
    Menghitung metrik akurasi peramalan: RMSE, MAE, dan MAPE.

    Args:
        y_true (array-like): Nilai aktual.
        y_pred (array-like): Nilai prediksi.

    Returns:
        dict: Sebuah dictionary yang berisi RMSE, MAE, dan MAPE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Hitung RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Hitung MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)

    # Hitung MAPE (Mean Absolute Percentage Error)
    # Hindari pembagian dengan nol
    non_zero_true = y_true[y_true != 0]
    non_zero_pred = y_pred[y_true != 0]

    if len(non_zero_true) > 0:
        mape = np.mean(np.abs((non_zero_true - non_zero_pred) / non_zero_true)) * 100
    else:
        mape = np.nan # Atau Anda bisa mengatur ke 0 atau mengeluarkan peringatan

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

print("Fungsi `calculate_forecast_metrics` telah didefinisikan.")

# Contoh penggunaan (opsional, bisa dijalankan jika Anda memiliki data y_true dan y_pred)
# y_actual = [10, 12, 11, 15, 13]
# y_predicted = [10.5, 11.8, 11.2, 14.5, 13.5]
# metrics = calculate_forecast_metrics(y_actual, y_predicted)
# print(f"Metrik Peramalan: {metrics}")
