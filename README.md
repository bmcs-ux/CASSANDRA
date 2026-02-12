### Ringkasan Proyek CASSANDRA: Multi-Timeframe VARX-GARCH Ensemble

**Tujuan Utama Proyek:**
CASSANDRA adalah sistem perdagangan algoritmik berbasis statistik yang dirancang untuk menggabungkan model VARX (Vector Autoregression with Exogenous Variables), DCC-GARCH, dan Kalman Filter. Proyek ini mengadopsi arsitektur hibrida, di mana *Heavy Training* dilakukan di Google Colab, sementara *Real-time Adaptation* dilakukan di VPS menggunakan Recursive Least Squares (RLS).

**Strategi Engineering & Modifikasi Pipeline Tingkat Lanjut:**
Modifikasi utama pada pipeline standar adalah pengintegrasian sistem Multi-Timeframe (MTF) yang komprehensif, dengan pilar-pilar berikut:

1.  **Arsitektur Multi-Timeframe (Three-Layer Engine):** Sistem ini tidak lagi mengandalkan satu model tunggal, melainkan tiga lapisan spesialisasi:
    *   **Layer D1 (Global Compass):** Mengintegrasikan data makro FRED (Suku Bunga, Inflasi, S&P500) dengan VARX (menggunakan `maxlags=1`) untuk menentukan arah tren jangka panjang.
    *   **Layer H1 (Tactical Radar):** Berfokus pada hubungan Cross-Asset (misalnya, pengaruh Emas terhadap AUD/USD) menggunakan VARX (menggunakan `maxlags=5`) untuk menangkap momentum intraday. DCC-GARCH digunakan di sini untuk menghitung matriks korelasi dinamis dan volatilitas.
    *   **Layer M1 (Execution Sniper):** Menggunakan Kalman Filter untuk menyaring *market noise* dan secara kontinu mengestimasi "harga asli" di tengah volatilitas menit ke menit.

2.  **Mekanisme Training-to-Deployment (Colab to VPS):** Proses kalkulasi berat dipisahkan:
    *   **Colab Side:** Melakukan optimasi parameter, uji kausalitas Granger, dan inisialisasi koefisien model. Hasilnya disimpan dalam `fitted_models.pkl`.
    *   **VPS Side:** Memuat koefisien dari Colab dan menggunakan RLS (Recursive Least Squares) dengan *forgetting factor* ($\lambda$) untuk mengupdate koefisien secara rekursif setiap kali ada bar harga baru, tanpa melakukan *retraining* dari nol.

3.  **Pipeline Data & Preprocessing Modular:**
    *   **Automated Stationarity:** Melakukan uji ADF/KPSS dan menerapkan *differencing* secara otomatis hanya pada kolom yang tidak stasioner.
    *   **Dynamic Exogenous Mapping:** Menggunakan hasil Granger Causality untuk membangun `exog_map`, memastikan VARX hanya memasukkan variabel eksternal yang terbukti memiliki pengaruh signifikan secara statistik, mengurangi *dimensionality curse*.

4.  **Forecasting & Volatility Guard:** Sistem ini tidak hanya memprediksi harga (Log Return) tetapi juga Confidence Interval menggunakan residu dari model GARCH. Jika lebar interval melebihi `VOLATILITY_RISK_THRESHOLD`, sistem akan menunda eksekusi (Filter Risiko).

**Catatan Penting:**
*   **Log-Return Restoration:** Model bekerja di ruang log-return untuk stabilitas statistik. Fungsi `restore_log_returns_to_price` diimplementasikan untuk mengembalikan prediksi ke skala harga asli (OHLC) sebelum dikirim ke platform perdagangan.
*   **Struktur Folder Relevan:**
    *   `raw`: direktori modul untuk memproses sumber data *using download, api or local csv*
    *   `preprocesing`: Modul untuk log-return, stasionaritas, dan transformasi FRED.
    *   `vps_sync`: Folder sinkronisasi untuk *pickle* model dan *status JSON*.
    *   `data_base`: Penyimpanan lokal CSV untuk *backtesting*.
    *   `fitted_models`: Folder modul untuk uji kausalitas dan konfigurasi fitting models
    *   `main.py`: *Entry point* utama untuk *training* di Colab.
    *   `parameter.py`: Pusat konfigurasi sistem.
    *   `requirements.txt`: Daftar dependensi library.

**Status Sistem:** currently under development! CASSANDRA is an advanced version of the VARX_REGRESION Project. The latest version is experimental! Current Version: 2.0 (Multi-Timeframe Integrated).
