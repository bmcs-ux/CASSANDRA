
---

# CASSANDRA Project: Multi-Timeframe Quant sistem

## 1. Ringkasan Proyek
CASSANDRA adalah sistem perdagangan algoritmik berbasis statistik tingkat lanjut yang menggabungkan model **VARX**, **DCC-GARCH**, dan **Kalman Filter** ke dalam satu kesatuan sistem adaptif. Proyek ini dirancang dengan arsitektur hibrida untuk memaksimalkan efisiensi komputasi:
* **Heavy Training (Colab):** Optimasi parameter, uji kausalitas, dan inisialisasi koefisien.
* **Real-time Adaptation (VPS):** Eksekusi mandiri menggunakan **Recursive Least Squares (RLS)** untuk update koefisien tanpa *retraining* penuh.

## 2. Arsitektur Pipeline (Three-Layer Engine)
Sistem memproses data melalui tiga lapisan timeframe untuk menyaring *market noise*:
1.  **Layer D1:** Analisis makro FRED (Suku Bunga, Inflasi) via VARX ($maxlags=1$).
2.  **Layer H1:** Analisis korelasi dinamis antar-aset via VARX ($maxlags=5$) dan DCC-GARCH.
3.  **Layer M1:** Filtrasi volatilitas tinggi menggunakan Kalman Filter dan monitoring per menit.



## 3. Struktur Direktori Saat Ini
Sistem telah dimodularisasi untuk memisahkan antara logika perdagangan, simulasi, dan manajemen data:

```text
.
├── adapters/               # Layer Abstraksi API
│   ├── mt5_adapter.py      # Adapter untuk Live Trading (MT5/mt5linux)
│   └── dummy_MetaTrader5.py# Simulator untuk Backtest di Colab
├── monitoring/             # Logika Runtime VPS
│   └── monitor_for_vps.py  # Script monitoring & adaptasi RLS real-time
├── data_base/              # Partitioned Data Lake (Parquet)
│   ├── asset_class=forex/symbol=AUDUSD/timeframe=M1/data_2026_03.parquet
│   └── ... (Partisi otomatis berdasarkan Asset, Symbol, & Timeframe)
├── fitted_models/          # Core Engine & Estimator
│   ├── dcc_garch.py        # Estimasi Korelasi Dinamis
│   ├── kalman_filter.py    # State-Space Filtering
│   └── transformer/        # Deep Learning untuk prediksi perilaku masa depan
├── backtest/               # Simulasi & Validasi
│   └── replay.py           # Engine untuk Simulated Replay menggunakan dummy adapter
├── preprocessing/          # Transformasi Data & Stationarity Test
├── raw/                    # Data Ingestion (FRED & Broker API)
├── vps_sync/               # Sinkronisasi Artefak Model (.pkl)
└── main.py                 # Orchestrator Utama (Training & Deployment)
```

## 4. Fitur Unggulan
* **Modular Adapters:** Memungkinkan transisi mulus antara lingkungan *Live* (VPS) dan *Backtest* (Colab) hanya dengan menukar file di folder `adapters`.
* **Partitioned Parquet Storage:** Penyimpanan data berbasis kolom yang dioptimalkan untuk akses cepat selama proses *heavy backtesting* dan pelatihan Transformer.
* **Transformer Integration:** Terletak di `fitted_models/transformer/`, digunakan untuk mempelajari pola dari metrik monitoring (RLS & Kalman) guna memprediksi anomali atau peluang di masa depan.
* **Dynamic Exogenous Mapping:** Otomatisasi pemilihan variabel eksternal berdasarkan hasil uji **Granger Causality**.

## 5. Alur Deployment (Hybrid Model)
1.  **Fase Inisialisasi:** `main.py` menjalankan pipeline lengkap di Colab, mengunduh data makro, melakukan fitting, dan menghasilkan `vps_sync/fitted_ensemble.pkl`.
2.  **Fase Validasi:** `backtest/replay.py` memvalidasi model menggunakan `adapters/dummy_MetaTrader5.py` terhadap data historis di `data_base`.
3.  **Fase VPS:** `monitoring/monitor_for_vps.py` dimuat di server, menggunakan `adapters/mt5_adapter.py` untuk berinteraksi dengan pasar secara real-time dengan update adaptif RLS.

---

## 6. Kegunaan & Potensi Ekonomi
CASSANDRA melampaui trading bot biasa dengan menyediakan:
* **Analisis Transmisi Kebijakan:** Memetakan bagaimana data FRED mempengaruhi volatilitas mata uang secara statistik.
* **Sistem Peringatan Dini:** Deteksi dini perubahan korelasi antar aset (misal: Emas vs USD) melalui DCC-GARCH.
* **State-Space Modeling:** Mengidentifikasi "nilai wajar" aset di tengah kebisingan pasar menit-ke-menit melalui Kalman Filter.

---
**Catatan Teknis:** Pastikan `parameter.py` telah dikonfigurasi dengan API Key FRED yang valid dan path direktori yang sesuai dengan environment (Colab vs VPS).

---
**Status Sistem:** currently under development! CASSANDRA is an advanced version of the VARX_REGRESION Project. The latest version is experimental! Current Version: 2.1 (Backtest end to end and tranformer Integrated).
