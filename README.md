### Ringkasan Proyek CASSANDRA: Multi-Timeframe VARX-GARCH Ensemble

**Tujuan Utama Proyek:**
CASSANDRA adalah sistem perdagangan algoritmik berbasis statistik yang dirancang untuk menggabungkan model VARX (Vector Autoregression with Exogenous Variables), DCC-GARCH, dan Kalman Filter. Proyek ini mengadopsi arsitektur hibrida, di mana *Heavy Training* dilakukan di Google Colab, sementara anda juga bisa nenggunakan hasil models.pkl untuk *Real-time Adaptation* lakukan di VPS menggunakan Recursive Least Squares (RLS) atau algoritma adaptif lainnya.

**New update Strategi Engineering & Modifikasi Pipeline Tingkat Lanjut:**
Modifikasi utama pada pipeline standar adalah pengintegrasian sistem Multi-Timeframe (MTF) yang komprehensif, dengan pilar-pilar berikut:

1.  **Arsitektur Multi-Timeframe (Three-Layer Engine):** Sistem ini tidak lagi mengandalkan satu model tunggal, melainkan tiga lapisan spesialisasi yang dapat di sesuaikan dengan kebutuhan analisis:
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
    *   `preprocessing`: Modul untuk log-return, stasionaritas, dan transformasi FRED.
    *   `vps_sync`: Folder sinkronisasi untuk *pickle* model dan *status JSON*.
    *   `data_base`: Penyimpanan lokal CSV untuk *backtesting*.
    *   `fitted_models`: Folder modul untuk uji kausalitas dan konfigurasi fitting models
    *   `main.py`: *Entry point* utama untuk *training* di Colab.
    *   `parameter.py`: Pusat konfigurasi sistem.
    *   `requirements.txt`: Daftar dependensi library.

## Gambaran Umum Proyek: Pipeline Peramalan Keuangan Multi-Timeframe

Proyek ini mengimplementasikan pipeline peramalan keuangan multi-timeframe (MTF) yang kuat, dirancang untuk menganalisis berbagai instrumen keuangan (FX, Komoditas, Indeks) dan indikator makroekonomi (data FRED). Pipeline ini mengintegrasikan akuisisi data, pra-pemrosesan ekstensif, pemodelan ekonometrik canggih (Granger Causality, VARX/ARX, DCC-GARCH, Kalman Filter), dan peramalan, yang secara khusus disesuaikan untuk aplikasi di lingkungan Google Colab.

### Fitur Utama:

1.  **Desain Modular**: Pipeline ini distrukturkan ke dalam modul-modul yang berbeda dan dapat dimuat ulang (`parameter`, `raw`, `preprocessing`, `fitted_models`, `forecast`, `restored`), memfasilitasi pengembangan dan pemeliharaan yang mudah di Colab.
2.  **Penanganan Data Multi-Timeframe**: Mendukung pemrosesan data secara simultan di berbagai timeframe (misalnya, D1, H1, M1) untuk analisis pasar yang komprehensif dan inferensi kausal lintas-timeframe.
3.  **Akuisisi Data**: 
    *   **Data Pasangan Keuangan**: Menggunakan `raw.pair_raw.load_base_data_mtf` untuk mengambil data historis OHLCV untuk pasangan yang ditentukan (misalnya, GBPUSD, XAUUSD, US500).
    *   **Data Makroekonomi (FRED)**: Mengintegrasikan `raw.makro_raw.download_macro_data` untuk mengambil seri FRED yang relevan, penting untuk analisis fundamental dan penyertaan variabel eksogen dalam model.
4.  **Pra-pemrosesan Ekstensif**: 
    *   `preprocessing.log_return`: Menerapkan log return ke harga untuk stasioneritas.
    *   `preprocessing.fred_transform`: Mentransformasi data FRED (log-return, log-diff, level-and-diff) berdasarkan kebijakan.
    *   `preprocessing.handle_missing`: Mengelola nilai yang hilang dalam data FRED.
    *   `preprocessing.combine_data`: Menggabungkan berbagai aliran data.
    *   `preprocessing.stationarity_test`: Menguji dan menstasionerkan data deret waktu.
5.  **Pemodelan Ekonometrik Tingkat Lanjut**: 
    *   `fitted_models.granger`: Melakukan uji kausalitas Granger untuk mengidentifikasi variabel eksogen yang signifikan secara statistik untuk model VARX.
    *   `fitted_models.def_varx`: Mengimplementasikan fitting model VARX/ARX untuk peramalan variabel endogen.
    *   `fitted_models.kalman_filter`: Mengintegrasikan filter Kalman untuk pemodelan ruang-keadaan, terutama untuk data frekuensi yang lebih tinggi (misalnya, M1).
    *   `fitted_models.dcc_garch_process`: Memfitting model DCC-GARCH ke residu dari VARX/ARX untuk menangkap korelasi dan volatilitas bersyarat dinamis.
6.  **Peramalan & Restorasi**: 
    *   `forecast.auto_varx_forecast`: Menghasilkan peramalan multi-langkah ke depan dengan interval kepercayaan dari model VARX/ARX yang sudah di-fitting.
    *   `restored.restore_log_returns_to_price`: Mengkonversi peramalan log-return kembali ke peramalan harga yang dapat diinterpretasikan (OHLC).
7.  **Penanganan Error & Pencatatan yang Kuat**: Menggunakan wrapper `safe_run` untuk setiap langkah pipeline untuk menangani pengecualian secara elegan dan menyediakan pencatatan yang rinci, memastikan kelangsungan pipeline dan kemampuan diagnostik.
8.  **Konfigurasi melalui `parameter.py`**: Semua parameter penting, seperti kunci API FRED, ID seri, daftar pasangan, periode lookback, lag model, dan tingkat kepercayaan, dipusatkan dalam `parameter.py` untuk modifikasi yang mudah.

### Alur Pipeline:

Skrip `main.py` mengatur urutan berikut:

1.  **Inisialisasi**: Menyiapkan pencatatan dan mendefinisikan ID eksekusi.
2.  **Pengunduhan Data FRED**: Mengambil data makroekonomi, menerapkan pemeriksaan placeholder untuk kunci API.
3.  **Pemuatan Data MTF**: Mengunduh atau memuat data historis untuk timeframe D1, H1, dan M1.
4.  **Penyelarasan Data**: Menyelaraskan semua timeframe ke timestamp penutupan umum untuk memastikan horizon observasi yang konsisten.
5.  **Pra-pemrosesan**: Menerapkan log return, transformasi FRED, dan menangani data yang hilang di seluruh timeframe.
6.  **Uji Kausalitas Granger**: Mengidentifikasi variabel eksogen yang signifikan untuk model setiap timeframe.
7.  **Fitting Model (VARX/ARX)**: Memfitting model ekonometrik untuk timeframe D1 dan H1, memanfaatkan variabel eksogen yang teridentifikasi.
8.  **Pengaturan Filter Kalman**: Menginisialisasi filter Kalman untuk data M1.
9.  **Fitting DCC-GARCH**: Memfitting model DCC-GARCH ke residu dari model VARX/ARX (terutama untuk H1) untuk pemodelan volatilitas.
10. **Pengemasan untuk VPS**: Menserialisasi model yang di-fitting, harga aktual terakhir, dan peta eksogen untuk penyebaran atau penggunaan lebih lanjut.
11. **Peramalan & Restorasi**: Menghasilkan dan merestorasi peramalan harga dengan interval kepercayaan untuk tampilan dan analisis.

## Kegunaan Pipeline CASSANDRA

Pipeline CASSANDRA dirancang sebagai sistem peramalan keuangan yang komprehensif dan adaptif, dengan kegunaan utama sebagai berikut:

1.  **Analisis Pasar Multi-Timeframe yang Mendalam**: Dengan memproses data dari berbagai kerangka waktu (harian, jam, menit) secara bersamaan, CASSANDRA memungkinkan pemahaman yang lebih kaya tentang dinamika pasar. Ini membantu mengidentifikasi tren jangka panjang, pergerakan intraday, dan hubungannya, yang sering kali terlewatkan jika hanya menganalisis satu timeframe.
2.  **Identifikasi Hubungan Kausal (Granger Causality)**: Kemampuan untuk mengidentifikasi variabel makroekonomi atau aset lain yang 'menyebabkan' pergerakan harga aset target adalah kekuatan besar. Ini memberikan wawasan tentang pendorong pasar yang mendasari, bukan hanya korelasi.
3.  **Peramalan Harga yang Akurat dengan Interval Kepercayaan**: Dengan menggunakan model ekonometrik canggih seperti VARX/ARX dan Kalman Filter, pipeline ini dapat menghasilkan peramalan harga di masa depan. Yang lebih penting, adanya interval kepercayaan memungkinkan pengguna untuk memahami tingkat ketidakpastian dalam peramalan tersebut, yang sangat krusial dalam pengambilan keputusan.
4.  **Pemodelan Volatilitas (DCC-GARCH)**: Kemampuan untuk memodelkan volatilitas pasar sangat penting bagi manajemen risiko. DCC-GARCH tidak hanya memperkirakan volatilitas, tetapi juga korelasi antara aset, yang esensial untuk diversifikasi portofolio dan strategi lindung nilai.
5.  **Fleksibilitas dan Skalabilitas**: Desain modular dan penggunaan Colab memudahkan adaptasi pipeline untuk aset, indikator, atau model yang berbeda. Ini berarti CASSANDRA dapat terus berkembang seiring dengan kebutuhan analisis.
6.  **Pencatatan dan Debugging yang Efisien**: Fitur `safe_run` memastikan bahwa meskipun ada error di satu tahapan, pipeline dapat terus berjalan dan mencatat masalah tersebut. Ini mempercepat proses pengembangan dan pemeliharaan.

## Potensi Pemanfaatan dalam Bidang Ekonomi

Dalam bidang ekonomi, pipeline CASSANDRA memiliki potensi pemanfaatan yang signifikan:

1.  **Peramalan Makroekonomi**: Model ini dapat digunakan untuk meramalkan indikator makroekonomi penting seperti inflasi, suku bunga, pertumbuhan PDB, atau tingkat pengangguran, dengan menggabungkan data FRED yang relevan. Peramalan yang lebih akurat membantu pembuat kebijakan dalam perencanaan ekonomi.
2.  **Analisis Transmisi Kebijakan**: Dengan mengidentifikasi hubungan kausalitas (Granger), ekonom dapat mempelajari bagaimana perubahan dalam satu variabel ekonomi (misalnya, suku bunga yang ditetapkan bank sentral) mempengaruhi variabel lain (misalnya, investasi, konsumsi, atau harga aset). Ini krusial untuk mengevaluasi efektivitas kebijakan moneter atau fiskal.
3.  **Manajemen Risiko Ekonomi dan Keuangan**: Kemampuan CASSANDRA untuk memodelkan volatilitas dan korelasi antar-aset sangat relevan untuk bank sentral, lembaga keuangan, atau pemerintah dalam mengelola risiko sistemik, memantau stabilitas keuangan, atau merancang instrumen keuangan yang lebih tangguh.
4.  **Penelitian dan Pengembangan Model Ekonomi**: Kerangka kerja modular CASSANDRA dapat menjadi platform yang sangat baik bagi para peneliti untuk menguji hipotesis ekonomi baru, membandingkan kinerja model peramalan yang berbeda, atau mengintegrasikan teori ekonomi ke dalam model empiris.
5.  **Perencanaan Investasi dan Kebijakan Alokasi Aset**: Peramalan harga aset (misalnya, indeks saham, komoditas, nilai tukar mata uang) beserta interval kepercayaannya sangat berharga bagi investor institusi, manajer dana, atau bahkan pemerintah yang mengelola dana kekayaan negara, untuk membuat keputusan alokasi aset yang lebih terinformasi.
6.  **Peringatan Dini Krisis**: Dengan memantau pergerakan harga aset, volatilitas, dan hubungan kausal dengan indikator makroekonomi, CASSANDRA dapat membantu dalam membangun sistem peringatan dini untuk potensi krisis keuangan atau ekonomi.


**Status Sistem:** currently under development! CASSANDRA is an advanced version of the VARX_REGRESION Project. The latest version is experimental! Current Version: 2.0 (Multi-Timeframe Integrated).
