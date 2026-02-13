# Hasil Tinjauan Codebase

Berikut empat tugas yang diajukan berdasarkan temuan saat meninjau codebase:

1. **Perbaiki salah ketik**
   - Temuan: berkas inisialisasi paket tertulis `preprocessing/ __init__.py` (ada spasi di awal nama berkas).
   - Tugas: ganti nama menjadi `preprocessing/__init__.py` agar paket Python terdeteksi secara konsisten.

2. **Perbaiki bug**
   - Temuan: pada konfigurasi path, kode menggunakan `sys.sys.path.append(ROOT_DIR)`.
   - Tugas: ubah menjadi `sys.path.append(ROOT_DIR)` karena atribut `sys.sys` tidak valid.

3. **Perbaiki komentar / ketidaksesuaian dokumentasi**
   - Temuan: dokumentasi fungsi `check_data_freshness` menjelaskan toleransi M1 sekitar 5–10 menit, tetapi implementasi sebelumnya memakai 24 jam.
   - Tugas: samakan implementasi dengan dokumentasi dengan menetapkan toleransi M1 menjadi 10 menit.

4. **Tingkatkan pengujian**
   - Temuan: belum ada pengujian otomatis untuk perilaku inti `combine_log_returns`.
   - Tugas: tambahkan unit test untuk memverifikasi:
     - mode `return_type='dict'` membuat alias kolom `Log_Return`,
     - mode `return_type='df'` memberi prefix nama pasangan secara benar.


## Tinjauan Tambahan (Integrasi Exness)

1. **Perbaiki salah ketik**
   - Temuan: frasa seperti "mecari/mengahasil" muncul pada instruksi operasional internal.
   - Tugas: standarkan menjadi "mencari/menghasilkan" pada dokumentasi operasional agar tidak membingungkan implementasi.

2. **Perbaiki bug**
   - Temuan: fallback `pair_raw` sebelumnya masih mengandalkan sumber non-Exness saat CSV lokal tidak ditemukan.
   - Tugas: ubah fallback menjadi pengunduhan tick Exness (aturan bulanan untuk bulan lampau dan harian untuk bulan berjalan), lalu resample ke OHLC timeframe pipeline.

3. **Perbaiki komentar/dokumentasi**
   - Temuan: dokumentasi loading lokal belum menjelaskan pola nama file timeframe (`_m1`, `_h1`, `_d1`) dan aturan URL Exness harian/bulanan.
   - Tugas: perbarui docstring helper resolver URL/path agar konsisten dengan implementasi.

4. **Tingkatkan pengujian**
   - Temuan: belum ada test yang memverifikasi fallback Exness serta aturan pembentukan URL bulanan+harian.
   - Tugas: tambahkan unit test untuk `_build_exness_urls` dan fallback `load_base_data_mtf` dengan mock respons ZIP tick Exness.


## Tinjauan Tambahan (Stabilitas Orchestrator)

1. **Perbaiki salah ketik**
   - Temuan: pesan log campuran istilah Inggris/Indonesia membuat troubleshooting kurang konsisten.
   - Tugas: standarkan frasa log error utama (contoh: "cannot unpack" dipadankan dengan konteks Indonesia) agar mudah ditelusuri.

2. **Perbaiki bug**
   - Temuan: proses orchestrator gagal `cannot unpack non-iterable NoneType object` ketika `main.main()` tidak mengembalikan tuple yang diharapkan.
   - Tugas: tambahkan validasi bentuk output `main.main()` dan guard pada titik unpack hasil `safe_run` di pipeline utama.

3. **Perbaiki komentar/dokumentasi**
   - Temuan: kontrak fungsi `safe_run` (menyisipkan `log_stream` sebagai argumen pertama) belum terdokumentasi kuat untuk semua module helper.
   - Tugas: perjelas docstring/komentar fungsi helper agar signature modul konsisten dengan pemanggilan.

4. **Tingkatkan pengujian**
   - Temuan: belum ada test yang menutupi skenario `main.main()` mengembalikan nilai tidak valid dan setup Kalman saat data M1 kosong.
   - Tugas: tambahkan unit test untuk guard orchestrator dan guard setup Kalman.


## Tinjauan Tambahan (Hybrid Freshness & Common Close)

1. **Perbaiki salah ketik**
   - Temuan: istilah "time frame" dan "timeframe" masih tercampur pada catatan internal.
   - Tugas: standarkan istilah menjadi "timeframe" di seluruh dokumentasi pipeline.

2. **Perbaiki bug**
   - Temuan: pipeline berhenti total saat M1 stale meski tujuan utama adalah training model hybrid lintas timeframe.
   - Tugas: ubah dari hard-stop menjadi alignment/cutting data ke `common close` lintas timeframe agar training tetap berjalan.

3. **Perbaiki komentar/dokumentasi**
   - Temuan: komentar fungsi freshness sebelumnya menyiratkan kontrol stop-run, padahal kebutuhan aktual adalah monitoring + alignment.
   - Tugas: revisi docstring/komentar agar mencerminkan perilaku baru (informational check + common-close alignment).

4. **Tingkatkan pengujian**
   - Temuan: belum ada test untuk skenario stale M1 namun pipeline tetap lanjut lewat alignment common-close.
   - Tugas: tambahkan unit/integration test untuk verifikasi pemotongan index semua TF ke cutoff yang sama.


## Tinjauan Tambahan (Stabilitas Preprocess & FRED Logging)

1. **Perbaiki salah ketik**
   - Temuan: istilah `preprosesing`/`preprocessing` masih bercampur pada catatan operasional.
   - Tugas: standarkan istilah menjadi `preprocessing` agar konsisten di log dan dokumentasi.

2. **Perbaiki bug**
   - Temuan: `combined_df = ... or pd.DataFrame()` memicu error ambiguity pada objek `DataFrame`.
   - Tugas: ubah fallback menjadi pemeriksaan tipe eksplisit (`isinstance(..., pd.DataFrame)`) tanpa evaluasi truthy DataFrame.

3. **Perbaiki komentar/dokumentasi**
   - Temuan: inspeksi FRED mengasumsikan semua index punya atribut `.tz`, padahal bisa `Index` non-datetime.
   - Tugas: perbarui logika inspeksi dan komentarnya agar aman untuk index non-datetime.

4. **Tingkatkan pengujian**
   - Temuan: belum ada test yang menutup kasus fallback preprocess ketika hasil `Combine DF` gagal.
   - Tugas: tambahkan test untuk memastikan preprocess tetap mengembalikan tuple valid walau hasil DataFrame tidak tersedia.


## Tinjauan Tambahan (Robustness Type-Check Pipeline)

1. **Perbaiki salah ketik**
   - Temuan: istilah `symbol` dan `simbol` bercampur pada catatan konfigurasi pair.
   - Tugas: standarkan istilah menjadi `simbol` pada dokumentasi internal berbahasa Indonesia.

2. **Perbaiki bug**
   - Temuan: beberapa jalur mengasumsikan objek selalu DataFrame dan langsung mengakses `.empty`, padahal pada kasus tertentu bisa berupa `list`.
   - Tugas: tambahkan guard tipe (`isinstance(..., pd.DataFrame)`) sebelum akses `.empty` agar pipeline tidak crash dengan error `'list' object has no attribute 'empty'`.

3. **Perbaiki komentar/dokumentasi**
   - Temuan: kontrak input antar tahap (khususnya `log_returns` dan pool eksogen MTF) belum menegaskan tipe data yang diharapkan.
   - Tugas: perbarui komentar/docstring agar eksplisit menyatakan tipe yang valid di setiap tahap.

4. **Tingkatkan pengujian**
   - Temuan: belum ada test untuk validasi konfigurasi pair `US500` dan ketahanan guard tipe.
   - Tugas: tambahkan unit test untuk memastikan konfigurasi pair selaras dan guard kompatibilitas tetap aktif.


## Tinjauan Tambahan (Granger & Forecast Contract Consistency)

1. **Perbaiki salah ketik**
   - Temuan: istilah `empety` masih muncul pada laporan error operasional.
   - Tugas: standarkan menjadi `empty` pada dokumentasi troubleshooting.

2. **Perbaiki bug**
   - Temuan: alur granger dan pipeline masih mengasumsikan semua nilai di `log_returns` selalu DataFrame, sehingga saat tipe menyimpang muncul error `.empty` pada objek non-DataFrame.
   - Tugas: terapkan guard `isinstance(..., pd.DataFrame)` sebelum akses `.empty` di semua titik kritis.

3. **Perbaiki komentar/dokumentasi**
   - Temuan: kontrak fungsi `auto_varx_forecast` berbeda dengan kontrak `safe_run` (yang menyisipkan `log_stream`), namun belum dijelaskan di komentar.
   - Tugas: dokumentasikan bahwa `auto_varx_forecast` dipanggil langsung (tanpa `safe_run`) agar parameter tidak bergeser.

4. **Tingkatkan pengujian**
   - Temuan: belum ada test untuk memastikan hasil Granger diparse via `identify_significant_exog` dan pemanggilan forecasting tidak miss-argument.
   - Tugas: tambahkan test kontrak fungsi Granger->exog_map dan kontrak pemanggilan forecasting.
