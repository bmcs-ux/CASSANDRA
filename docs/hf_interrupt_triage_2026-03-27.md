# Analisis Interupsi Paksa (`^C`) pada Preprocessing HF

Tanggal analisis: 2026-03-27.

## Ringkasan akar masalah

`^C` paling mungkin **bukan** berasal dari Anda menekan CTRL+C langsung, tetapi dari proses yang menerima sinyal interrupt saat pekerjaan CPU/memory-heavy sedang berjalan.

Ada dua indikator utama pada pipeline:

1. **Anomali timestamp menjadi epoch 1970** pada hasil unduhan HF.
   - Di log Anda, `Latest Close ... (at 1970-01-01 00:00:00+00:00)` untuk banyak pair.
   - Ini konsisten dengan masalah normalisasi kolom waktu di dummy backend (`copy_rates_range`) saat konversi `time` memakai `pd.to_datetime(...).astype('int64') // 10**9`, yang rentan salah interpretasi unit bila input sudah numerik non-datetime bersih.
2. **Tahap gabung DataFrame (`return_type='df'`) mahal secara komputasi**.
   - Fungsi `combine_log_returns(..., return_type='df')` melakukan merge berulang (`outer`) antarpair, yang dapat sangat lambat untuk ratusan ribu bar per pair.
   - Pada kondisi data besar/indeks bermasalah, tahap ini mudah membuat proses tampak hang lalu di-interrupt oleh lingkungan runner/terminal (yang tercetak sebagai `^C`).

## Bukti teknis dari kode

- Proses preprocessing memang memanggil mode `dict` lebih dulu lalu mode `df` (sesuai urutan log Anda):
  - `preprocess_high_frequency_data(...)` memanggil `combine_log_returns(..., return_type='dict')` lalu `combine_log_returns(..., return_type='df')`.
- `combine_log_returns` mode `df` melakukan merge berulang antarpair dengan `how='outer'`.
- Dummy MT5 melakukan konversi waktu dan memunculkan warning cache miss untuk simbol yang tidak ditemukan (`DXYm`), sesuai log Anda.

## Kesimpulan operasional

Kemunculan `^C` sangat mungkin berasal dari **interrupt eksternal** saat proses `combine_log_returns(..., 'df')` sedang berat, yang dipicu/ diperparah oleh **kualitas indeks waktu yang anomali (epoch 1970)** dan ukuran data yang terlalu besar.

---

## Usulan backlog tugas (sesuai permintaan)

1. **Perbaikan salah ketik (typo)**
   - Tugas: rename modul `models_tests/acuration_metrick.py` ke nama yang konsisten, mis. `accuracy_metric.py`, lalu rapikan seluruh import yang terdampak.

2. **Perbaikan bug**
   - Tugas: perbaiki normalisasi kolom waktu di `adapters/dummy_MetaTrader5.py::copy_rates_range` agar aman untuk input datetime, epoch seconds, dan epoch milliseconds tanpa jatuh ke 1970.

3. **Perbaikan komentar/dokumentasi tidak sinkron**
   - Tugas: selaraskan docstring `combine_log_returns` agar menjelaskan dengan eksplisit biaya komputasi mode `df` (merge berulang), serta bedanya output `dict` vs `df` untuk skenario HF.

4. **Peningkatan pengujian**
   - Tugas: tambah test regresi untuk pipeline HF yang memverifikasi:
     - timestamp terakhir berada pada rentang tanggal request (bukan 1970),
     - mode `df` tidak hang untuk dataset sintetis besar,
     - perilaku saat satu simbol missing (contoh `DXYm`) tidak memutus keseluruhan preprocessing.
