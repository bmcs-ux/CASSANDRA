# Audit Alur Logika & Data Pipeline CASSANDRA

Dokumen ini merangkum audit dari ujung ke ujung: parameter, jalur data MTF, pemuatan data, preprocessing, stasioneritas, missing data, kecukupan observasi, uji kausalitas, pelatihan model, hingga hasil model.

## 1) Ringkasan temuan utama

1. **Konfigurasi parameter belum sepenuhnya konsisten antar modul**
   - Sebagian modul memanggil nama legacy (`maxlag_test`, `alpha`, `lookback_days`) sementara `parameter.py` memakai nama baru (`maxlag_granger`, `alpha_granger`, `fred_lookback_days`).
   - Dampak: risiko runtime error / parameter mismatch.

2. **Pemuatan sumber data sudah lebih fleksibel**
   - Local CSV suffix timeframe (`_m1`, `_h1`, `_d1`) didukung.
   - Fallback ke tick Exness sudah tersedia dengan aturan URL: bulanan untuk bulan lampau, harian untuk bulan berjalan.

3. **Preprocessing cukup lengkap, namun validasi data input model perlu dipertegas**
   - Log-return, transform FRED, handling missing, stationarity test, dan Granger sudah ada.
   - Namun guardrail observasi minimum ideal per model/TF belum dinormalisasi di satu tempat konfigurasi.

4. **Alur model sudah sesuai arah arsitektur proyek**
   - D1/H1: VARX/ARX berbasis exog dari Granger.
   - H1 residual -> DCC-GARCH.
   - M1 -> Kalman state-space.
   - Packaging output `.pkl` sudah ada.

## 2) Status per tahap audit

### A. Parameter pipeline & fleksibilitas konfigurasi
- Positif:
  - `MTF_INTERVALS`, `LOOKBACK_DAYS`, `VARX_ENDOG_GROUPS`, `KALMAN_CONFIG`, threshold risiko sudah tersedia.
- Risiko:
  - Ada ketidaksinkronan nama parameter antar modul.
- Rekomendasi:
  - Satukan naming final + sediakan compatibility alias sementara.

### B. Jalur data multi-timeframe & pemuatan sumber data
- Positif:
  - `load_base_data_mtf` sudah: local CSV -> Exness fallback.
  - Exness URL logic mengikuti catatan operasional (monthly vs daily).
- Risiko:
  - Kualitas data tick/csv tidak selalu seragam (kolom, timezone, duplicate timestamp).
- Rekomendasi:
  - Tambah metrik kualitas data per pair (coverage, duplicate ratio, missing ratio) sebelum masuk preprocessing.

### C. Preprocessing, stasioneritas, missing data
- Positif:
  - Rangkaian transformasi sudah modular.
- Risiko:
  - Pemantauan dampak `dropna/ffill` terhadap sample size belum diringkas per tahap.
- Rekomendasi:
  - Tambahkan laporan “rows before/after” per langkah preprocessing untuk audit statistik.

### D. Kecukupan observasi
- Positif:
  - Ada minimum checks (contoh Granger dan VARX/ARX fit path).
- Risiko:
  - Batas minimum belum dibedakan eksplisit per model/timeframe dalam parameter tunggal.
- Rekomendasi:
  - Definisikan `MIN_OBS_GRANGER`, `MIN_OBS_VARX_D1`, `MIN_OBS_VARX_H1`, `MIN_OBS_DCC` secara eksplisit di parameter.

### E. Uji kausalitas, pelatihan model, dan hasil model
- Positif:
  - Granger -> peta exog -> fit model -> residual volatility -> packaging model result.
- Risiko:
  - Jika exog map kosong, fallback bisa menurunkan kualitas model tanpa alarm kualitas yang jelas.
- Rekomendasi:
  - Tambahkan status kualitas result per grup model (mis. `quality_flag`, `reason`).

## 3) Tugas yang diajukan (sesuai 4 kategori)

1. **Perbaiki salah ketik**
   - Tugas: standarkan istilah dokumentasi internal (contoh: "preprosessing", "jumplah", "projeck", "mecari", "mengahasil") agar konsisten ke istilah baku.

2. **Perbaiki bug**
   - Tugas: selaraskan ketergantungan parameter legacy/new agar import dan pemanggilan parameter tidak memicu runtime mismatch.

3. **Perbaiki komentar/dokumentasi**
   - Tugas: perbarui dokumen alur pipeline agar urutan D1/H1/M1, role Granger/FRED, dan fallback data source (CSV->Exness) dijelaskan satu sumber kebenaran.

4. **Tingkatkan pengujian**
   - Tugas: tambahkan integration-like tests (dengan mock data) untuk memastikan alur lengkap:
     - load data MTF,
     - preprocessing + stationarity,
     - Granger exog mapping,
     - fit D1/H1,
     - residual -> DCC,
     - M1 Kalman,
     - output package `.pkl`.
