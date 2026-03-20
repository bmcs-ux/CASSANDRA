# Plan Implementasi Backtest dan Fase Sprint CASSANDRA

Dokumen ini mencatat rencana implementasi praktis untuk mengembangkan baseline replay backtest yang sudah ada di repo menjadi fondasi evaluasi strategi yang sesuai dengan arah di `docs/backtest_strategy.md`.

## 1. Ringkasan kondisi saat ini

- Baseline yang tersedia ada di `backtest/replay.py` melalui `run_one_bar_replay_backtest`.
- Implementasi saat ini masih minimal: membaca `trade_signals`, `latest_actual_prices`, lalu menghasilkan `total_trades`, `win_rate`, `gross_pnl`, dan `avg_pnl_per_trade`.
- Belum ada decision ledger, trade ledger, export parquet, gate attribution, grid eksperimen, atau walk-forward validation.
- Test replay saat ini baru mencakup satu skenario BUY one-bar yang profit.

## 2. Tujuan Sprint

### Sprint 1 — Fondasi replay yang siap dipakai
Fokus sprint pertama adalah membangun fondasi yang stabil dan mudah diuji.

Deliverable utama:
- replay engine berbasis ledger
- KPI trading/risk dasar
- export parquet minimal
- test replay yang lebih lengkap

### Sprint 2 — Dataset evaluasi dan attribution
Fokus sprint kedua adalah memperkaya output replay agar bisa dipakai untuk analisis lanjutan.

Deliverable utama:
- decision ledger lengkap dengan field action/gating
- label multi-horizon (`pnl_1`, `pnl_3`, `pnl_5`, `max_adverse`, `max_favorable`)
- gate attribution analysis
- ringkasan alasan HOLD dan blocked-by gate

### Sprint 3 — Eksperimen dan validasi anti-bias
Fokus sprint ketiga adalah evaluasi skala besar dan ranking konfigurasi.

Deliverable utama:
- grid eksperimen threshold/gate
- walk-forward split
- evaluasi per regime
- ranking berbasis return yang disesuaikan risiko

## 3. Fase implementasi rinci

### Fase A — Bekukan kontrak data replay
Tujuan:
- Menetapkan schema input dan output agar refactor replay tidak ambigu.

Checklist:
- Tetapkan struktur `cycle_results` minimum: `timestamp`, `trade_signals`, `latest_actual_prices`.
- Definisikan schema `decision ledger`.
- Definisikan schema `trade ledger`.
- Dokumentasikan field yang belum wajib di Sprint 1 tetapi akan ditambahkan di Sprint 2.

Output:
- docstring dan komentar modul yang jelas
- fixture data replay contoh untuk test

### Fase B — Refactor replay menjadi ledger-first
Tujuan:
- Mengubah perhitungan langsung di loop menjadi pipeline kecil yang lebih mudah diuji.

Checklist:
- Pisahkan helper untuk direction, entry price, exit price, dan perhitungan return.
- Tambahkan builder untuk trade rows.
- Tambahkan builder untuk decision rows.
- Pertahankan wrapper `run_one_bar_replay_backtest` untuk kompatibilitas.

Output:
- API replay yang menghasilkan row-level data
- summary tetap tersedia sebagai wrapper kompatibel

### Fase C — KPI trading dan risk dasar
Tujuan:
- Menyediakan metrik minimum yang benar-benar berguna untuk evaluasi awal.

Checklist:
- Hitung total trades, win rate, gross return, net return, average return per trade.
- Tambahkan equity curve sederhana.
- Hitung max drawdown sederhana.
- Catat skipped trade karena data harga tidak lengkap.

Output:
- summary KPI minimum Sprint 1

### Fase D — Export parquet minimal
Tujuan:
- Menyimpan hasil replay secara row-level untuk analisis lanjutan.

Checklist:
- Simpan decision/trade ledger ke parquet.
- Gunakan field minimal yang stabil untuk Sprint 1.
- Tambahkan `schema_version`, `generated_at`, dan metadata run.

Output:
- file parquet hasil replay yang siap dipakai ulang

### Fase E — Pengujian dan validasi dasar
Tujuan:
- Menurunkan risiko regresi saat replay berkembang.

Checklist:
- Tambahkan test BUY profit.
- Tambahkan test SELL profit.
- Tambahkan test fee/slippage.
- Tambahkan test fallback entry price.
- Tambahkan test skip ketika harga exit tidak tersedia.
- Tambahkan test multi-cycle dan multi-symbol.

Output:
- suite test replay yang lebih representatif

## 4. Checklist implementasi Sprint 1

### Backlog utama
- [ ] Definisikan schema `decision ledger` dan `trade ledger`.
- [ ] Refactor `backtest/replay.py` menjadi ledger-first.
- [ ] Tambahkan KPI: gross return, net return, avg return, win rate, max drawdown.
- [ ] Tambahkan export parquet minimal.
- [ ] Perluas `tests/test_backtest_replay.py` untuk skenario BUY/SELL/cost/missing price.
- [ ] Perbarui dokumentasi baseline replay agar scope Sprint 1 jelas.

### Acceptance criteria
- [ ] API lama `run_one_bar_replay_backtest` masih bisa dipakai.
- [ ] Replay mendukung BUY dan SELL.
- [ ] Biaya transaksi dihitung dua sisi.
- [ ] Ada output row-level yang bisa diexport.
- [ ] Ada test untuk skenario utama dan edge case dasar.

## 5. Pembagian fase sprint

### Sprint 1
Scope:
- Fase A, B, C, D, dan E dalam versi minimum

Hasil akhir:
- replay engine yang sudah cukup rapi untuk dipakai iterasi berikutnya

### Sprint 2
Scope:
- tambahkan field gating dan feature model
- multi-horizon labels
- gate attribution

Hasil akhir:
- dataset perilaku sistem yang cocok untuk analisis filter dan training lanjutan

### Sprint 3
Scope:
- grid eksperimen
- walk-forward validation
- segmentasi regime
- ranking kombinasi parameter

Hasil akhir:
- framework evaluasi strategi yang bisa dipakai membandingkan banyak konfigurasi secara sistematis

## 6. Urutan kerja yang direkomendasikan

1. Rapikan kontrak data replay terlebih dahulu.
2. Refactor `backtest/replay.py` menjadi ledger-first.
3. Tambahkan KPI minimum dan export parquet.
4. Perluas test replay.
5. Baru lanjut ke gate attribution dan eksperimen.

## 7. Backlog temuan code review

Berikut empat tugas yang sudah teridentifikasi dari review codebase:

### 7.1 Perbaikan salah ketik
- Rapikan typo di `docs/backtest_strategy.md`, termasuk kata seperti `skema`, `operasi`, dan `constraints`.

### 7.2 Perbaikan bug
- Cegah `pytest` mengoleksi helper `test_and_stationarize_data` di `preprocessing/stationarity_test.py` sebagai test case.

### 7.3 Perbaikan komentar kode / ketidaksesuaian dokumentasi
- Selaraskan ekspektasi `BASE_DATA_DIR` antara `parameter.py`, fallback di `main.py`, dan test kompatibilitas.

### 7.4 Peningkatan pengujian
- Tambahkan coverage replay untuk SELL, biaya transaksi, fallback harga, dan missing exit price.
