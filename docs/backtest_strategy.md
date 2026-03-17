# Usulan Metode Backtest untuk Loop Monitoring CASSANDRA

Dokumen ini mendeskripsikan metode backtest yang konsisten dengan alur monitoring cycle (fetch data -> update RLS -> generate signal -> kirim sinyal).

## 1) Tujuan
- Mengukur kualitas sinyal (`BUY/SELL/HOLD`) yang keluar dari `trade_signals`.
- Mengukur stabilitas model (`rls_health`, `parameter_deviations`, `dcc_metrics`, `kalman_metrics`) terhadap performa trading.
- Menilai dampak safety filter (`RLS confidence`, `deviation gate`, `news restriction`) terhadap drawdown dan missed opportunity.

## 2) Metode Utama (Replay Backtest)
Pendekatan paling tepat untuk loop saat ini adalah **event replay backtest** berbasis artefak `current_cycle_results`.

### Kenapa replay?
- Struktur data keputusan sudah tersedia per cycle (`trade_signals`, `latest_actual_prices`, alasan skip, dsb).
- Lebih cepat dibanding simulasi penuh MT5.
- Mudah diuji ulang untuk banyak konfigurasi threshold/gate.

### Aturan baseline
- Eksekusi sinyal pada akhir cycle `t`.
- Tutup posisi pada cycle `t+1` (one-bar holding) untuk baseline yang deterministik.
- Tambahkan biaya transaksi (fee + slippage) dua sisi.

## 3) Metrik yang wajib dilaporkan
- Trading KPI: total trades, win rate, gross/net return, average return per trade.
- Risk KPI: max drawdown, volatility return, Sharpe/Sortino sederhana.
- Decision KPI: hit-rate per pasangan, distribusi alasan HOLD.
- Model KPI: korelasi confidence vs PnL, pred_var vs error, deviasi parameter vs loss tail.

## 4) Matriks Eksperimen
Lakukan grid eksperimen pada parameter berikut:
- `RLS_CONFIDENCE_ENTRY_THRESHOLD`
- `RLS_DEVIATION_THRESHOLD`
- `RLS_DEVIATION_CLOSE_ALL_THRESHOLD`
- `CONSENSUS_THRESHOLD`
- `KALMAN_FLIP_ZSCORE`

Output: tabel per kombinasi + ranking by risk-adjusted return.

## 5) Validasi Anti-Bias
- Walk-forward split (train window -> test window bergulir).
- Pisahkan periode volatilitas rendah/tinggi.
- Jangan gunakan data masa depan saat membangun fitur cycle `t`.

## 6) Implementasi awal di repo
Implementasi baseline tersedia di `backtest/replay.py` melalui fungsi `run_one_bar_replay_backtest`.

Fungsi ini sengaja minimal sebagai fondasi diskusi sebelum menambah:
- simulasi multi-bar holding,
- sizing dari `position_units`,
- stop-loss/take-profit intrabar,
- portfolio-level constraints.
