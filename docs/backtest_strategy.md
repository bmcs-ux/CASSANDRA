# Usulan Metode Backtest untuk Loop Monitoring CASSANDRA

Dokumen ini mendeskripsikan metode backtest yang konsisten dengan alur monitoring cycle (fetch data -> update RLS -> generate signal -> kirim sinyal).

## 1) Tujuan
- Mengukur kualitas sinyal (`BUY/SELL/HOLD`) yang keluar dari `trade_signals`.
- Mengukur stabilitas model (`rls_health`, `parameter_deviations`, `dcc_metrics`, `kalman_metrics`) terhadap performa trading.
- Menilai dampak safety filter (`RLS confidence`, `deviation gate`, `news restriction`) terhadap drawdown dan missed opportunity.
- Menghasilkan kumpulan data perilaku sistem: nilai metrics (`rls_health`, `parameter_deviations`, `dcc_metrics`, `kalman_metrics`) saat setiap entry dibuat, filter mana yang paling merugikan / menguntungkan, lalu menyimpan metrics yang menguntungkan dan merugikan ke parquet agar bisa digunakan untuk melatih model transformer.

## 2) Metode Utama (Replay Backtest)
Pendekatan paling tepat untuk loop saat ini adalah **event replay backtest** berbasis artefak `current_cycle_results`.

### Kenapa replay?
- Struktur data keputusan sudah tersedia per cycle (`trade_signals`, `latest_actual_prices`, alasan skip, dan sebagainya).
- Lebih cepat dibanding simulasi penuh MT5.
- Mudah diuji ulang untuk banyak konfigurasi threshold/gate.

### Aturan baseline
- Eksekusi sinyal pada akhir cycle `t`.
- Tutup posisi pada cycle `t+1`.
- Tambahkan layer evaluasi kedua (bukan mengganti baseline) agar baseline tetap deterministik.
- Tambahkan biaya transaksi (fee + slippage) dua sisi.
- Buat "Gate Attribution Analysis".

Counterfactual Replay: original decision (dengan semua gate), lalu replay ulang dengan menghapus satu gate sambil mempertahankan gate lain.

```text
baseline: HOLD
without_confidence_gate: BUY → +0.5%
without_deviation_gate: HOLD
```

Untuk setiap gate:

```text
impact = pnl_without_gate - pnl_with_all_gates
lalu agregasi:
mean_impact
median_impact
tail_impact (p95 loss)
```

## 3) Metrik yang wajib dilaporkan
- Trading KPI: total trades, win rate, gross/net return, average return per trade.
- Risk KPI: max drawdown, volatility return, Sharpe/Sortino sederhana.
- Decision KPI: hit-rate per pasangan, distribusi alasan HOLD.
- Model KPI: korelasi confidence vs PnL, pred_var vs error, deviasi parameter vs loss tail.
- Setiap trade / HOLD simpan:

```text
X = {
  rls_confidence,
  deviation_score,
  kalman_zscore,
  dcc_correlation,
  predicted_return,
  pred_var,
  spread,
  regime_label
}
y = {
  pnl_1bar,
  pnl_3bar,
  pnl_5bar,
  max_adverse,
  max_favorable
}
action = {+1, 0, -1}
action_mask = {
  can_buy: bool,
  can_sell: bool,
  blocked_by: [list of gates]
}
```

## 4) Matriks Eksperimen
Lakukan grid eksperimen pada parameter berikut:
- `RLS_CONFIDENCE_ENTRY_THRESHOLD`
- `RLS_DEVIATION_THRESHOLD`
- `RLS_DEVIATION_CLOSE_ALL_THRESHOLD`
- `CONSENSUS_THRESHOLD`
- `KALMAN_FLIP_ZSCORE`

Output: tabel per kombinasi + ranking by risk-adjusted return, max DD, tail loss (CVaR), stability (variance antar window).

## 5) Validasi Anti-Bias
- Walk-forward split (train window -> test window bergulir).
- Pisahkan periode volatilitas rendah/tinggi, trending vs ranging, high correlation vs low correlation. Contoh:

```text
regime = kmeans(features=[
  volatility,
  trend_strength,
  correlation,
  spread
])
```

- Jangan gunakan data masa depan saat membangun fitur cycle `t`.

## 6) Implementasi awal di repo
Implementasi baseline tersedia di `backtest/replay.py` melalui fungsi `run_one_bar_replay_backtest`.

## 7) Struktur Parquet
Skema yang disarankan:

```text
{
  timestamp,
  symbol,
  # raw decision
  action,              # final
  preferred_action,    # pre-gate
  # execution states
  signal_generated,
  passed_all_gates,
  actually_executed,
  # gating
  blocked_by,          # bitmask
  gate_pass_mask,      # multi-hot
  # features
  rls_confidence,
  deviation_score,
  kalman_zscore,
  dcc_correlation,
  predicted_return,
  pred_var,
  spread,
  regime_label,
  # outcomes
  pnl_1,
  pnl_3,
  pnl_5,
  max_adverse,
  max_favorable,
  # derived labels
  hit_1,
  hit_3,
  hit_5,
  t_profit,
}
```

## 8) Optimasi proses
- Gunakan polars untuk operasi yang kompleks.
- Gunakan `ProcessPoolExecutor` untuk membagi beban ke seluruh core CPU.
- Runtime berbasis Google Colab (tanpa koneksi broker maupun MT5), dengan perhitungan komisi dan spread berbasis simulasi.
### Rekomendasi struktur modul
Agar implementasinya bersih, saya sarankan pecah backtest menjadi beberapa modul:
- backtest/schema.py
`dataclass / typed dict untuk cycle row, decision row, trade row`
- backtest/replay.py
`normalisasi cycle + build ledger`
- backtest/metrics.py
`KPI trading/risk/decision`
- backtest/attribution.py
`counterfactual gate replay`
- backtest/export.py
`parquet writer`
- backtest/experiments.py
`grid runner dan walk-forward`
Dengan begitu run_one_bar_replay_backtest bisa tetap dipakai sebagai compatibility wrapper sambil fondasinya diperluas.

Fungsi ini sengaja minimal sebagai fondasi diskusi sebelum menambah:
- simulasi multi-bar holding,
- sizing dari `position_units`,
- stop-loss/take-profit intrabar,
- portfolio-level constraints.
