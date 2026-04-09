# Replay Backtest & Sprint 3 Evaluation Framework

Dokumen ini menjelaskan cara menggunakan replay backtest CASSANDRA dan framework evaluasi Sprint 3 untuk membandingkan banyak konfigurasi strategi secara sistematis.

## 1. Ringkasan kapabilitas

Komponen utama saat ini:

- `backtest.replay.build_replay_ledgers`
  - membangun `decision_ledger`, `trade_ledger`, KPI baseline, multi-horizon label, dan gate attribution.
- `backtest.replay.run_one_bar_replay_backtest`
  - compatibility wrapper yang hanya mengembalikan ringkasan KPI.
- `backtest.replay.export_replay_ledgers_to_parquet`
  - menyimpan ledger ke Parquet + metadata JSON.
- `backtest.experiments.run_replay_experiment_grid`
  - menjalankan grid eksperimen, walk-forward validation, segmentasi regime, dan ranking konfigurasi.

## 2. Kontrak input replay

Minimal `cycle_results` perlu memiliki struktur berikut:

```python
cycle_results = [
    {
        "timestamp": "2026-01-01T00:00:00Z",
        "latest_actual_prices": {"EURUSD": 1.1000},
        "trade_signals": {
            "EURUSD": {
                "signal": "HOLD",               # final action sesudah gate
                "preferred_action": "BUY",      # action pre-gate
                "entry_price": 1.1000,
                "feature_model": {
                    "rls_confidence": 0.85,
                    "deviation_score": 0.03,
                    "kalman_zscore": 0.7,
                    "dcc_correlation": 0.62,
                    "predicted_return": 0.014,
                    "pred_var": 0.0004,
                    "spread": 0.0002,
                    "regime_label": "trend_up",
                },
                "action_mask": {
                    "confidence_gate": True,
                    "deviation_gate": True,
                },
                "blocked_by": [],
            }
        },
    },
    {
        "timestamp": "2026-01-02T00:00:00Z",
        "latest_actual_prices": {"EURUSD": 1.1015},
        "trade_signals": {},
    },
]
```

Catatan:

- replay baseline tetap mengeksekusi `signal` pada akhir cycle `t` lalu menutup posisi di cycle `t+1`;
- evaluasi Sprint 3 memakai `preferred_action` + label horizon (`pnl_1`, `pnl_3`, `pnl_5`) agar eksperimen threshold/gate dapat diulang tanpa membangun ulang semua artefak;
- bila `entry_price` tidak ada, replay akan fallback ke `latest_actual_prices[symbol]`.

## 2.1 Menormalisasi output `monitor_for_vps.py` ke kontrak replay

Untuk workflow Colab/backtest yang memakai artefak monitoring realtime, gunakan bridge `backtest.monitor_bridge` untuk mengubah summary monitoring menjadi `cycle_results` yang kompatibel dengan replay engine.

```python
from backtest.monitor_bridge import normalize_monitor_cycles_for_replay
from backtest.replay import build_replay_ledgers

normalized_cycles = normalize_monitor_cycles_for_replay(
    monitoring_cycles,
    symbol_to_group={"EURUSD": "fx_major"},
)
result = build_replay_ledgers(normalized_cycles, fee_bps=2.0, slippage_bps=1.0)
```

Bridge ini akan menurunkan field replay seperti `preferred_action`, `feature_model`, `action_mask`, dan `blocked_by` dari payload monitoring yang sudah ada.

## 3. Cara menjalankan replay baseline

```python
from backtest.replay import build_replay_ledgers, run_one_bar_replay_backtest

summary = run_one_bar_replay_backtest(cycle_results, fee_bps=2.0, slippage_bps=1.0)
print(summary)

result = build_replay_ledgers(cycle_results, fee_bps=2.0, slippage_bps=1.0)
print(result.summary)
print(result.decision_ledger[0])
print(result.trade_ledger[0])
```

Output penting:

- `summary.total_trades`, `summary.win_rate`, `summary.net_return`, `summary.max_drawdown`;
- `decision_ledger`
  - keputusan final, action pre-gate, fitur model, gate pass mask, blocked-by, horizon labels;
- `trade_ledger`
  - trade yang benar-benar tereksekusi pada baseline.

## 4. Cara export ledger replay

```python
from backtest.replay import export_replay_ledgers_to_parquet

paths = export_replay_ledgers_to_parquet(
    result,
    output_dir="artifacts/replay_run_001",
    run_metadata={"run_id": "replay_run_001"},
)
print(paths)
```

File yang dihasilkan:

- `decision_ledger.parquet`
- `trade_ledger.parquet`
- `metadata.json`

## 5. Cara penggunaan replay backtest untuk grid eksperimen Sprint 3

### 5.1 Menjalankan grid eksperimen langsung dari `cycle_results`

```python
from backtest.experiments import run_replay_experiment_grid

param_grid = {
    "rls_confidence_entry_threshold": [0.4, 0.6, 0.8],
    "rls_deviation_threshold": [0.05, 0.10],
    "rls_deviation_close_all_threshold": [0.15, 0.20],
    "consensus_threshold": [0.5, 1.0],
    "kalman_flip_zscore": [1.0, 1.5, 2.0],
}

report = run_replay_experiment_grid(
    cycle_results,
    param_grid,
    fee_bps=2.0,
    slippage_bps=1.0,
    train_size=120,
    test_size=40,
    step_size=40,
    horizon_field="pnl_1",
)

print(report["ranking"][:5])
```

### 5.2 Menjalankan eksperimen dari ledger replay yang sudah ada

```python
from backtest.experiments import build_experiment_grid, evaluate_walk_forward_grid

configs = build_experiment_grid(param_grid)
report = evaluate_walk_forward_grid(
    result.decision_ledger,
    configs,
    train_size=120,
    test_size=40,
    step_size=40,
    horizon_field="pnl_1",
)
```

## 6. Cara kerja metodologi Sprint 3 yang diterapkan

### 6.1 Grid eksperimen

Framework akan membentuk semua kombinasi parameter berikut:

- `rls_confidence_entry_threshold`
- `rls_deviation_threshold`
- `rls_deviation_close_all_threshold`
- `consensus_threshold`
- `kalman_flip_zscore`

Setiap kombinasi dievaluasi sebagai konfigurasi `cfg_001`, `cfg_002`, dan seterusnya.

### 6.2 Walk-forward validation

- data diurutkan secara kronologis berdasarkan `timestamp`;
- setiap window dibagi menjadi `train_size` lalu `test_size`;
- cursor bergeser menurut `step_size` (default = `test_size`);
- tidak ada row masa depan yang masuk ke window sebelumnya.

### 6.3 Segmentasi regime

Prioritas segmentasi:

1. gunakan `regime_label` dari artefak monitoring jika tersedia;
2. jika tidak tersedia, pakai heuristik dari:
   - `pred_var` → `high_vol` / `low_vol`
   - `predicted_return` → `trend_up` / `trend_down`
   - `dcc_correlation` → `high_corr` / `low_corr`
   - `spread` → `wide_spread` / `tight_spread`

### 6.4 Ranking konfigurasi

Ranking memakai gabungan:

- total return test (`total_test_net_return`)
- rata-rata Sharpe test (`average_test_sharpe`)
- penalti drawdown (`max_test_drawdown`)
- tail loss (`mean_test_cvar_95`)
- stabilitas antar-window (`stability_variance`)

Interpretasi praktis:

- skor lebih tinggi berarti konfigurasi lebih menarik;
- drawdown lebih rendah dan stabilitas lebih baik akan membantu ranking;
- tetap review detail `windows`, `blocked_by_summary`, dan `regime_summary`, bukan hanya skor total.

## 7. Struktur output laporan eksperimen

Field penting pada `report`:

- `windows`
  - daftar walk-forward window dengan index dan timestamp train/test;
- `configs`
  - detail hasil per konfigurasi;
- `configs[*].windows[*].train_summary`
  - performa konfigurasi pada train split;
- `configs[*].windows[*].test_summary`
  - performa konfigurasi pada test split;
- `configs[*].aggregate`
  - ringkasan seluruh window untuk satu konfigurasi;
- `ranking`
  - tabel ranking siap dipakai untuk seleksi parameter.

## 8. Checklist penggunaan praktis

- [ ] Pastikan `cycle_results` berisi `timestamp`, `trade_signals`, dan `latest_actual_prices`.
- [ ] Jika ingin evaluasi Sprint 3 yang kaya, sertakan `preferred_action`, `feature_model`, dan `blocked_by` pada signal.
- [ ] Jalankan `build_replay_ledgers` terlebih dahulu bila ingin audit row-level.
- [ ] Gunakan `run_replay_experiment_grid` untuk eksplorasi banyak konfigurasi.
- [ ] Tinjau `ranking`, `aggregate`, dan `regime_summary` bersama-sama.
- [ ] Export ledger jika hasil akan dianalisis ulang di notebook atau pipeline lain.

## 9. Batasan implementasi saat ini

- ranking masih memakai formula heuristik sederhana, belum optimizer portofolio penuh;
- evaluasi eksperimen saat ini memakai label horizon dari replay (`pnl_1` default), bukan simulasi holding multi-bar yang dinamis;
- `rls_deviation_close_all_threshold` saat ini diperlakukan sebagai gate tambahan berbasis `deviation_score` pada level keputusan;
- segmentasi regime fallback masih rule-based, belum clustering seperti KMeans.

Batasan ini sengaja menjaga implementasi tetap deterministik dan cepat, sambil konsisten dengan metodologi di `docs/backtest_strategy.md`.
