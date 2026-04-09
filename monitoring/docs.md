# Monitoring & Backtest Update Notes

## Ringkasan perubahan

Perubahan ini menyatukan eksekusi cycle monitoring agar **live loop** (`start_realtime_monitoring`) dan **backtest replay** sama-sama memakai `run_single_monitoring_cycle(...)` sebagai entrypoint tunggal.

## Detail perubahan kode

### 1) `adapters/mt5_adapter.py`
- `MT5Adapter.eval()` kini fallback berurutan:
  1. `backend.eval(...)`
  2. `backend._MetaTrader5__conn.eval(...)`
  3. `AttributeError` bila keduanya tidak tersedia.
- Tujuan: menghindari crash backend yang tidak expose RPC `eval`.

### 2) `adapters/dummy_MetaTrader5.py`
- Menambahkan state simulasi historis agar adapter membaca data seolah-olah live:
  - `inject_historical_data(symbol, df)`
  - `set_simulation_step(index)`
  - `get_current_sim_time(...)`
  - shim kompatibilitas `eval(code)`
- Normalisasi waktu saat injeksi data dibuat eksplisit (`time`/`timestamp` -> datetime UTC).

### 3) `monitoring/monitor_for_vps.py`
- `fetch_high_frequency_data(...)` sekarang langsung memanggil `mt5_adapter_instance.copy_rates_range(...)`.
- Nama variabel diseragamkan ke `mt5_adapter`/`mt5_adapter_instance` (menghilangkan drift `mt5_adaptor`).
- Ditambahkan/dioptimalkan `run_single_monitoring_cycle(...)`:
  - fetch + preprocess + keputusan sinyal per cycle,
  - saat `is_backtest=True`, tidak mengirim order ke trade-engine live,
  - saat `is_backtest=True`, tidak sleep.
- Loop utama `start_realtime_monitoring(...)` sekarang memanggil `run_single_monitoring_cycle(...)` setiap iterasi (single source of truth untuk jalur cycle).

### 4) `backtest/engine_runner.py`
- Runner replay otomatis folder `data_base`:
  1. discover parquet per simbol,
  2. inject ke dummy MT5,
  3. step waktu simulasi,
  4. panggil `run_single_monitoring_cycle(..., is_backtest=True)`,
  5. kumpulkan hasil per cycle.

## Catatan penting tentang alur keputusan

- Saat ini, jalur `run_single_monitoring_cycle(...)` dipakai oleh live loop **dan** replay loop agar perilaku konsisten.
- Untuk replay/backtest, behavior sengaja dibatasi: tidak ada pengiriman signal live ke trade-engine dan tidak ada sleep antar cycle.
- Jika Anda ingin mengembalikan seluruh logika RLS/Kalman kompleks lama, implementasikan di dalam `run_single_monitoring_cycle(...)` agar live/backtest tetap satu jalur.

## Cara menjalankan

### A. Replay backtest
```bash
python -m backtest.engine_runner
```

### B. Programatik
```python
from backtest.engine_runner import run_backtest_simulation

results = run_backtest_simulation(data_base_dir="data_base")
print(len(results))
```
