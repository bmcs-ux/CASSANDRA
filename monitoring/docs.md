# Monitoring & Backtest Update Notes

## Ringkasan perubahan

Perubahan ini menambahkan jalur **single-cycle monitoring** untuk kebutuhan replay/backtest, memperbaiki fallback MT5 adapter saat backend tidak mendukung `eval()`, dan memperkaya mock MT5 agar bisa menyimpan state simulasi historis.

## Detail perubahan

### 1) `adapters/mt5_adapter.py`
- `MT5Adapter.eval()` sekarang mencoba beberapa fallback:
  1. `backend.eval(...)` jika ada.
  2. koneksi internal `backend._MetaTrader5__conn.eval(...)` jika ada.
  3. raise `AttributeError` jika dua-duanya tidak tersedia.
- Tujuan: mencegah crash langsung pada backend yang tidak expose `eval`.

### 2) `monitoring/monitor_for_vps.py`
- `fetch_high_frequency_data(...)` diubah agar langsung memakai `mt5_adapter_instance.copy_rates_range(...)`.
- Tidak lagi bergantung pada string-eval RPC (`mt5_adaptor.eval(code)`), sehingga kompatibel dengan backend native/dummy.
- Ditambahkan fungsi baru:
  - `run_single_monitoring_cycle(...)`
    - Menjalankan 1 siklus monitoring (fetch + preprocess + sinyal sederhana untuk replay).
    - Mendukung flag `is_backtest=True` supaya:
      - tidak mengirim signal live ke trade-engine,
      - tidak menjalankan sleep antar siklus.

### 3) `adapters/dummy_MetaTrader5.py`
- Menambahkan dukungan state simulasi historis:
  - `inject_historical_data(symbol, df)` (normalisasi kolom waktu ke datetime)
  - `set_simulation_step(index)` (alias kompatibel untuk replay runner)
  - `get_current_sim_time(...)`
- Menambahkan `eval(code)` shim agar kompatibel jika adapter masih memanggil eval pada backend dummy.

### 4) `backtest/engine_runner.py`
- Skrip runner baru untuk replay seluruh folder `data_base`.
- Fitur utama:
  1. Discover semua parquet per simbol.
  2. Inject data historis ke dummy MT5.
  3. Loop tiap step waktu via `set_simulation_step`.
  4. Jalankan `monitor.run_single_monitoring_cycle(..., is_backtest=True)`.
  5. Kumpulkan hasil per siklus.

## Cara menjalankan

### A. Jalankan replay backtest
```bash
python -m backtest.engine_runner
```

### B. Pemakaian programatik
```python
from backtest.engine_runner import run_backtest_simulation

results = run_backtest_simulation(data_base_dir="data_base")
print(len(results))
```

## Catatan penting
- Pada mode backtest, sinyal tidak dikirim ke trade-engine live.
- `run_single_monitoring_cycle` saat ini fokus untuk kebutuhan replay/simulasi dan aman dipanggil berulang dalam loop eksternal.
