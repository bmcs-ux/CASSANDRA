import importlib
import os
import sys
from typing import Any, Dict

import pandas as pd


ROOT_DIR = '/content/drive/MyDrive/books/CASSANDRA/'
MODULES_TO_RELOAD = [
    'parameter', 'raw.pair_raw', 'raw.makro_raw', 'preprocessing.log_return',
    'preprocessing.fred_transform', 'preprocessing.handle_missing',
    'preprocessing.combine_data', 'preprocessing.stationarity_test',
    'fitted_models.granger', 'fitted_models.def_varx', 'forecast', 'restored',
    'fitted_models.dcc_garch_process', 'main'
]
GLOBAL_VARS_TO_CLEAR = [
    'run_id', 'base_dfs', 'fred_df', 'fred_meta', 'log_returns', 'cleaned_fred',
    'granger_df', 'exog_map', 'combined_fred_for_model', 'models',
    'fitted_dcc_garch_models', 'combined_forecasts_with_intervals',
    'restored_price_forecasts_with_intervals', 'execution_log', 'model_summary_df',
    'last_actual_prices_dict', 'monitoring_results', 'full_monitoring_log'
]


def _ensure_root_in_path() -> None:
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)


def _reload_modules() -> list[str]:
    modules_to_process = sorted(MODULES_TO_RELOAD, key=lambda x: x.count('.') * 100 + len(x))
    reloaded_modules: list[str] = []

    for module_name in modules_to_process:
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                print(f"Reloaded: {module_name}")
            else:
                importlib.import_module(module_name)
                print(f"Imported: {module_name}")
            reloaded_modules.append(module_name)
        except ImportError as err:
            print(f"Error importing/reloading {module_name}: {err}")
        except Exception as err:  # noqa: BLE001 - orchestration log should continue
            print(f"An unexpected error occurred with {module_name}: {err}")

    print(f"All necessary modules processed: {', '.join(reloaded_modules)}.")
    return reloaded_modules


def _clear_globals(scope: Dict[str, Any]) -> None:
    for var_name in GLOBAL_VARS_TO_CLEAR:
        if var_name in scope:
            del scope[var_name]


def _print_result_keys(name: str, value: Any) -> None:
    print(f"\nKeys in {name} after running main.py:")
    if isinstance(value, dict) and value:
        print(list(value.keys()))
    else:
        print(f"{name} is not available or empty.")


def run_pipeline() -> Dict[str, Any]:
    _ensure_root_in_path()
    _reload_modules()

    import main

    _clear_globals(globals())

    print("\nMenjalankan pipeline utama dan menyimpan hasil peramalan...")

    results: Dict[str, Any] = {}
    try:
        main_result = main.main()
        if not main_result or not isinstance(main_result, tuple) or len(main_result) != 16:
            raise RuntimeError("main.main() tidak mengembalikan tuple 16 elemen yang valid.")

        (
            results['run_id'],
            results['base_dfs'],
            results['fred_df'],
            results['fred_meta'],
            results['log_returns'],
            results['cleaned_fred'],
            results['granger_df'],
            results['exog_map'],
            results['combined_fred_for_model'],
            results['models'],
            results['fitted_dcc_garch_models'],
            results['combined_forecasts_with_intervals'],
            results['restored_price_forecasts_with_intervals'],
            results['execution_log'],
            results['model_summary_df'],
            results['last_actual_prices_dict'],
        ) = main_result

        print("\n✅ Pipeline utama selesai. Semua variabel hasil sekarang tersedia secara global.")

        output_filepath = os.path.join(ROOT_DIR, 'pipeline_output.txt')
        main.save_pipeline_outputs_to_file(
            filepath=output_filepath,
            execution_log=results['execution_log'],
            model_summary_df=results['model_summary_df'],
            combined_log_returns_forecasts=results['combined_forecasts_with_intervals'],
            restored_price_forecasts=results['restored_price_forecasts_with_intervals'],
            last_actual_prices_dict=results['last_actual_prices_dict'],
        )

        print("\n--- Konfirmasi Variabel Global Baru ---")
        print(
            "fitted_dcc_garch_models keys: "
            f"{list(results['fitted_dcc_garch_models'].keys()) if results['fitted_dcc_garch_models'] else 'N/A'}"
        )
        print(
            "combined_forecasts_with_intervals keys: "
            f"{list(results['combined_forecasts_with_intervals'].keys()) if results['combined_forecasts_with_intervals'] else 'N/A'}"
        )
        print(
            "restored_price_forecasts_with_intervals keys: "
            f"{list(results['restored_price_forecasts_with_intervals'].keys()) if results['restored_price_forecasts_with_intervals'] else 'N/A'}"
        )

    except Exception as err:  # noqa: BLE001 - orchestration fallback for notebook usage
        print(f"\n❌ Terjadi error saat menjalankan pipeline utama: {err}")
        results = {
            'run_id': None,
            'base_dfs': {},
            'fred_df': None,
            'fred_meta': None,
            'log_returns': {},
            'cleaned_fred': {},
            'granger_df': pd.DataFrame(),
            'exog_map': {},
            'combined_fred_for_model': pd.DataFrame(),
            'models': {},
            'fitted_dcc_garch_models': {},
            'combined_forecasts_with_intervals': {},
            'restored_price_forecasts_with_intervals': {},
            'execution_log': f"Error during pipeline execution: {err}",
            'model_summary_df': pd.DataFrame(),
            'last_actual_prices_dict': {},
            'monitoring_results': [],
            'full_monitoring_log': "",
        }

    _print_result_keys('base_dfs', results.get('base_dfs'))
    _print_result_keys('models', results.get('models'))
    _print_result_keys('combined_forecasts_with_intervals', results.get('combined_forecasts_with_intervals'))

    globals().update(results)
    return results


if __name__ == '__main__':
    run_pipeline()
