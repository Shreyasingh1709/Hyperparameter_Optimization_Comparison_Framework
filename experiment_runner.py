from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any

from config import (
    BEST_CONFIGS_PATH,
    BANK_BEST_CONFIGS_PATH,
    BANK_FINAL_RESULTS_PATH,
    BANK_METADATA_PATH,
    BANK_RESULTS_DIR,
    BANK_SUMMARY_PATH,
    BANK_TRIAL_RESULTS_PATH,
    METADATA_PATH,
    RANDOM_SEED,
    RESULTS_DIR,
    FINAL_RESULTS_PATH,
    SUMMARY_PATH,
    TRIAL_RESULTS_PATH,
)
from data.adult import load_adult
from data.bank_marketing import load_bank_marketing
from models.lightgbm import SEARCH_SPACE as LIGHTGBM_SEARCH_SPACE
from models.random_forest import SEARCH_SPACE as RANDOM_FOREST_SEARCH_SPACE
from models.xgboost import SEARCH_SPACE as XGBOOST_SEARCH_SPACE
from objective import (
    ExperimentContext,
    evaluate_configuration,
    evaluate_test_configuration,
)
from optimizers import optuna_tpe, random_search
from optimizers import (
    cma_es,
    genetic_algorithm,
    gaussian_process,
    grid_search,
    hyperband,
    hyperopt_tpe,
    pairwise,
    smac3,
    successive_halving,
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _append_csv(path, row: dict[str, Any], results_dir=None) -> None:
    (results_dir or RESULTS_DIR).mkdir(exist_ok=True)
    encoded_row = {
        key: json.dumps(value, default=_json_default)
        if isinstance(value, dict)
        else value
        for key, value in row.items()
    }
    existing_rows: list[dict[str, Any]] = []
    fields = list(encoded_row)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            fields = list(reader.fieldnames or [])
            for key in encoded_row:
                if key not in fields:
                    fields.append(key)
    existing_rows.append(encoded_row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing_rows)


def _trial_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    parameters = row.get("parameters", {})
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    return (
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("optimizer", "")),
        str(row.get("trial_id", "")),
        str(row.get("data_rows", "")),
        json.dumps(parameters, sort_keys=True, default=_json_default),
    )


def _append_unique_trials(records: list[dict[str, Any]], trial_path, results_dir) -> None:
    existing: set[tuple[str, str, str, str, str, str]] = set()
    if trial_path.exists():
        with trial_path.open(newline="", encoding="utf-8-sig") as handle:
            existing = {_trial_key(row) for row in csv.DictReader(handle)}
    for record in records:
        if _trial_key(record) not in existing:
            _append_csv(trial_path, record, results_dir)
            existing.add(_trial_key(record))


def _dataset_paths(dataset_name: str):
    if dataset_name == "adult":
        return (
            RESULTS_DIR,
            TRIAL_RESULTS_PATH,
            SUMMARY_PATH,
            BEST_CONFIGS_PATH,
            METADATA_PATH,
            FINAL_RESULTS_PATH,
        )
    if dataset_name == "bank_marketing":
        return (
            BANK_RESULTS_DIR,
            BANK_TRIAL_RESULTS_PATH,
            BANK_SUMMARY_PATH,
            BANK_BEST_CONFIGS_PATH,
            BANK_METADATA_PATH,
            BANK_FINAL_RESULTS_PATH,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def run_experiment(
    optimizer_name: str,
    trials: int = 10,
    seed: int = 42,
    max_rows: int | None = None,
    model_name: str = "random_forest",
    dataset_name: str = "adult",
) -> list[dict[str, Any]]:
    model_spaces = {
        "random_forest": RANDOM_FOREST_SEARCH_SPACE,
        "xgboost": XGBOOST_SEARCH_SPACE,
        "lightgbm": LIGHTGBM_SEARCH_SPACE,
    }
    if model_name not in model_spaces:
        raise ValueError(f"Unsupported model: {model_name}")
    search_space = model_spaces[model_name]
    results_dir, trial_path, summary_path, best_path, metadata_path, _ = _dataset_paths(
        dataset_name
    )
    splits = (
        load_adult(seed, max_rows=max_rows)
        if dataset_name == "adult"
        else load_bank_marketing(seed, max_rows=max_rows)
    )
    context = ExperimentContext(
        dataset_name, model_name, splits.X_train, splits.X_validation,
        splits.y_train, splits.y_validation, splits.categorical_columns, seed,
        splits.X_test, splits.y_test,
    )
    evaluations = 0

    def evaluator(parameters: dict[str, Any]) -> dict[str, Any]:
        nonlocal evaluations
        evaluations += 1
        print(f"Starting trial {evaluations}/{trials}...", flush=True)
        result = evaluate_configuration(parameters, context)
        print(
            f"Finished trial {evaluations}/{trials}: "
            f"{result['status']}, macro-F1={result['validation_macro_f1']}",
            flush=True,
        )
        return result
    runners = {
        "random_search": random_search.run,
        "optuna_tpe": optuna_tpe.run,
        "grid_search": grid_search.run,
        "hyperopt_tpe": hyperopt_tpe.run,
        "gaussian_process": gaussian_process.run,
        "smac3": smac3.run,
        "cma_es": cma_es.run,
        "genetic_algorithm": genetic_algorithm.run,
        "successive_halving": successive_halving.run,
        "hyperband": hyperband.run,
        "pairwise": pairwise.run,
    }
    if optimizer_name not in runners:
        raise ValueError(f"Stage 1 supports only: {', '.join(sorted(runners))}")
    records = runners[optimizer_name](search_space, evaluator, trials, seed)
    data_rows = len(splits.X_train) + len(splits.X_validation) + len(splits.X_test)
    for record in records:
        record["data_rows"] = data_rows
    _append_unique_trials(records, trial_path, results_dir)
    completed = [record for record in records if record["status"] == "completed"]
    best = max(completed, key=lambda record: record["validation_macro_f1"]) if completed else None
    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "optimizer": optimizer_name,
        "requested_trials": trials,
        "objective_evaluations": len(records),
        "completed_trials": len(completed),
        "best_validation_macro_f1": best["validation_macro_f1"] if best else None,
        "best_parameters": best["parameters"] if best else None,
        "max_rows": max_rows,
    }
    _append_csv(summary_path, summary, results_dir)
    best_configs = {}
    if best_path.exists():
        best_configs = json.loads(best_path.read_text(encoding="utf-8"))
    best_configs[f"{dataset_name}/{model_name}/{optimizer_name}"] = summary
    best_path.parent.mkdir(exist_ok=True)
    best_path.write_text(
        json.dumps(best_configs, indent=2, default=_json_default),
        encoding="utf-8",
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
        "optimizer": optimizer_name,
        "trials": trials,
        "max_rows": max_rows,
        "search_space": search_space,
        "test_set_used_during_optimization": False,
    }
    metadata_path.parent.mkdir(exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return records


def run_final_test_evaluation(
    seed: int = RANDOM_SEED,
    model_name: str = "random_forest",
    dataset_name: str = "adult",
) -> list[dict[str, Any]]:
    _, trial_path, _, _, _, final_path = _dataset_paths(dataset_name)
    if not trial_path.exists():
        raise FileNotFoundError(f"Trial results not found: {trial_path}")

    splits = (
        load_adult(seed)
        if dataset_name == "adult"
        else load_bank_marketing(seed)
    )
    context = ExperimentContext(
        dataset_name,
        model_name,
        splits.X_train,
        splits.X_validation,
        splits.y_train,
        splits.y_validation,
        splits.categorical_columns,
        seed,
        splits.X_test,
        splits.y_test,
    )
    with trial_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = len(splits.X_train) + len(splits.X_validation) + len(splits.X_test)
    eligible = [
        row
        for row in rows
        if row.get("dataset") == dataset_name
        and row.get("model") == model_name
        and row.get("status") == "completed"
        and row.get("data_rows") == str(expected_rows)
    ]
    if not eligible:
        raise ValueError("No completed full-dataset trials are available")

    best_by_optimizer: dict[str, dict[str, str]] = {}
    for row in eligible:
        optimizer = row["optimizer"]
        current = best_by_optimizer.get(optimizer)
        if current is None or float(row["validation_macro_f1"]) > float(
            current["validation_macro_f1"]
        ):
            best_by_optimizer[optimizer] = row

    final_records: list[dict[str, Any]] = []
    for optimizer, row in sorted(best_by_optimizer.items()):
        parameters = json.loads(row["parameters"])
        print(f"Evaluating final test configuration: {optimizer}...", flush=True)
        result = evaluate_test_configuration(parameters, context)
        final_records.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "optimizer": optimizer,
                "trial_id": row["trial_id"],
                "parameters": parameters,
                "validation_macro_f1": float(row["validation_macro_f1"]),
                "validation_accuracy": float(row["validation_accuracy"]),
                "validation_balanced_accuracy": float(
                    row["validation_balanced_accuracy"]
                ),
                "status": result["status"],
                "test_macro_f1": result["test_macro_f1"],
                "test_accuracy": result["test_accuracy"],
                "test_balanced_accuracy": result["test_balanced_accuracy"],
                "error": result["error"],
                "elapsed_seconds": result["elapsed_seconds"],
                "data_rows": expected_rows,
            }
        )

    existing_final_records: list[dict[str, Any]] = []
    if final_path.exists():
        with final_path.open(newline="", encoding="utf-8-sig") as handle:
            existing_final_records = list(csv.DictReader(handle))
    existing_final_records = [
        record for record in existing_final_records if record.get("model") != model_name
    ]
    all_final_records = existing_final_records + final_records
    fields = list(all_final_records[0])
    for record in all_final_records:
        for key in record:
            if key not in fields:
                fields.append(key)
    final_path.parent.mkdir(exist_ok=True)
    with final_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in all_final_records:
            writer.writerow(
                {
                    key: json.dumps(value, default=_json_default)
                    if isinstance(value, dict)
                    else value
                    for key, value in record.items()
                }
            )
    return final_records
