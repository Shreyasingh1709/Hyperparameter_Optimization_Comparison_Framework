from pathlib import Path

RANDOM_SEED = 42
N_TRIALS = 10
N_JOBS = 1
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
TRIAL_RESULTS_PATH = RESULTS_DIR / "trial_results.csv"
FINAL_RESULTS_PATH = RESULTS_DIR / "final_test_results.csv"
SUMMARY_PATH = RESULTS_DIR / "experiment_summary.csv"
BEST_CONFIGS_PATH = RESULTS_DIR / "best_configurations.json"
METADATA_PATH = RESULTS_DIR / "run_metadata.json"
