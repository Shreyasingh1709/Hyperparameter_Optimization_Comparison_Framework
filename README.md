# Comparative HPO Framework

The pilot path supports Adult Census Income with Random Forest, XGBoost, or
LightGBM and eleven identifiable optimizer adapters. The objective uses validation macro-F1 and
never uses the test split during tuning.

## Python 3.10 (system-wide, no virtual environment)

In VS Code, select:
`C:\Users\shrey\AppData\Local\Programs\Python\Python310\python.exe`.

PowerShell commands:

```powershell
py -3.10 -m pytest -q
py -3.10 main.py --optimizer random_search --trials 10
py -3.10 main.py --optimizer optuna_tpe --trials 10
py -3.10 main.py --optimizer grid_search --trials 10
py -3.10 main.py --optimizer hyperopt_tpe --trials 10
py -3.10 main.py --optimizer gaussian_process --trials 10
py -3.10 main.py --optimizer smac3 --trials 10
py -3.10 main.py --optimizer cma_es --trials 10
py -3.10 main.py --optimizer genetic_algorithm --trials 10
py -3.10 main.py --optimizer successive_halving --trials 10
py -3.10 main.py --optimizer hyperband --trials 10
py -3.10 main.py --optimizer pairwise --trials 10
py -3.10 main.py --model xgboost --optimizer random_search --trials 10
# Repeat the XGBoost command with each of the other ten optimizer names above.
py -3.10 main.py --model lightgbm --optimizer random_search --trials 10
# Repeat the LightGBM command with each of the other ten optimizer names above.
py -3.10 main.py --dataset bank_marketing --model lightgbm --optimizer random_search --trials 10
# Replace lightgbm with random_forest or xgboost for the other models and repeat all 11 optimizers.
# Bank Marketing final evaluation:
py -3.10 main.py --dataset bank_marketing --model lightgbm --final-evaluate
# Evaluate the best full-dataset configuration for a model on the test split:
py -3.10 main.py --model xgboost --final-evaluate
# Faster, explicitly reduced-data smoke pilot:
py -3.10 main.py --optimizer random_search --trials 5 --max-rows 10000
```

The Adult dataset is downloaded by scikit-learn from OpenML on first use.
Its target column is verified as `class`, categorical columns are one-hot
encoded inside a training-fitted pipeline, and splits are stratified 70/15/15.

Results are appended to `results/trial_results.csv` and
`results/experiment_summary.csv`; best configurations and run metadata are
written as JSON. Duplicate trial records from rerunning the same seeded
command are skipped. The default uses all Adult rows. `--max-rows` is an
explicit reduced-data pilot option and is recorded in the summary and
metadata; reduced-data results must not be presented as full-dataset research
results. A full experiment is not launched automatically.

The resource-based methods record their resource level in trial results.
Their actual objective-call count can differ from the requested nominal
budget and must be reported as such.

The active Python 3.10 installation already contains the Stage 1 dependencies.
For a missing package, install it system-wide with
`py -3.10 -m pip install -r requirements.txt`.
