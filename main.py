import argparse

from config import N_TRIALS, RANDOM_SEED
from experiment_runner import run_experiment, run_final_test_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 HPO pilot")
    parser.add_argument(
        "--optimizer",
        choices=[
            "random_search", "optuna_tpe", "grid_search", "hyperopt_tpe",
            "gaussian_process", "smac3", "cma_es", "genetic_algorithm",
            "successive_halving", "hyperband", "pairwise",
        ],
        required=False,
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost", "lightgbm"],
        default="random_forest",
    )
    parser.add_argument(
        "--dataset",
        choices=["adult", "bank_marketing", "dry_bean"],
        default="adult",
    )
    parser.add_argument(
        "--final-evaluate",
        action="store_true",
        help="Evaluate each optimizer's best validation configuration on the test set.",
    )
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional pilot cap on Adult rows; omit for the full dataset.",
    )
    args = parser.parse_args()
    if args.final_evaluate:
        if args.optimizer is not None:
            parser.error("--final-evaluate cannot be combined with --optimizer")
        run_final_test_evaluation(args.seed, args.model, args.dataset)
    elif args.optimizer is None:
        parser.error("either --optimizer or --final-evaluate is required")
    else:
        run_experiment(
            args.optimizer,
            args.trials,
            args.seed,
            args.max_rows,
            args.model,
            args.dataset,
        )


if __name__ == "__main__":
    main()
