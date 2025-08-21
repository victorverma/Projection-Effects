import argparse
import pandas as pd
from joblib import dump
from numpy.typing import ArrayLike
from pathlib import Path
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a support vector classifier on a given partition."
    )
    parser.add_argument(
        "--partition",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Number of the partition to use (1-5)."
    )
    parser.add_argument(
        "--use_corrected_data",
        action="store_true",
        help="Whether to use corrected data (default: False)."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of jobs to run in parallel (default: -1, i.e., use all processors)."
    )
    return parser.parse_args()

def calc_tss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) - fp / (fp + tn)

def train_model(partition: int, use_corrected_data: bool, n_jobs: int) -> None:
    parent_parent_dir = Path("..") / ".."
    partition_str = f"partition{partition}"
    prefix = "corrected_" if use_corrected_data else ""
    train_summary_df = (
        pd.read_parquet(
            parent_parent_dir / "data" / "processed" / partition_str / f"{prefix}summary_df.parquet"
        )
        .sort_values(["type", "file"]) # Make the results fully reproducible
    )
    improvements = pd.read_csv(
        parent_parent_dir / f"Partition{partition}Improvements.csv"
    )

    pipeline_ = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", class_weight="balanced"))
        ]
    )

    powers_of_10 = [10**i for i in range(-4, 4)]
    param_grid = {
        "svc__C": powers_of_10, "svc__gamma": ["scale"] + powers_of_10
    }

    grid_search = GridSearchCV(
        pipeline_,
        param_grid,
        scoring=make_scorer(calc_tss),
        n_jobs=n_jobs,
        refit=True,
        cv=StratifiedGroupKFold(n_splits=3),
        verbose=4,
        return_train_score=True
    )

    specs_col = "CORR_Specs" if use_corrected_data else "Specs"
    X = train_summary_df.loc[:, improvements.head(25)[specs_col]]
    y = train_summary_df["type"]
    groups = train_summary_df["ar_num"]
    grid_search.fit(X, y, groups=groups)

    dump(grid_search, Path(partition_str) / "grid_search.joblib", compress=True)

if __name__ == "__main__":
    args = parse_args()
    train_model(args.partition, args.use_corrected_data, args.n_jobs)
