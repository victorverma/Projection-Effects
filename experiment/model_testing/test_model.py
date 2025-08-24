import argparse
import os
import pandas as pd
from joblib import load
from numpy.typing import ArrayLike
from pathlib import Path
from plotnine import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

GRID_SEARCH_DIR = Path("..") / "model_training"
TOP_FEATURES_DIR = Path("..") / "feature_selection" / "feature_rankings" / "reproduced"
DF_DIR = Path("..") / ".." / "data" / "processed"
Y_LABELS = ["FL", "NF"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a support vector classifier trained on a given partition on the other partitions."
    )
    parser.add_argument(
        "--train_partition",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Number of the partition the classifer was trained on (1-5)."
    )
    parser.add_argument(
        "--use_corrected_data",
        action="store_true",
        help="Whether to use corrected data (default: False)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="Maximum number of worker processes (default: CPU count)."
    )
    return parser.parse_args()

def calc_tpr(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    tp, fn, _, _ = confusion_matrix(y_true, y_pred, labels=Y_LABELS).ravel()
    return tp / (tp + fn)

def calc_fpr(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    _, _, fp, tn = confusion_matrix(y_true, y_pred, labels=Y_LABELS).ravel()
    return fp / (fp + tn)

def calc_tss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=Y_LABELS).ravel()
    return tp / (tp + fn) - fp / (fp + tn)

def calc_hss2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=Y_LABELS).ravel()
    numer = 2 * (tp * tn - fn * fp)
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return numer / denom

def calc_metrics(
        train_grid_search: GridSearchCV,
        train_top_features: pd.Series,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
    X_test = test_df[train_top_features]
    y_test = test_df["type"]
    y_test_pred = train_grid_search.best_estimator_.predict(X_test)
    metrics = pd.DataFrame(
        {
            "tpr": calc_tpr(y_test, y_test_pred),
            "fpr": calc_fpr(y_test, y_test_pred),
            "tss": calc_tss(y_test, y_test_pred),
            "hss2": calc_hss2(y_test, y_test_pred)
        },
        index=[0]
    )
    return metrics

if __name__ == "__main__":
    args = parse_args()
    train_partition = args.train_partition
    use_corrected_data = args.use_corrected_data
    max_workers = args.max_workers

    data_type = "corrected" if use_corrected_data else "uncorrected"
    prefix = "corrected_" if use_corrected_data else ""

    train_partition_str = f"partition{train_partition}"
    train_grid_search = load(
        GRID_SEARCH_DIR / train_partition_str / "grid_search.joblib"
    )
    train_top_features = (
        pd.read_parquet(TOP_FEATURES_DIR / f"{train_partition_str}.parquet")
        .query("rank <= 25")
        ["predictor"]
    )
    results_dir = Path(train_partition_str)

    test_partitions = [
        partition
        for partition in range(1, 6) if partition != train_partition
    ]
    results = []
    for test_partition in test_partitions:
        test_df = (
            pd.read_parquet(
                DF_DIR / f"partition{test_partition}" / f"{prefix}summary_df.parquet"
            )
            .dropna()
        )
        metrics = calc_metrics(
            train_grid_search, train_top_features, test_df
        )
        metrics.insert(0, "test_partition", test_partition)
        results.append(metrics)

    results = pd.concat(results, ignore_index=True)
    results.insert(0, "data_type", data_type)
    results.insert(1, "train_partition", train_partition)
    results.to_parquet(results_dir / f"{prefix}results.parquet")
