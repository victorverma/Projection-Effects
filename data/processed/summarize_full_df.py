import argparse
import pandas as pd
from pathlib import Path

PARAMS = [
    "ABSNJZH", "EPSX", "EPSY", "EPSZ", "MEANALP", "MEANGAM", "MEANGBH",
    "MEANGBT", "MEANGBZ", "MEANJZD", "MEANJZH", "MEANPOT", "MEANSHR", "R_VALUE",
    "SAVNCPP", "SHRGT45", "TOTBSQ", "TOTFX", "TOTFY", "TOTFZ", "TOTPOT",
    "TOTUSJH", "TOTUSJZ", "USFLUX"
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a full data frame by computing summary statistics for each CSV row group."
    )
    parser.add_argument(
        "--partition",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Number of the partition the full data frame belongs to (1-5)."
    )
    parser.add_argument(
        "--use_corrected_data",
        action="store_true",
        help="Whether to summarize a full data frame with corrected data (default: False)."
    )
    return parser.parse_args()

def summarize_df(partition: int, use_corrected_data: bool) -> None:
    """
    Summarize a full data frame by computing summary statistics for each CSV row
    group.

    See summary_stat_suffixes.ipynb for more information on the names of the
    columns of the output data frame
    """
    partition_dir = Path(f"partition{partition}")
    file_name_prefix = "corrected_" if use_corrected_data else ""
    id_vars = ["partition", "type", "flare_class", "ar_num", "file"]
    full_df = (
        pd.read_parquet(partition_dir / f"{file_name_prefix}full_df.parquet")
        [id_vars + PARAMS]
    )

    grouped_full_df = full_df.groupby("file", sort=False)

    id_cols = grouped_full_df[id_vars].first()

    basic_cols = grouped_full_df[PARAMS].agg(
        ["mean", "median", "std", "var", "max", "min", "skew"]
    )
    basic_crosswalk = {
        "mean": "mean",
        "median": "median",
        "std": "stddev",
        "var": "var",
        "max": "max",
        "min": "min",
        "skew": "skewness"
    }
    basic_cols.columns = [
        f"{col[0]}_{basic_crosswalk[col[1]]}" for col in basic_cols.columns
    ]

    kurtosis_cols = grouped_full_df[PARAMS].apply(lambda x: x.kurtosis())
    kurtosis_cols.columns = [f"{col}_kurtosis" for col in kurtosis_cols.columns]

    last_value_cols = grouped_full_df[PARAMS].last(skipna=False)
    last_value_cols.columns = [
        f"{col}_last_value" for col in last_value_cols.columns
    ]

    diffs = grouped_full_df[PARAMS].diff()
    gderivative_cols = diffs.groupby(full_df["file"]).agg(["mean", "std"])
    gderivative_crosswalk = {
        "mean": "gderivative_mean", "std": "gderivative_stddev"
    }
    gderivative_cols.columns = [
        f"{col[0]}_{gderivative_crosswalk[col[1]]}"
        for col in gderivative_cols.columns
    ]

    summary_df = pd.concat(
        [basic_cols, kurtosis_cols, last_value_cols, gderivative_cols], axis=1
    )
    summary_df = summary_df[sorted(summary_df.columns)]
    summary_df = pd.concat([id_cols, summary_df], axis=1).reset_index(drop=True)
    summary_df.to_parquet(
        partition_dir / f"{file_name_prefix}summary_df.parquet"
    )

if __name__ == "__main__":
    args = parse_args()
    summarize_df(args.partition, args.use_corrected_data)
