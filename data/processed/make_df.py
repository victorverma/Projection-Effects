import argparse
import numpy as np
import os
import pandas as pd
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from numpy.polynomial.polynomial import polyval
from pathlib import Path

ROOT_DIR = "../raw/swan_sf/"
CSV_PATH_PATTERN = "*/*.csv"
POLY_COEFS_DIR = "../../dictionary_fits/"
PARAMS = [
    "ABSNJZH", "EPSX", "EPSY", "EPSZ", "MEANALP", "MEANGAM", "MEANGBH",
    "MEANGBT", "MEANGBZ", "MEANJZD", "MEANJZH", "MEANPOT", "MEANSHR", "R_VALUE",
    "SAVNCPP", "SHRGT45", "TOTBSQ", "TOTFX", "TOTFY", "TOTFZ", "TOTPOT",
    "TOTUSJH", "TOTUSJZ", "USFLUX"
]
SUMMARY_STATS = [
    "mean", "median", "std", "var", "max", "min", "skew", "kurt", "last",
    "diff_mean", "diff_std"
]

poly_coefs = pd.DataFrame(columns=PARAMS)
for param in PARAMS:
    with open(f"{POLY_COEFS_DIR}/saved_dictionary_{param}.pkl", "rb") as f:
        # The constant term of 1 seems to be missing, so prepend 1
        poly_coefs[param] = np.insert(pickle.load(f)["Mean Fit"], 0, 1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a Pandas data frame containing a one-row summary of each SWAN-SF CSV."
    )
    parser.add_argument(
        "--partition",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Number of the partition to use (1-5)."
    )
    parser.add_argument(
        "--df_type",
        type=str,
        choices=["full", "summary"],
        default="full",
        help="Type of data frame to produce, 'full' (no summarization) or 'summary' (default: 'full')."
    )
    parser.add_argument(
        "--correct_params",
        action="store_true",
        help="Whether to correct the parameters using the correction factor polynomials (default: False)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="Maximum number of worker processes (default: CPU count)."
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Chunk size for ProcessPoolExecutor.map (default: 1)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Rows to buffer in memory before appending to the Parquet file (default: 1000)."
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
        help="Print a progress message every N files (0 to disable, default: 1000)."
    )
    return parser.parse_args()

def get_partition_and_type(csv_path: Path) -> tuple[int, str]:
    partition = csv_path.parent.parent.name
    partition = int(partition[-1])
    type = csv_path.parent.name # either FL or NF
    return partition, type

def correct_params(csv_df: pd.DataFrame) -> pd.DataFrame:
    for param in PARAMS:
        csv_df[param] /= polyval(csv_df["HC_ANGLE"], poly_coefs[param])
    return csv_df

def process_csv(csv_path: Path, *, correct_params_: bool) -> pd.DataFrame:
    """
    Put the data in the given CSV on the given parameters in a data frame.
    """
    partition, type = get_partition_and_type(csv_path)

    csv_df = pd.read_csv(
        csv_path, sep="\t", usecols=["Timestamp", "HC_ANGLE"] + PARAMS
    )
    csv_df[PARAMS] = csv_df[PARAMS].interpolate(
        method="linear", limit_direction="both"
    )
    if correct_params_:
        csv_df = correct_params(csv_df)

    csv_df.insert(0, "partition", partition)
    csv_df.insert(1, "type", type)
    csv_df.insert(2, "file", csv_path.name)

    return csv_df

def summarize_csv(csv_path: Path, *, correct_params_: bool) -> pd.DataFrame:
    """
    Construct a one-row data frame with summary statistics for the specified CSV
    and parameters.
    """
    partition, type = get_partition_and_type(csv_path)

    csv_df = pd.read_csv(csv_path, sep="\t", usecols=["HC_ANGLE"] + PARAMS)
    csv_df = csv_df.interpolate(method="linear", limit_direction="both")
    if correct_params_:
        csv_df = correct_params(csv_df)

    out = {"partition": partition, "type": type, "file": csv_path.name}

    for col in PARAMS:
        col_vals = csv_df[col]
        num_non_nas = col_vals.notna().sum()

        if num_non_nas == 0:
            for summary_stat in SUMMARY_STATS:
                out[f"{col}_{summary_stat}"] = np.nan
            continue

        mean = col_vals.mean()
        median = col_vals.median()
        std = col_vals.std() if num_non_nas > 1 else np.nan
        var = col_vals.var() if num_non_nas > 1 else np.nan
        max_ = col_vals.max()
        min_ = col_vals.min()
        skew = col_vals.skew() if num_non_nas > 2 else np.nan
        kurt = col_vals.kurtosis() if num_non_nas > 3 else np.nan

        last = col_vals.iloc[-1]
        diffs = col_vals.diff().dropna()
        diff_mean = diffs.mean() if len(diffs) else np.nan
        diff_std = diffs.std() if len(diffs) else np.nan

        stat_vals = [
            mean, median, std, var, max_, min_, skew, kurt,
            last, diff_mean, diff_std
        ]
        for summary_stat, stat_val in zip(SUMMARY_STATS, stat_vals):
            out[f"{col}_{summary_stat}"] = stat_val

    return pd.DataFrame([out])

def make_df(
        partition: int,
        df_type: str,
        correct_params_: bool,
        max_workers: int,
        chunksize: int,
        batch_size: int,
        progress_every: int
    ) -> None:
    """
    Compute summary statistics for all CSVs and save results to a Parquet file.
    """
    prefix = "corrected_" if correct_params_ else ""
    out_path = f"partition{partition}/{prefix}{df_type}_df.parquet"
    if os.path.exists(out_path):
        os.remove(out_path)

    # partial is used to create functions that are picklable and thus usable by
    # the ProcessPoolExecutor instance. Fixing the value of correct_params_ in a
    # partial call requires correct_params_ to be a keyword argument of
    # process_csv and summarize_csv
    if df_type == "full":
        fun = partial(process_csv, correct_params_=correct_params_)
        first_word = "Processed"
    else:
        fun = partial(summarize_csv, correct_params_=correct_params_)
        first_word = "Summarized"

    writer = None
    rows = []
    num_finished = 0

    def flush_batch():
        nonlocal rows, writer
        if not rows:
            return
        batch_df = pd.concat(rows, ignore_index=True)
        rows = []
        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

    partition_path = Path(ROOT_DIR) / f"partition{partition}"
    num_csvs = sum(1 for _ in partition_path.glob(CSV_PATH_PATTERN))
    files_iter = partition_path.glob(CSV_PATH_PATTERN)
    correction_phrase = " with corrections" if correct_params_ else ""
    with ProcessPoolExecutor(max_workers) as ex:
        for row in ex.map(fun, files_iter, chunksize=chunksize):
            rows.append(row)
            num_finished += 1
            if len(rows) >= batch_size:
                flush_batch()
            if progress_every and num_finished % progress_every == 0:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{ts}] {first_word}{correction_phrase} {num_finished:,} out of {num_csvs:,} files...",
                    flush=True
                )

    flush_batch() # Final flush
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{ts}] {first_word}{correction_phrase} all {num_csvs:,} files.",
        flush=True
    )
    if writer is not None:
        writer.close()

if __name__ == "__main__":
    args = parse_args()
    make_df(
        args.partition,
        args.df_type,
        args.correct_params,
        args.max_workers,
        args.chunksize,
        args.batch_size,
        args.progress_every
    )
