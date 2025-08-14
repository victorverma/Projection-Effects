import argparse
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT_DIR = "swan_sf"
CSV_PATH_PATTERN = "partition[1-5]/*/*.csv"
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a Pandas data frame containing a one-row summary of each SWAN-SF CSV."
    )
    parser.add_argument(
        "--df_type",
        type=str,
        choices=["full", "summary"],
        default="full",
        help="Type of data frame to produce, 'full' (no summarization) or 'summary' (default: 'full')."
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

def process_csv(csv_path: Path, params: list[str] = PARAMS) -> pd.DataFrame:
    """
    Put the data in the given CSV on the given parameters in a data frame.
    """
    partition, type = get_partition_and_type(csv_path)

    usecols = ["Timestamp", "HC_ANGLE"] + params
    csv_df = pd.read_csv(csv_path, sep="\t", usecols=usecols)
    csv_df[params] = csv_df[params].interpolate(
        method="linear", limit_direction="both"
    )

    csv_df.insert(0, "partition", partition)
    csv_df.insert(1, "type", type)
    csv_df.insert(2, "file", csv_path.name)

    return csv_df

def summarize_csv(csv_path: Path, params: list[str] = PARAMS) -> pd.DataFrame:
    """
    Construct a one-row data frame with summary statistics for the specified CSV
    and parameters.
    """
    partition, type = get_partition_and_type(csv_path)

    csv_df = pd.read_csv(csv_path, sep="\t", usecols=params)
    csv_df = csv_df.interpolate(method="linear", limit_direction="both")

    out = {"partition": partition, "type": type, "file": csv_path.name}

    for col in params:
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

def make_summary_df(
        df_type: str,
        max_workers: int,
        chunksize: int,
        batch_size: int,
        progress_every: int
    ) -> None:
    """
    Compute summary statistics for all CSVs and save results to a Parquet file.
    """
    out_path = f"{df_type}_df.parquet"
    if os.path.exists(out_path):
        os.remove(out_path)

    fun = process_csv if df_type == "full" else summarize_csv

    writer = None
    rows = []
    num_summarized = 0

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

    num_files = sum(1 for _ in Path(ROOT_DIR).glob(CSV_PATH_PATTERN))
    files_iter = Path(ROOT_DIR).glob(CSV_PATH_PATTERN)
    with ProcessPoolExecutor(max_workers) as ex:
        for row in ex.map(fun, files_iter, chunksize=chunksize):
            rows.append(row)
            num_summarized += 1
            if len(rows) >= batch_size:
                flush_batch()
            if progress_every and num_summarized % progress_every == 0:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{ts}] Summarized {num_summarized:,} out of {num_files:,} files...",
                    flush=True
                )

    flush_batch() # Final flush
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Summarized all {num_files:,} files.", flush=True)
    if writer is not None:
        writer.close()

if __name__ == "__main__":
    args = parse_args()
    make_summary_df(
        args.df_type,
        args.max_workers,
        args.chunksize,
        args.batch_size,
        args.progress_every
    )
