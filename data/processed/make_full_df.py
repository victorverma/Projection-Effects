import argparse
import numpy as np
import os
import pandas as pd
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
import re
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

poly_coefs = pd.DataFrame(columns=PARAMS)
for param in PARAMS:
    with open(f"{POLY_COEFS_DIR}/saved_dictionary_{param}.pkl", "rb") as f:
        # The constant term of 1 seems to be missing, so prepend 1
        poly_coefs[param] = np.insert(pickle.load(f)["Mean Fit"], 0, 1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a data frame containing the data in all the CSVs in a SWAN-SF partition."
    )
    parser.add_argument(
        "--partition",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Number of the partition to use (1-5)."
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
        help="Rows to buffer in memory before appending to the output Parquet file (default: 1000)."
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
        help="Print a progress message every N files (0 to disable, default: 1000)."
    )
    return parser.parse_args()

# See data/processed/csv_name_patterns.ipynb for some work that this is based on
def get_info_from_path(csv_path: Path) -> tuple[int, str, str, int]:
    partition = csv_path.parent.parent.name
    partition = int(partition[-1])
    type = csv_path.parent.name # either FL or NF
    flare_class = "FQ" if csv_path.name[:2] == "FQ" else csv_path.name[0]
    ar_num = int(re.search(r"ar([0-9]+)", csv_path.name).group(1))
    return partition, type, flare_class, ar_num

def correct_params(csv_df: pd.DataFrame) -> pd.DataFrame:
    for param in PARAMS:
        csv_df[param] /= polyval(csv_df["HC_ANGLE"], poly_coefs[param])
    return csv_df

def process_csv(csv_path: Path, *, correct_params_: bool) -> pd.DataFrame:
    """
    Put the data in the given CSV in a data frame. Optionally correct the data.
    """
    partition, type, flare_class, ar_num = get_info_from_path(csv_path)

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
    csv_df.insert(2, "flare_class", flare_class)
    csv_df.insert(3, "ar_num", ar_num)
    csv_df.insert(4, "file", csv_path.name)

    return csv_df

def make_full_df(
        partition: int,
        correct_params_: bool,
        max_workers: int,
        chunksize: int,
        batch_size: int,
        progress_every: int
    ) -> None:
    """
    Make a data frame containing the data in all the CSVs in a SWAN-SF partition.
    """
    out_dir = Path(f"partition{partition}")
    prefix = "corrected_" if correct_params_ else ""
    out_path = out_dir / f"{prefix}full_df.parquet"
    if os.path.exists(out_path):
        os.remove(out_path)

    # partial is used to create functions that are picklable and thus usable by
    # the ProcessPoolExecutor instance. Fixing the value of correct_params_ in a
    # partial call requires correct_params_ to be a keyword argument of
    # process_csv
    process_csv_ = partial(process_csv, correct_params_=correct_params_)

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
        for row in ex.map(process_csv_, files_iter, chunksize=chunksize):
            rows.append(row)
            num_finished += 1
            if len(rows) >= batch_size:
                flush_batch()
            if progress_every and num_finished % progress_every == 0:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{ts}] Processed{correction_phrase} {num_finished:,} out of {num_csvs:,} files...",
                    flush=True
                )

    flush_batch() # Final flush
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{ts}] Processed{correction_phrase} all {num_csvs:,} files.",
        flush=True
    )
    if writer is not None:
        writer.close()

if __name__ == "__main__":
    args = parse_args()
    make_full_df(
        args.partition,
        args.correct_params,
        args.max_workers,
        args.chunksize,
        args.batch_size,
        args.progress_every
    )
