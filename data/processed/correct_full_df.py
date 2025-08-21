import argparse
import numpy as np
import pandas as pd
import pickle
from numpy.polynomial.polynomial import polyval
from pathlib import Path

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
        description="Correct the data in a full data frame using the correction factor polynomials."
    )
    parser.add_argument(
        "--partition",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Number of the partition to use (1-5)."
    )
    return parser.parse_args()

def correct_full_df(partition: int) -> None:
    partition_dir = Path(f"partition{partition}")
    full_df = pd.read_parquet(partition_dir / "full_df.parquet")
    for param in PARAMS:
        full_df[param] /= polyval(full_df["HC_ANGLE"], poly_coefs[param])
    full_df.to_parquet(partition_dir / "corrected_full_df.parquet")

if __name__ == "__main__":
    args = parse_args()
    correct_full_df(args.partition)
