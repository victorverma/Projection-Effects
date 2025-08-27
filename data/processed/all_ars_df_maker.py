import numpy as np
import pandas as pd
from pathlib import Path

CSVS_DIR = Path("..") / "raw" / "swan_sf" / "SWAN"
CSV_PATH_PATTERN = Path("partition*") / "*.csv"
TIMES = ["Timestamp"]
ANGLES = ["CRLN_OBS", "CRLT_OBS", "CRVAL1", "CRVAL2", "HC_ANGLE"]
PARAMS = [
    "ABSNJZH", "EPSX", "EPSY", "EPSZ", "MEANALP", "MEANGAM", "MEANGBH",
    "MEANGBT", "MEANGBZ", "MEANJZD", "MEANJZH", "MEANPOT", "MEANSHR", "R_VALUE",
    "SAVNCPP", "SHRGT45", "TOTBSQ", "TOTFX", "TOTFY", "TOTFZ", "TOTPOT",
    "TOTUSJH", "TOTUSJZ", "USFLUX"
]

def make_ar_df(csv_path: Path) -> pd.DataFrame:
    partition = int(csv_path.parent.name[-1])
    ar_num = int(csv_path.stem)

    ar_df = pd.read_csv(
        csv_path,
        sep="\t",
        usecols=TIMES + ANGLES + PARAMS,
        dtype={col: "string" for col in TIMES}
    )
    ar_df.insert(0, "partition", partition)
    ar_df.insert(1, "ar_num", ar_num)
    for col in TIMES:
        ar_df[col] = pd.to_datetime(
            ar_df[col].str.strip(),
            format="%Y-%m-%d %H:%M:%S",
            utc=True,
            errors="raise"
        )
    ar_df[PARAMS] = ar_df[PARAMS].replace([np.inf, -np.inf], np.nan)

    return ar_df

def make_all_ars_df() -> None:
    csvs_iterator = CSVS_DIR.glob(str(CSV_PATH_PATTERN))
    ar_dfs = [make_ar_df(csv_path) for csv_path in csvs_iterator]
    pd.concat(ar_dfs, ignore_index=True).to_parquet("all_ars_df.parquet")

if __name__ == "__main__":
    make_all_ars_df()
