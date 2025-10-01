import numpy as np
import pandas as pd
from arviz import InferenceData
from bayesian_cqr import SameSlopesBayesianCQR
from datetime import datetime
from numpy.typing import ArrayLike
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from typing import Any, Callable

PARAMS = [
    "ABSNJZH", "EPSX", "EPSY", "EPSZ", "MEANALP", "MEANGAM", "MEANGBH",
    "MEANGBT", "MEANGBZ", "MEANJZD", "MEANJZH", "MEANPOT", "MEANSHR", "R_VALUE",
    "SAVNCPP", "SHRGT45", "TOTBSQ", "TOTFX", "TOTFY", "TOTFZ", "TOTPOT",
    "TOTUSJH", "TOTUSJZ", "USFLUX"
]

def calc_posterior_mean_quantiles(
    quantile_levels: ArrayLike,
    poly_degree: int,
    idata: InferenceData,
    hc_angles: ArrayLike
) -> pd.DataFrame:
    X = (
        PolynomialFeatures(degree=poly_degree, include_bias=False)
        .fit_transform(np.deg2rad(hc_angles[:, None]))
    )

    sample_dfs = []
    for sample_num in [-1]:
        beta_0 = np.mean(idata.posterior.beta_0.values[0, :, :], axis=0)
        beta = np.mean(idata.posterior.beta.values[0, :, :], axis=0)
        quantile_dfs = []
        for quantile_num, quantile_level in enumerate(quantile_levels):
            quantile_df = pd.DataFrame({
                "sample_num": sample_num,
                "quantile_level": quantile_level,
                "HC_ANGLE": hc_angles,
                "quantile": beta_0[quantile_num] + X @ beta
            })
            quantile_dfs.append(quantile_df)
        sample_df = pd.concat(quantile_dfs, ignore_index=True)
        sample_dfs.append(sample_df)
    samples_df = pd.concat(sample_dfs, ignore_index=True)

    return samples_df

def make_cdf(points: pd.DataFrame) -> Callable[[pd.Series], pd.Series]:
    """Construct a CDF passing through given points."""
    points.reset_index(drop=True, inplace=True)

    def cdf(quantiles: pd.Series) -> pd.Series:
        i = np.searchsorted(points["quantile"], quantiles, side="right")
        quantile_levels = np.where(
            i == 0,
            0,
            np.where(
                i == len(points),
                1,
                points.loc[np.clip(i - 1, 0, len(points) - 1), "quantile_level"]
            )
        )
        return pd.Series(quantile_levels)

    return cdf

def make_inverse_cdf(points: pd.DataFrame) -> Callable[[pd.Series], pd.Series]:
    """Construct an inverse CDF passing through given points."""
    points.reset_index(drop=True, inplace=True)

    def inverse_cdf(quantile_levels: pd.Series) -> pd.Series:            
        i = np.searchsorted(points["quantile_level"], quantile_levels)
        quantiles = np.where(
            i == 0,
            points["quantile"].iloc[0],
            np.where(
                i == len(points),
                points["quantile"].iloc[-1],
                points["quantile"].iloc[np.clip(i, 0, len(points) - 1)]
            )
        )
        return pd.Series(quantiles)

    return inverse_cdf

def _check_correct_vals_inputs(
    param: str,
    *,
    poly_degree: int,
    target_hc_angle: float,
    round_to_nearest: float,
    n_train: int
) -> None:
    if param not in PARAMS:
        raise ValueError(f"`param` ({param}) isn't in `PARAMS` ({PARAMS}).")

    if not isinstance(poly_degree, int):
        raise TypeError(f"`poly_degree` ({poly_degree}) must be an integer.")
    if poly_degree <= 0:
        raise ValueError(f"`poly_degree` ({poly_degree}) must be positive.")

    if not (0 <= target_hc_angle <= 70):
        raise ValueError(
            f"`target_hc_angle` ({target_hc_angle}) must be in [0, 70]."
        )

    if round_to_nearest <= 0:
        raise ValueError(
            f"`round_to_nearest` ({round_to_nearest}) must be positive."
        )

    if not isinstance(n_train, int):
        raise TypeError(f"`n_train` ({n_train}) must be an integer.")
    if n_train <= 0:
        raise ValueError(f"`n_train` ({n_train}) must be positive.")

def correct_vals(
    param: str,
    quantile_levels: ArrayLike,
    *,
    poly_degree: int,
    target_hc_angle: float,
    round_to_nearest: float,
    n_train: int,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    random_seed: Any,
    target_accept: float
) -> pd.Series:
    #################
    # Validate inputs
    #################

    _check_correct_vals_inputs(
        param,
        poly_degree=poly_degree,
        target_hc_angle=target_hc_angle,
        round_to_nearest=round_to_nearest,
        n_train=n_train
    )
    # quantile_levels will be validated by
    # BayesianCompositeQuantileRegression._check_quantile_levels().
    # draws, tune, chains, cores, random_seed, and target_accept will be
    # validated by BayesianCompositeQuantileRegression._check_sample_args()

    ###########################################################
    # Identify and extract the records that should be corrected
    ###########################################################

    recs_to_correct_mask = (
        all_ars_df["IS_TMFI"] &
        all_ars_df[["HC_ANGLE", param]].notna().all(axis=1)
    )
    recs_to_correct = (
        all_ars_df
        .loc[recs_to_correct_mask, ["HC_ANGLE", "HC_ANGLE_radians", param]]
        .copy()
    )

    ################################
    # Group records by rounded angle
    ################################

    recs_to_correct["HC_ANGLE_group"] = (
        (recs_to_correct["HC_ANGLE"] / round_to_nearest).round().astype(int)
    )
    recs_to_correct["HC_ANGLE_rounded"] = (
        round_to_nearest * recs_to_correct["HC_ANGLE_group"]
    )

    ############################################################################
    # Get the rounded angles at which quantiles will be computed under the model 
    ############################################################################

    hc_angles = recs_to_correct["HC_ANGLE_rounded"].unique()
    hc_angle_diffs = recs_to_correct["HC_ANGLE"].sort_values().diff()
    _2tol = hc_angle_diffs[hc_angle_diffs > 0].min()
    # target_hc_angle has at most one match in hc_angles since two matches would
    # be within _2tol of each other, which is impossible.
    if np.all(np.abs(hc_angles - target_hc_angle) >= _2tol / 2):
        hc_angles = np.append(hc_angles, target_hc_angle)
    hc_angles = np.sort(hc_angles)

    ###############
    # Fit the model
    ###############

    reg = SameSlopesBayesianCQR(
        quantile_levels=quantile_levels,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        target_accept=target_accept
    )

    recs_to_correct_subset = recs_to_correct.sample(
        n_train, random_state=random_seed # Don't reuse random_seed here
    )
    X = (
        PolynomialFeatures(degree=poly_degree, include_bias=False)
        .fit_transform(recs_to_correct_subset[["HC_ANGLE_radians"]])
    )
    y = recs_to_correct_subset[param] # standardize somehow to deal with huge values

    idata = reg.fit(X, y)

    #####################################
    # Calculate quantiles under the model
    #####################################

    posterior_mean_quantiles = calc_posterior_mean_quantiles(
        quantile_levels, poly_degree, idata, hc_angles
    )

    #######################################################
    # Construct the inverse CDF for the target distribution
    #######################################################

    target_mask = (
        (posterior_mean_quantiles["HC_ANGLE"] - target_hc_angle).abs() <
        _2tol / 2
    )
    target_inverse_cdf = make_inverse_cdf(posterior_mean_quantiles[target_mask])

    #################################################
    # Construct the CDFs for the source distributions
    #################################################

    source_cdfs = (
        posterior_mean_quantiles
        .assign(HC_ANGLE_group=lambda df: (
            (df["HC_ANGLE"] / round_to_nearest).round().astype(int)
        ))
        .groupby("HC_ANGLE_group", observed=True)
        [["quantile_level", "quantile"]]
        .apply(make_cdf)
        .to_dict()
    )

    ####################################################################
    # Correct each group using its source CDF and the target inverse CDF
    ####################################################################

    corrected_vals = (
        recs_to_correct
        .groupby("HC_ANGLE_group", observed=True)
        [param]
        .transform(lambda s: target_inverse_cdf(source_cdfs[s.name](s)))
    )

    all_vals = all_ars_df[param].copy()
    all_vals[recs_to_correct_mask] = corrected_vals
    all_vals

    return all_vals, idata

if __name__ == "__main__":
    all_ars_df = pd.read_parquet("all_ars_df.parquet")
    all_ars_df["has_nas"] = all_ars_df[["HC_ANGLE"] + PARAMS].isna().any(axis=1)
    all_ars_df["HC_ANGLE_radians"] = np.deg2rad(all_ars_df["HC_ANGLE"])

    # I should make the parameters of correct_vals command-line arguments
    for param in PARAMS:
        all_vals, idata = correct_vals(
            param,
            np.linspace(0.1, 0.9, 3),
            poly_degree=10,
            target_hc_angle=0.0,
            round_to_nearest=1,
            n_train=1000,
            draws=1000, # increase
            tune=1000,
            chains=1,  # 1 chain is okay
            cores=1,
            random_seed=1,
            target_accept=0.9
        )
        all_ars_df[param] = all_vals
        idata.to_netcdf(Path("idatas") / f"{param}.nc")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Corrected {param}...\n", flush=True)

    all_ars_df.to_parquet("all_ars_df_corrected.parquet")
