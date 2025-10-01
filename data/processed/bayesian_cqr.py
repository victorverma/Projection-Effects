import numpy as np
import pymc as pm
from arviz import InferenceData
from numpy.random import Generator
from numpy.typing import ArrayLike
from sklearn.utils import check_X_y

class SameSlopesBayesianCQR():
    def __init__(
        self,
        *,
        quantile_levels: ArrayLike = (0.5,)
    ):
        self.quantile_levels = quantile_levels

    def _check_quantile_levels(self) -> None:
        try:
            quantile_levels_ = np.asarray(self.quantile_levels, dtype=float)
        except:
            raise TypeError(
                f"`quantile_levels` ({self.quantile_levels}) "
                "must be array-like."
            )
        
        if not quantile_levels_.size:
            raise ValueError("`quantile_levels` must be non-empty.")
        if quantile_levels_.ndim > 1:
            raise ValueError(
                f"`quantile_levels` ({self.quantile_levels}) must be 1D."
            )
        if np.any((quantile_levels_ <= 0) | (quantile_levels_ >= 1)):
            raise ValueError(
                f"All entries of `quantile_levels` ({self.quantile_levels}) "
                "must be in (0, 1)."
            )
        self.quantile_levels_ = quantile_levels_

    @staticmethod
    def _check_pos_int_arg(
        arg: int | None, arg_name: str, *, can_be_none: bool = False
    ) -> None:
        if not arg is None:
            if not isinstance(arg, int):
                suffix = " or None." if can_be_none else "."
                raise TypeError(f"{arg_name} must be an int{suffix}")
            if arg <= 0:
                raise ValueError(f"{arg_name} must be positive.")

    def _check_random_seed(
        self, *, random_seed: pm.util.RandomState, chains: int | None
    ) -> None:
        if random_seed is None:
            return
        if isinstance(random_seed, (int, np.integer)):
            return
        if isinstance(random_seed, Generator):
            return
        if isinstance(random_seed, (list, tuple, np.ndarray)):
            if not all(
                isinstance(x, (int, np.integer)) for x in random_seed
            ):
                raise TypeError("random_seed must contain ints.")
            if chains is not None and len(random_seed) != chains:
                raise ValueError("random_seed must have one int per chain.")
            return
        raise TypeError(
            "random_seed must be an int, a sequence of ints (one per chain), "
            "a numpy.random.Generator, or None."
        )

    def _check_sample_args(
        self,
        *,
        draws: int,
        tune: int,
        chains: int | None,
        cores: int | None,
        random_seed: pm.util.RandomState,
        target_accept: float
    ) -> None:
        self._check_pos_int_arg(draws, "draws")
        self._check_pos_int_arg(tune, "tune")
        self._check_pos_int_arg(chains, "chains", can_be_none=True)
        self._check_pos_int_arg(cores, "cores", can_be_none=True)

        self._check_random_seed(random_seed=random_seed, chains=chains)

        if not (0 < target_accept < 1):
            raise ValueError("target_accept must be in (0, 1).")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int | None = None,
        cores: int | None = None,
        random_seed: pm.util.RandomState = None,
        target_accept: float = 0.8
    ) -> InferenceData:
        # The types, default values, and bounds for draws, tune, chains, cores,
        # and random_seed are from
        # https://www.pymc.io/projects/docs/en/v5.23.0/api/generated/pymc.sample.html
        #
        # The type, default value, and bounds for target_accept are from
        # https://www.pymc.io/projects/docs/en/v5.23.0/api/generated/classmethods/pymc.step_methods.hmc.NUTS.__init__.html

        self._check_quantile_levels()
        X, y = check_X_y(X, y, y_numeric=True)
        self._check_sample_args(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            target_accept=target_accept
        )

        quantile_levels_repeated = self.quantile_levels_[:, None]
        kappa = np.sqrt(
            quantile_levels_repeated / (1 - quantile_levels_repeated)
        )
        b = np.sqrt(
            quantile_levels_repeated * (1 - quantile_levels_repeated)
        )
        observed = y[None, :]

        with pm.Model():
            beta_0_ = pm.Flat("beta_0", shape=len(self.quantile_levels_))
            beta_ = pm.Flat("beta", shape=X.shape[1])
            mu = beta_0_[:, None] + X @ beta_
            # Based on the example at https://www.pymc.io/projects/examples/en/latest/bart/bart_quantile_regression.html,
            # it seems like mu and possibly other parameters can be 2D, so maybe
            # repeating, tiling, etc. aren't necessary. However, the
            # documentation doesn't seem to say that those parameters can be 2D.
            pm.AsymmetricLaplace(
                "y", kappa=kappa, mu=mu, b=b, observed=observed
            )

            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                target_accept=target_accept
            )

        return idata
