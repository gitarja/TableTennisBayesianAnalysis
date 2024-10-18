import numpy as np
import xarray as xr
from pymc import HalfCauchy, Model, Normal, sample, Deterministic, StudentT,Exponential

np.random.seed(1945)
def OutliersDetection(X, y):
    labels = np.ones_like(y)
    with Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = HalfCauchy("sigma", beta=10)
        intercept = Normal("intercept", 0, sigma=20)
        slope = Normal("slope", 0, sigma=20)

        # Define nu
        nu_minus_one = Exponential("nu_minus_one", 1 / 29.0)
        nu = Deterministic("nu", nu_minus_one + 1)
        nu_log10 = Deterministic("nu_log10", np.log10(nu))

        # Define likelihood
        mu = Deterministic("mu", intercept + slope * X)
        likelihood = StudentT("y", mu=mu, sigma=sigma, nu=5, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        idata = sample(4000, cores=4, chains=4)

    # idata.posterior["y_model"] = idata.posterior["Intercept"] + idata.posterior["slope"] * xr.DataArray(X)

    # posterior predictive
    isamples = np.random.randint(0, len(idata) - 1, 100)
    post_pred = idata.posterior['intercept'][isamples] + idata.posterior['slope'][isamples] * xr.DataArray(X)

    y_pred = post_pred.data.reshape((-1, len(X)))

    y_min = np.min(y_pred, axis=0)
    y_max = np.max(y_pred, axis=0)
    y_preds_mean = np.average(y_pred, axis=0)
    y_preds_std = np.std(y_pred, axis=0)
    inlier_mask = ((y >= y_min) & (y <= y_max))
    outlier_mask = ~inlier_mask


    labels[(outlier_mask == True) & (y_preds_mean > y)] = 2  # it predicts greater than the actual
    labels[(outlier_mask == True) & (y_preds_mean < y)] = 3  # it predicts lower than the actual


    return labels




