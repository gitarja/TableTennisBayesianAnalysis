import pymc as pm
import xarray as xr
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
def OutliersDetection(X, y):
    labels = np.ones_like(y)
    with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = pm.HalfCauchy("sigma", beta=10)
        intercept = pm.Normal("intercept", 0, sigma=1)
        slope = pm.Normal("slope", 0, sigma=1)

        # Define likelihood
        mu = pm.Deterministic("mu", intercept + slope * X)
        likelihood = pm.StudentT("y", nu=3, mu=mu, sigma=sigma, observed=y)
        # likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        trace = pm.sample(tune=500, chains=4, cores=7, random_seed=100, draws=1000)

    post = az.extract(trace.posterior)
    preds = post["intercept"] + post["slope"] * xr.DataArray(X)


    mean = np.average(preds, axis=0)
    min_pred = np.min(preds, axis=0)
    max_pred = np.max(preds, axis=0)





    outlier_idx =  ~((y >  min_pred) & (y < max_pred))
    labels[(outlier_idx) & (y > mean)] = 3
    labels[(outlier_idx) & (y < mean)] = 2


    # show the linear reg
    # plt.plot(X, mean, color="black")
    # # plt.plot(X, preds.transpose(), alpha=0.01, color="C1")
    # plt.scatter(X[labels==1], y[labels==1], label="control", color="#377eb8")
    # plt.scatter(X[labels == 2], y[labels == 2], label="overestimate", color="#e41a1c")
    # plt.scatter(X[labels == 3], y[labels == 3], label="underestimate", color="#4daf4a")
    #
    # plt.legend(loc=0)
    # plt.show()
    return labels



