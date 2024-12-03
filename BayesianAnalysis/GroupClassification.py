import pandas as pd
import pymc as pm
import xarray as xr
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
from Utils.Conf import SINGLE_FEATURES_FILE_PATH, DOUBLE_SUMMARY_FILE_PATH
def outliersDetection(X, y):
    labels = np.ones_like(y)
    with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = pm.HalfCauchy("sigma", beta=10)
        intercept = pm.Normal("intercept", 0, sigma=5)
        slope = pm.Normal("slope", 0, sigma=5)

        # Define likelihood
        mu = pm.Deterministic("mu", intercept + slope * X)
        likelihood = pm.StudentT("y", nu=3, mu=mu, sigma=sigma, observed=y)
        # likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        idata = pm.sample(tune=500, chains=4, cores=4, random_seed=100, draws=1000, target_accept=0.95, idata_kwargs={"log_likelihood": True})

    hierarchical_loo = az.loo(idata)

    print(hierarchical_loo)
    post = az.extract(idata.posterior)
    preds = post["intercept"] + post["slope"] * xr.DataArray(X)

    std = np.std(preds, axis=0) * 3 # ref https://doi.org/10.1109/TIP.2008.926150
    mean = np.average(preds, axis=0)
    # min_pred = np.min(preds, axis=0)
    # max_pred = np.max(preds, axis=0)
    min_pred = mean - std
    max_pred = mean + std

    outlier_idx =  ~((y >  min_pred) & (y < max_pred))
    labels[(outlier_idx) & (y > mean)] = 3 # efficient
    labels[(outlier_idx) & (y < mean)] = 2 #  inefficient

    # show the linear reg
    # plt.plot(X, mean, color="black")
    # sortedX_idx = np.argsort(X)
    # plt.fill_between(X[sortedX_idx], mean[sortedX_idx] - std[sortedX_idx], mean[sortedX_idx] + std[sortedX_idx], alpha=.1, color="#377eb8")
    # # plt.plot(X, preds.transpose(), alpha=0.01, color="C1")
    # plt.scatter(X[labels==1], y[labels==1], label="control", color="#377eb8", s=30)
    #
    # plt.scatter(X[labels == 2], y[labels == 2], label="overestimate", color="#e41a1c", s=30)
    # plt.scatter(X[labels == 3], y[labels == 3], label="underestimate", color="#4daf4a", s=30)
    #
    # # plt.legend(loc=0)
    # plt.show()
    return labels



def outliersLabeling(X, y):
    labels = np.ones_like(y) * 3
    with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = pm.HalfCauchy("sigma", beta=10)
        intercept = pm.Normal("intercept", 0, sigma=5)
        slope = pm.Normal("slope", 0, sigma=5)

        # Define likelihood
        mu = pm.Deterministic("mu", intercept + slope * X)
        likelihood = pm.StudentT("y", nu=3, mu=mu, sigma=sigma, observed=y)
        # likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        idata = pm.sample(tune=500, chains=4, cores=4, random_seed=100, draws=1000, target_accept=0.95, idata_kwargs={"log_likelihood": True})

    hierarchical_loo = az.loo(idata)

    # print(hierarchical_loo)
    post = az.extract(idata.posterior)
    preds = post["intercept"] + post["slope"] * xr.DataArray(X)

    std = np.std(preds, axis=0) * 0.25 # ref https://doi.org/10.1109/TIP.2008.926150
    mean = np.average(preds, axis=0)


    min_pred = mean - std
    max_pred = mean + std

    outlier_idx =  ~((y >  min_pred) & (y < max_pred))
    labels[(outlier_idx) & (y > mean)] = 1 # efficient
    labels[(outlier_idx) & (y < mean)] = 0 #  inefficient

    # labels[ (y > mean)] = 1 # upper
    # labels[ (y < mean)] = 0 #  lower

    # # show the linear reg
    # plt.plot(X, mean, color="black")
    # sortedX_idx = np.argsort(X)
    # plt.fill_between(X[sortedX_idx], mean[sortedX_idx] - std[sortedX_idx], mean[sortedX_idx] + std[sortedX_idx], alpha=.1, color="#377eb8")
    #
    #
    # plt.scatter(X[labels == 0], y[labels == 0], label="overestimate", color="#e41a1c", s=20)
    # plt.scatter(X[labels == 1], y[labels == 1], label="underestimate", color="#4daf4a", s=20)
    #
    # # plt.legend(loc=0)
    # plt.show()


    return labels




def movementSimilarity(X, feature_col="ec_fs_ball_racket_ratio"):
    single_df = pd.read_pickle(SINGLE_FEATURES_FILE_PATH)
    double_summary = pd.read_csv(DOUBLE_SUMMARY_FILE_PATH)

    double_summary_X = double_summary.loc[double_summary.file_name.isin(X)]
    labels = np.zeros(len(double_summary_X))
    sim_scores = []
    pval_scores = []
    for index, row in double_summary_X.iterrows():
        subject_1_sample = single_df.loc[single_df["id_subject"] == row["Subject1"]]
        subject_2_sample =  single_df.loc[single_df["id_subject"] == row["Subject2"]]

        sim_score = stats.ks_2samp(subject_1_sample[feature_col].values, subject_2_sample[feature_col].values)

        sim_scores.append(sim_score.statistic)
        pval_scores.append(sim_score.pvalue)

    sim_scores = np.asarray(sim_scores)
    pval_scores = np.asarray(pval_scores)

    labels[pval_scores < 0.05] = 1
    return sim_scores, pval_scores, labels





