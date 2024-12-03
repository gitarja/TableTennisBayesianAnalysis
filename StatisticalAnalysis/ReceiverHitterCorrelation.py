import matplotlib.pyplot as plt
from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, features_explanation
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST
import seaborn as sns
import scipy as sp
import pickle
import arviz as az

inefficient_group, efficient_group = groupLabeling()

# sns.set_theme()
# sns.set(font_scale=5)
# sns.set(font="Arial")
# sns.set_style("white")
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams['font.size'] = 20

if __name__ == '__main__':
    # inefficient group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=inefficient_group, exclude_failure=True,
                                                    exclude_no_pair=False, hmm_probs=True)
    inefficient_features = inefficient_reader.getStableUnstableFailureFeatures(group_name="inefficient",
                                                                               success_failure=True,
                                                                               mod="skill_perception_action",
                                                                               with_control=True)
    inefficient_features["group"] = "inefficient"
    # efficient group
    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=efficient_group, exclude_failure=True,
                                                  exclude_no_pair=False, hmm_probs=True)
    efficient_features = efficient_reader.getStableUnstableFailureFeatures(group_name="efficient", success_failure=True,
                                                                           mod="skill_perception_action",
                                                                           with_control=True)
    efficient_features["group"] = "efficient"

    df = pd.concat([inefficient_features, efficient_features])
    th_prec = 200
    efficient_features = efficient_features.dropna(subset=["hitter_p1_al_prec", "receiver_p1_al_prec"])
    efficient_features = efficient_features.loc[(efficient_features["hitter_p1_al_prec"] <= th_prec) & (efficient_features["receiver_p1_al_prec"] <= th_prec)]
    inefficient_features = inefficient_features.dropna(subset=["hitter_p1_al_prec", "receiver_p1_al_prec"])
    inefficient_features = inefficient_features.loc[(inefficient_features["hitter_p1_al_prec"] <= th_prec) & (inefficient_features["receiver_p1_al_prec"] <= th_prec)]

    g = sns.JointGrid()
    x, y = efficient_features["hitter_p1_al_prec"], efficient_features["receiver_p1_al_prec"]
    # sns.scatterplot(x=x, y=y,  s=10, ax=g.ax_joint, alpha=.3)
    # sns.regplot(x=x, y=y, order=1, color="#81b1d3",
    #             line_kws=dict(color="#252525"),
    #             scatter=False, n_boot=1000, ax=g.ax_joint,)
    #
    # print(sp.stats.pearsonr(efficient_features["hitter_p1_al_prec"], efficient_features["receiver_p1_al_prec"]))
    # print(sp.stats.pearsonr(inefficient_features["hitter_p1_al_prec"], inefficient_features["receiver_p1_al_prec"]))
    # sns.histplot(x=x,  linewidth=.75, ax=g.ax_marg_x)
    # sns.histplot(y=y,  linewidth=.75, ax=g.ax_marg_y)
    # plt.show()

    data = np.array([x, y])
    print(data)
    coords = {"axis": ["x", "y"], "axis_bis": ["x", "y"], "obs_id": np.arange(len(x))}
    with pm.Model(coords=coords) as model:
        # Priors for means of x and y
        mu = pm.Normal("mu", mu=0, sigma=10, size=2)

        # Prior for standard deviations

        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=2)
        )
        cov = pm.Deterministic("cov", chol.dot(chol.T), dims=("axis", "axis_bis"))

        # Multivariate normal likelihood
        mvn = pm.MvNormal("mvn", mu=mu, chol=chol, observed=data.T, dims=("obs_id", "axis"))

    with model:
        # debug the model
        # pm.model_to_graphviz(model).view()
        print(model.debug())

        # Inference!
        idata = pm.sample_prior_predictive()

        idata.extend(
            pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True, "dims": {"chol_stds": ["axis"], "chol_corr": ["axis", "axis_bis"]}},
                      draws=N_SAMPLES,
                      chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
        )

    print(az.summary(idata, var_names="~chol", round_to=2))
    az.plot_trace(
        idata,
        var_names="chol_corr",
        coords={"axis": "x", "axis_bis": "y"},

    )
    plt.show()
    # save the model
    with open(DOUBLE_RESULTS_PATH_TTEST + "model\\" + "idata_correlation_p1_al_prec.pkl", 'wb') as handle:
        print("write data into: " + "idata_ttest_idata_correlation_p1_al_prec.pkl")
        pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

