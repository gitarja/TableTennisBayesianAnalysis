import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, features_explanation
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST
import arviz as az
import pickle
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20

if __name__ == '__main__':

    inefficient_group, efficient_group = groupLabeling()

    # inefficient group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=inefficient_group, exclude_failure=True,
                                                    exclude_no_pair=False, hmm_probs=True)

    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=efficient_group, exclude_failure=True,
                                                  exclude_no_pair=False, hmm_probs=True)

    _, inefficient_ia = inefficient_reader.getSelfReportFeatures()

    # efficient group

    _, efficient_ia = efficient_reader.getSelfReportFeatures()

    inefficient_ia["group"] = "inefficient"
    efficient_ia["group"] = "efficient"
    df = pd.concat([inefficient_ia, efficient_ia])

    features = [

        "subject_indv_myskill",

    ]
    efficient_mean_list = []
    inefficient_mean_list = []
    efficient_hdi_list = []
    inefficient_hdi_list = []
    effect_size_mean_list = []
    effect_size_hdi_list = []
    features_list = []

    for f in features:
        print(f)
        analyzed_features = f
        clean_df = df.dropna(subset=[analyzed_features])

        # mean and std ori
        mean_ori = np.nanmean(clean_df[analyzed_features].values.reshape(-1, 1))
        std_ori = np.nanstd(clean_df[analyzed_features].values.reshape(-1, 1))

        mu_m = clean_df[analyzed_features].mean()
        mu_s = clean_df[analyzed_features].std() * 2

        subjects_idx = np.unique(np.concatenate([clean_df["subject_indv_id"].values]))

        gender_idx = np.unique(np.concatenate([clean_df["subject_indv_gender"].values]))
        # idx
        ineff_subjects_idx = np.searchsorted(subjects_idx, clean_df.loc[clean_df["group"] == "inefficient"][
            "subject_indv_id"].values)
        eff_subjects_idx = np.searchsorted(subjects_idx, clean_df.loc[clean_df["group"] == "efficient"][
            "subject_indv_id"].values)
        # gender
        ineff_gender_idx = np.searchsorted(gender_idx, clean_df.loc[clean_df["group"] == "inefficient"][
            "subject_indv_gender"].values)
        eff_gender_idx = np.searchsorted(gender_idx, clean_df.loc[clean_df["group"] == "efficient"][
            "subject_indv_gender"].values)
        # order of trial

        inefficient_obv = clean_df.loc[clean_df["group"] == "inefficient"][analyzed_features].values.astype(int)
        efficient_obv = clean_df.loc[clean_df["group"] == "efficient"][analyzed_features].values.astype(int)



        inefficient_obv = inefficient_obv - 1
        efficient_obv = efficient_obv - 1

        coords = {"subject_idx": subjects_idx,

                  "subject_genders": range(2),

                  }
        # K_efficient = len(np.unique(efficient_obv))
        # K_inefficient = len(np.unique(inefficient_obv))

        K =  len(np.unique(clean_df[analyzed_features].values))
        sigma_low = 10 ** -1
        sigma_high = 10
        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            # continous

            # random intercept
            subjects_intercept = pm.Normal("subjects_intercept", 0, 0.1, dims="subject_idx")

            subjects_gender_intercept = pm.Normal("subjects_gender_intercept", 0, 0.1,
                                                        dims="subject_genders")


             # using STD led to unconverge


            inefficient_mean = pm.Normal('inefficient_mean', mu=0, sigma=0.5)

            efficient_mean = pm.Normal('efficient_mean', mu=0, sigma=0.5)


            inefficient_mu = pm.Deterministic("inefficient_mu",
                                              inefficient_mean  + subjects_intercept[ineff_subjects_idx] +
                                              subjects_gender_intercept[ineff_gender_idx] )

            efficient_mu = pm.Deterministic("efficient_mu", efficient_mean  + subjects_intercept[eff_subjects_idx] +
                                            subjects_gender_intercept[eff_gender_idx] )

            # using different cutpoints caused spreading posterior
            cutpoints_mu = np.linspace(-1, 1, K - 1)
            cutpoints = pm.Normal("cutpoints", mu=cutpoints_mu, sigma=1,
                                            transform=pm.distributions.transforms.ordered)



            inefficient = pm.OrderedProbit("inefficient", eta=inefficient_mu, cutpoints=cutpoints,
                                             observed=inefficient_obv, sigma=1)


            efficient = pm.OrderedProbit("efficient", eta=efficient_mu, cutpoints=cutpoints,
                                           observed=efficient_obv, sigma=1)

            diff_of_means = pm.Deterministic("difference_of_means", efficient_mean - inefficient_mean)

            effect_size = pm.Deterministic(
                "effect_size", diff_of_means
            )

            # debug and sampling
        with model:
            # debug the model
            print(model.debug())
            # pm.model_to_graphviz(model).view()
            # Inference!
            idata = pm.sample_prior_predictive()
            idata.extend(
                pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                          draws=N_SAMPLES,
                          chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
            )
            idata.extend(pm.sample_posterior_predictive(idata))

        # print loo
        hierarchical_loo = az.plot_ppc(idata, kind='cumulative')
        plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "PPC_supp\\" + analyzed_features + ".png", format='png')
        plt.close()

        trace_post = az.extract(idata.posterior)

        # save the model
        with open(DOUBLE_RESULTS_PATH_TTEST + "model_supp\\" + "idata_" + analyzed_features + ".pkl", 'wb') as handle:
            print("write data into: " + "idata_ttest_" + analyzed_features + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del model
        del idata
