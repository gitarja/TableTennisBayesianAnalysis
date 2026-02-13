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

    inefficient_ja, _ = inefficient_reader.getSelfReportFeatures()

    # efficient group

    efficient_ja, _ = efficient_reader.getSelfReportFeatures()

    inefficient_ja["group"] = "inefficient"
    efficient_ja["group"] = "efficient"
    df = pd.concat([inefficient_ja, efficient_ja])

    features = [
        "double_self_report_score",
        "double_team_score",
        "double_facilitating_skill",
        "double_partner_skill",
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


        subjects_idx = np.unique(np.concatenate([clean_df["double_subject_id"].values]))

        gender_idx = np.unique(np.concatenate([clean_df["double_subject_gender"].values]))

        order_play_idx = np.unique(np.concatenate([clean_df["double_order_play"].values]))

        skill_compt_idx = np.unique(np.concatenate([clean_df["double_skill_comp"].values]))


        # factorize
        ineff_subjects_idx = np.searchsorted(subjects_idx, clean_df.loc[clean_df["group"] == "inefficient"]["double_subject_id"].values)
        eff_subjects_idx = np.searchsorted(subjects_idx, clean_df.loc[clean_df["group"] == "efficient"][
            "double_subject_id"].values)

        ineff_gender_idx = np.searchsorted(gender_idx, clean_df.loc[clean_df["group"] == "inefficient"][
            "double_subject_gender"].values)
        eff_gender_idx = np.searchsorted(gender_idx, clean_df.loc[clean_df["group"] == "efficient"][
            "double_subject_gender"].values)

        ineff_order_play_idx = np.searchsorted(order_play_idx, clean_df.loc[clean_df["group"] == "inefficient"][
            "double_order_play"].values)
        eff_order_play_idx = np.searchsorted(order_play_idx, clean_df.loc[clean_df["group"] == "efficient"][
            "double_order_play"].values)

        ineff_skill_comp_idx = np.searchsorted(skill_compt_idx, clean_df.loc[clean_df["group"] == "inefficient"][
            "double_skill_comp"].values)
        eff_skill_comp_idx = np.searchsorted(skill_compt_idx, clean_df.loc[clean_df["group"] == "efficient"][
            "double_skill_comp"].values)



        clean_df[analyzed_features].values.astype(int)
        inefficient_obv = clean_df.loc[clean_df["group"] == "inefficient"][analyzed_features].values.astype(int)
        efficient_obv = clean_df.loc[clean_df["group"] == "efficient"][analyzed_features].values.astype(int)

        inefficient_X = clean_df.loc[clean_df["group"] == "inefficient"]["double_score"].values
        efficient_X = clean_df.loc[clean_df["group"] == "efficient"]["double_score"].values

        inefficient_obv = inefficient_obv - 1
        efficient_obv = efficient_obv - 1
        # az.plot_dist(inefficient_obv)
        # plt.show()
        # az.plot_dist(efficient_obv)
        # plt.show()

        coords = {"subject_idx": subjects_idx,
                  "subject_genders": range(2),
                  "order_play": range(2),
                  "skill_comp": range(2),
                  }
        # K_efficient = len(np.unique(efficient_obv))
        # K_inefficient = len(np.unique(inefficient_obv))

        K = 7
        sigma_low = 10 ** -1
        sigma_high = 10
        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            # continous

            # random intercept
            subjects_intercept = pm.Normal("subjects_intercept", 0, 0.1, dims="subject_idx")

            gender_intercept = pm.Normal("subjects_gender_intercept", 0, 0.1,
                                                  dims="subject_genders")

            order_play_intercept = pm.Normal("order_play_intercept", 0, 0.1,
                                             dims="order_play")

            skill_component_intercept =  pm.Normal("skill_component_intercept", 0, 0.1,
                                             dims="skill_comp")

            inefficient_mean = pm.Normal('inefficient_mean', mu=0, sigma=0.5)

            efficient_mean = pm.Normal('efficient_mean', mu=0, sigma=0.5)

            # inefficient_std = pm.HalfCauchy("inefficient_std", .5)
            # efficient_std = pm.HalfCauchy("efficient_std", .5)

            inefficient_mu = pm.Deterministic("inefficient_mu",
                                              inefficient_mean + subjects_intercept[
                                                  ineff_subjects_idx] +
                                              gender_intercept[ineff_gender_idx] + order_play_intercept[
                                                  ineff_order_play_idx] + skill_component_intercept[ineff_skill_comp_idx])

            efficient_mu = pm.Deterministic("efficient_mu",
                                            efficient_mean + subjects_intercept[eff_subjects_idx] +
                                            gender_intercept[eff_gender_idx] + order_play_intercept[
                                                eff_order_play_idx] + skill_component_intercept[eff_skill_comp_idx])

            # using different cutpoints caused spreading posterior
            cutpoints_mu = np.linspace(-1, 1, K-1) # the questionarie ranges from 1-7
            cutpoints = pm.Normal("cutpoints", mu=cutpoints_mu, sigma=1,
                                  transform=pm.distributions.transforms.ordered)

            inefficient = pm.OrderedProbit("inefficient", eta=inefficient_mu, cutpoints=cutpoints,
                                           observed=inefficient_obv, sigma=1)

            efficient = pm.OrderedProbit("efficient", eta=efficient_mu, cutpoints=cutpoints,
                                         observed=efficient_obv, sigma=1)

            diff_of_means = pm.Deterministic("difference_of_means", efficient_mean - inefficient_mean)
            effect_size = pm.Deterministic(
                "effect_size", diff_of_means / 1
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
