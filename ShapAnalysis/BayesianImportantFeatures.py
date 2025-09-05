from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH
from Double.GlobalFeaturesReader import ImpressionFeatures
import pymc as pm
import pandas as pd
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST
import arviz as az
import pickle
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(1945)  # For Replicability

if __name__ == '__main__':

    lower_group, upper_group = groupLabeling()

    # lower group
    lower_reader = ImpressionFeatures(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                      file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                      include_subjects=lower_group, exclude_failure=False,
                                      exclude_no_pair=True)
    # upper group
    upper_reader = ImpressionFeatures(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                      file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                      include_subjects=upper_group, exclude_failure=False,
                                      exclude_no_pair=True)

    label = "all_lower_upper"
    # mod = "skill_personal_perception_action_impact_ecg_me"
    mod = "skill_personal_perception_action_impact_ecg_me"
    lower_features, skill_lower = lower_reader.getImpressionFeatures(group="lower",
                                                                     mod=mod,
                                                                     return_group_skill=True, return_control=True)

    upper_features, skill_upper = upper_reader.getImpressionFeatures(group="upper",
                                                                     mod=mod,
                                                                     return_group_skill=True, return_control=True)

    X_lower = lower_features.loc[:, lower_features.columns != 'labels']
    y_lower = lower_features["labels"].values

    X_upper = upper_features.loc[:, upper_features.columns != 'labels']
    y_upper = upper_features["labels"].values

    X_upper["group"] = "efficient"
    X_lower["group"] = "inefficient"
    X = pd.concat([X_lower, X_upper])

    subjects_idx = np.unique(np.concatenate([X["subject1"].values, X["subject2"].values]))
    # subject inefficient
    ineff_s1_idx = np.searchsorted(subjects_idx,
        X.loc[X["group"] == "inefficient"]["subject1"].values)

    ineff_s2_idx = np.searchsorted(subjects_idx,
        X.loc[X["group"] == "inefficient"]["subject2"].values)


    # subject efficient
    eff_s1_idx = np.searchsorted(subjects_idx,
        X.loc[X["group"] == "efficient"]["subject1"].values)
    eff_s2_idx = np.searchsorted(subjects_idx,
        X.loc[X["group"] == "efficient"]["subject2"].values)


    features = [
        "p1_al_prec_sim",
        "p1_cs_mean",
        "im_racket_ball_wrist_mean",
        "im_racket_ball_angle_sim",
        "p1_al_prec_mean",
        "me_whole_mean",
        "p2_cs_mean",
        "p1_al_onset_mean",
        "p2_cs_sim",
        "im_racket_ball_angle_mean",
        "ecg_lfhf_sim",
        "ec_start_fs_sim",
        "p2_al_onset_mean",
        "im_ball_updown_sim",
        "me_whole_sim"



    ]
    for f in features:
        scaler = StandardScaler()
        average_scaled = scaler.fit_transform(X[f].values.reshape(-1, 1))
        X[f] = average_scaled.flatten()

        mu_m = X[f].mean()
        mu_s = X[f].std() * 2
        inefficient_obv = X.loc[X["group"] == "inefficient"][f].values
        efficient_obv = X.loc[X["group"] == "efficient"][f].values
        print(np.average(inefficient_obv))
        print(np.average(efficient_obv))
        y_min = np.min(X[f].values)
        coords = {
            # subjects id
            "subject_idx": subjects_idx,
        }

        # az.plot_dist(X[f].values)
        #
        # plt.show()
        with pm.Model(coords=coords) as model:
            # random intercept
            subjects_intercept = pm.Normal("subjects_intercept", 0, 0.1, dims="subject_idx")

            inefficient_mean = pm.Normal('inefficient_mean', mu=mu_m, sigma=1)
            efficient_mean = pm.Normal('efficient_mean', mu=mu_m, sigma=1)

            inefficient_std = pm.HalfCauchy("inefficient_std", 1)
            efficient_std = pm.HalfCauchy("efficient_std", 1)

            inefficient_latent = pm.Deterministic("iefficient_latent", inefficient_mean
                                                  + (
                                                          subjects_intercept[ineff_s1_idx] +
                                                          subjects_intercept[ineff_s2_idx]) / 2)

            efficient_latent = pm.Deterministic("efficient_latent", efficient_mean
                                                + (
                                                        subjects_intercept[eff_s1_idx] +
                                                        subjects_intercept[eff_s2_idx]) / 2)



            nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            nu = pm.Deterministic("nu", nu_minus_one + 1)
            nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

            lambda_1 = efficient_std ** -2
            lambda_2 = inefficient_std ** -2
            inefficient = pm.StudentT("inefficient", nu=nu,
                                      mu=inefficient_latent, lam=lambda_2,
                                      observed=inefficient_obv)
            efficient = pm.StudentT("efficient", nu=nu, mu=efficient_latent,
                                    lam=lambda_1, observed=efficient_obv)



            diff_of_means = pm.Deterministic("difference_of_means", efficient_mean - inefficient_mean)
            diff_of_stds = pm.Deterministic("difference_of_stds", efficient_std - inefficient_std)
            effect_size = pm.Deterministic(
                "effect_size", diff_of_means / np.sqrt((inefficient_std ** 2 + efficient_std ** 2) / 2)
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
        hierarchical_loo = az.plot_ppc(idata)
        plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "PPC_supp\\" + f + ".png", format='png')
        plt.close()

        trace_post = az.extract(idata.posterior)

        # save the model
        with open(DOUBLE_RESULTS_PATH_TTEST + "model_supp\\" + "idata_" + f + "_study1.pkl", 'wb') as handle:
            print("write data into: " + "idata_ttest_" + f + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del model
        del idata
