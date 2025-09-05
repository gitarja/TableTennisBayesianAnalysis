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
from BayesianTTestModels import BayesianTTestModel
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
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
    mod = "skill_personal_perception_action_impact_ecg_me_other"
    inefficient_features = inefficient_reader.getStableUnstableFailureFeatures(group_name="inefficient",
                                                                               success_failure=True,
                                                                               mod=mod,
                                                                               with_control=True, timepoint=True,
                                                                               min_group_n=3)
    inefficient_features["group"] = "inefficient"
    # efficient group

    efficient_features = efficient_reader.getStableUnstableFailureFeatures(group_name="efficient", success_failure=True,
                                                                           mod=mod,
                                                                           with_control=True, timepoint=True,
                                                                           min_group_n=3)
    efficient_features["group"] = "efficient"

    df = pd.concat([inefficient_features, efficient_features])
    #
    # features = [
    #     "ecg_hf_sim",
    #     "ecg_lfhf_sim",
    #     "ecg_rmsdd_sim",
    #     "ecg_hf_mean",
    #     "ecg_lfhf_mean",
    #     "ecg_rmsdd_mean",
    #     "hitter_p1_al_onset",
    #     "hitter_p1_al_prec",
    #     "hitter_p1_al_mag",
    #     "hitter_p1_cs",
    #     "hitter_p2_al_onset",
    #     "hitter_p2_al_prec",
    #     "hitter_p2_al_mag",
    #     "hitter_p2_cs",
    #     "hitter_fx_onset",
    #     "hitter_fx_duration",
    #     "receiver_p1_al_onset",
    #     "receiver_p1_al_prec",
    #     "receiver_p1_al_mag",
    #     "receiver_p1_cs",
    #     "receiver_p2_al_onset",
    #     "receiver_p2_al_prec",
    #     "receiver_p2_al_mag",
    #     "receiver_p2_cs",
    #     "receiver_p3_fx_onset",
    #     "receiver_p3_fx_duration",
    #     "receiver_fixation_racket_latency",
    #     "receiver_start_fs",
    #     "receiver_distance_eye_hand",
    #     "receiver_im_ball_wrist",
    #     "receiver_im_racket_ball_wrist",
    #     "receiver_im_racket_ball_angle",
    #     "receiver_im_ball_updown",
    #     "hitter_bouncing_to_partner",
    #     "hitter_bouncing_to_self",
    #     "hitter_at_and_after_hit",
    #     "me_foot_sim",
    #     "me_shoulder_arm_sim",
    #     "me_whole_sim",
    #     "me_foot_mean",
    #     "me_shoulder_arm_mean",
    #     "me_whole_mean",
    #     "individual_skill",
    #     "individual_skill_sim",
    #     "individual_skill_max",
    #     "height_sim",
    #     "relationship",
    #     "hitter_bouncing_to_ratio",
    #
    # ]
    # numbers = [
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     True,
    #     False,
    #     False,
    #     False,
    #     True,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     True,
    #     False,
    #     False,
    #     False,
    #     True,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #
    # ]
    # per_individuals = [
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     False,
    #
    # ]

    # features = [
    #
    #     "age_sim",
    #     "gender_sim",
    #
    # ]
    # numbers = [
    #     False,
    #     True,
    #
    # ]
    # per_individuals = [
    #     True,
    #     True,
    #
    # ]

    features = [
        "hitter_bouncing_to_partner",

    ]

    numbers = [
        True

    ]

    per_individuals = [
        False
    ]
    efficient_mean_list = []
    inefficient_mean_list = []
    efficient_hdi_list = []
    inefficient_hdi_list = []
    effect_size_mean_list = []
    effect_size_hdi_list = []
    features_list = []

    truncated_features = [
        # "hitter_bouncing_to_partner",
        # "receiver_p3_fx_duration",
        # "hitter_fx_duration",
        # "me_whole_sim",
        # "height_sim",
        # "individual_skill_sim",
        # "ecg_lfhf_sim"
        "test"]
    for f, bin, pi in zip(features, numbers, per_individuals):
        print(f)
        analyzed_features = f
        clean_df = df.dropna(subset=[analyzed_features])

           # mean and std ori
        mean_ori = np.nanmean(clean_df[analyzed_features].values.reshape(-1, 1))
        std_ori = np.nanstd(clean_df[analyzed_features].values.reshape(-1, 1))

        # standarized features (do not do it for numbers)
        if not (bin):
            scaler = StandardScaler()
            average_scaled = scaler.fit_transform(clean_df[analyzed_features].values.reshape(-1, 1))
            clean_df[analyzed_features] = average_scaled.flatten()

        subjects_idx = np.unique(np.concatenate([clean_df["receiver"].values, clean_df["hitter"].values]))
        sessions_idx = np.unique(clean_df["session"])
        if pi:
            # for skills and personal and ecg
            ineff_hitter_idx = np.searchsorted(subjects_idx,
                                               clean_df.loc[clean_df["group"] == "inefficient"].groupby("session")[
                                                   "hitter"].last())

            eff_hitter_idx = np.searchsorted(subjects_idx,
                                             clean_df.loc[clean_df["group"] == "efficient"].groupby("session")[
                                                 "hitter"].last())

            ineff_receiver_idx = np.searchsorted(subjects_idx,
                                                 clean_df.loc[clean_df["group"] == "inefficient"].groupby("session")[
                                                     "receiver"].last())

            eff_receiver_idx = np.searchsorted(subjects_idx,
                                               clean_df.loc[clean_df["group"] == "efficient"].groupby("session")[
                                                   "receiver"].last())

            # factorize session
            ineff_session_idx = np.searchsorted(sessions_idx,
                                                clean_df.loc[clean_df["group"] == "inefficient"].groupby("session")[
                                                    "session"].last())
            eff_session_idx = np.searchsorted(sessions_idx,
                                              clean_df.loc[clean_df["group"] == "efficient"].groupby("session")[
                                                  "session"].last())

            # # for skills and personal and ecg
            inefficient_obv = clean_df.loc[clean_df["group"] == "inefficient"].groupby("session")[
                analyzed_features].mean().values
            efficient_obv = clean_df.loc[clean_df["group"] == "efficient"].groupby("session")[
                analyzed_features].mean().values
        else:

            # factorize hitter and receiver
            ineff_hitter_idx = np.searchsorted(subjects_idx,
                                               clean_df.loc[clean_df["group"] == "inefficient"]["hitter"].values)
            eff_hitter_idx = np.searchsorted(subjects_idx,
                                             clean_df.loc[clean_df["group"] == "efficient"]["hitter"].values)
            ineff_receiver_idx = np.searchsorted(subjects_idx,
                                                 clean_df.loc[clean_df["group"] == "inefficient"]["receiver"].values)

            eff_receiver_idx = np.searchsorted(subjects_idx,
                                               clean_df.loc[clean_df["group"] == "efficient"]["receiver"].values)

            # factorize session
            ineff_session_idx = np.searchsorted(sessions_idx,
                                                clean_df.loc[clean_df["group"] == "inefficient"]["session"].values)
            eff_session_idx = np.searchsorted(sessions_idx,
                                              clean_df.loc[clean_df["group"] == "efficient"]["session"].values)

            # for others predictor
            inefficient_obv = clean_df.loc[clean_df["group"] == "inefficient"][analyzed_features].values
            efficient_obv = clean_df.loc[clean_df["group"] == "efficient"][analyzed_features].values

        mu_m = clean_df[analyzed_features].mean()
        y_min = np.min(clean_df[analyzed_features])

        print(np.average(inefficient_obv))
        print(np.average(efficient_obv))

        # az.plot_dist(clean_df[analyzed_features] + y_min)
        #
        # plt.show()

        coords = {
            # subjects id
            "subject_idx": subjects_idx,
            "session_idx": sessions_idx,
            "gender_idx": range(3),
            "components": range(2)}
        model = BayesianTTestModel(coords,
                                   [inefficient_obv,  efficient_obv],
                                   [ineff_session_idx, eff_session_idx],
                                   [ineff_hitter_idx, ineff_receiver_idx, eff_hitter_idx, eff_receiver_idx],
                                   analyzed_features,
                                   truncated_features,
                                   mu_m, y_min, bin, pi)
        # with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement
        #
        #     if n:
        #         if f == "gender_sim":
        #             # id
        #             subjects_intercept = pm.Normal("subjects_intercept", 0, 0.5, dims=("subject_idx", "gender_idx"))
        #             sessions_intercept = pm.Normal("sessions_intercept", 0, 0.5, dims=("session_idx", "gender_idx"))
        #
        #             inefficient_mean = pm.Normal('inefficient_mean', 0, sigma=1, dims="gender_idx")
        #             efficient_mean = pm.Normal('efficient_mean', 0, sigma=1, dims="gender_idx")
        #             inefficient_std = inefficient_mean
        #             efficient_std = efficient_mean
        #
        #             inefficient = pm.Categorical("inefficient",
        #                                          logit_p=inefficient_mean
        #                                             + sessions_intercept[ineff_session_idx] * (
        #                                                     subjects_intercept[ineff_hitter_idx] +
        #                                                     subjects_intercept[ineff_receiver_idx]) / 2
        #
        #                                          ,
        #                                          observed=inefficient_obv)
        #             efficient = pm.Categorical("efficient",
        #                                        logit_p=efficient_mean
        #                                           + sessions_intercept[eff_session_idx] + (
        #                                                   subjects_intercept[eff_hitter_idx] +
        #                                                   subjects_intercept[eff_receiver_idx]) / 2
        #
        #                                        ,
        #                                        observed=efficient_obv)
        #         else:
        #             # number
        #             print("I am number")
        #             # id
        #             subjects_intercept = pm.HalfNormal("subjects_intercept", 0.5, dims="subject_idx")
        #             sessions_intercept = pm.HalfNormal("sessions_intercept", 0.5, dims="session_idx")
        #
        #             inefficient_mean = pm.HalfNormal('inefficient_mean', sigma=1)
        #             efficient_mean = pm.HalfNormal('efficient_mean', sigma=1)
        #             inefficient_std = inefficient_mean
        #             efficient_std = efficient_mean
        #
        #             inefficient = pm.Poisson("inefficient",
        #                                      mu=inefficient_mean
        #                                         + sessions_intercept[ineff_session_idx] * (
        #                                                 subjects_intercept[ineff_hitter_idx] +
        #                                                 subjects_intercept[ineff_receiver_idx]) / 2
        #
        #                                      ,
        #                                      observed=inefficient_obv)
        #             efficient = pm.Poisson("efficient",
        #                                    mu=efficient_mean
        #                                       + sessions_intercept[eff_session_idx] + (
        #                                               subjects_intercept[eff_hitter_idx] +
        #                                               subjects_intercept[eff_receiver_idx]) / 2
        #
        #                                    ,
        #                                    observed=efficient_obv)
        #
        #     else:
        #
        #         # continous
        #         # centered for subjects
        #         # 0.5 for wide, 0.1 for narrow. Narrow produces better results
        #         subjects_intercept = pm.Normal("subjects_intercept", mu=0, sigma=0.5, dims="subject_idx")
        #         # centered for sessions
        #         sessions_intercept = pm.Normal("sessions_intercept", mu=0, sigma=0.5, dims="session_idx")
        #         inefficient_std = pm.HalfCauchy("inefficient_std", 1.0)
        #         efficient_std = pm.HalfCauchy("efficient_std", 1.0)
        #         inefficient_mean = pm.Normal('inefficient_mean', mu=mu_m, sigma=1)
        #         efficient_mean = pm.Normal('efficient_mean', mu=mu_m, sigma=1)
        #
        #         nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        #         nu = pm.Deterministic("nu", nu_minus_one + 1)
        #
        #         a = pm.Gamma("a", alpha=2, beta=0.5)  # controls tails, higher = lighter tail
        #         b = pm.Gamma("b", alpha=2, beta=0.5)  # controls skewness
        #
        #         lambda_1 = efficient_std ** -2
        #         lambda_2 = inefficient_std ** -2
        #
        #         if pi:
        #             inefficient_latent = pm.Deterministic("iefficient_latent", inefficient_mean
        #                                                   + (
        #                                                           subjects_intercept[ineff_hitter_idx] +
        #                                                           subjects_intercept[ineff_receiver_idx]) / 2)
        #
        #             efficient_latent = pm.Deterministic("efficient_latent", efficient_mean
        #                                                 + (
        #                                                         subjects_intercept[eff_hitter_idx] +
        #                                                         subjects_intercept[eff_receiver_idx]) / 2)
        #         else:
        #             inefficient_latent = pm.Deterministic("iefficient_latent", inefficient_mean
        #                                                   + sessions_intercept[ineff_session_idx] + (
        #                                                           subjects_intercept[ineff_hitter_idx] +
        #                                                           subjects_intercept[ineff_receiver_idx]) / 2)
        #
        #             efficient_latent = pm.Deterministic("efficient_latent", efficient_mean
        #                                                 + sessions_intercept[eff_session_idx] + (
        #                                                         subjects_intercept[eff_hitter_idx] +
        #                                                         subjects_intercept[eff_receiver_idx]) / 2)
        #
        #         if analyzed_features in truncated_features:
        #             inefficient = pm.TruncatedNormal("inefficient",
        #                                              lower=y_min,
        #                                              mu=inefficient_latent,
        #                                              sigma=inefficient_std,
        #
        #                                              observed=inefficient_obv)
        #             efficient = pm.TruncatedNormal("efficient",
        #                                            lower=y_min,
        #                                            mu=efficient_latent,
        #                                            sigma=efficient_std, observed=efficient_obv)
        #
        #         else:
        #
        #             # inefficient = pm.StudentT("inefficient", nu=nu,
        #             #                           mu=inefficient_latent, lam=lambda_2,
        #             #                           observed=inefficient_obv)
        #             # efficient = pm.StudentT("efficient", nu=nu, mu=efficient_latent,
        #             #                         lam=lambda_1, observed=efficient_obv)
        #
        #             inefficient = pm.SkewStudentT("inefficient",
        #                                           a=a,
        #                                           b=b,
        #                                           mu=inefficient_latent,
        #                                           sigma=inefficient_std,
        #
        #                                           observed=inefficient_obv)
        #             efficient = pm.SkewStudentT("efficient",
        #                                         a=a,
        #                                         b=b,
        #                                         mu=efficient_latent,
        #                                         sigma=efficient_std, observed=efficient_obv)
        #
        #     # means difference and others
        #
        #     diff_of_means = pm.Deterministic("difference_of_means", efficient_mean - inefficient_mean)
        #     diff_of_stds = pm.Deterministic("difference_of_stds", efficient_std - inefficient_std)
        #
        #     if (n) or (f == "gender_sim"):
        #         print("effect_size")
        #         effect_size = pm.Deterministic(
        #             "effect_size", diff_of_means / np.sqrt((inefficient_std + efficient_std) / 2)
        #         )
        #     else:
        #         effect_size = pm.Deterministic(
        #             "effect_size", diff_of_means / np.sqrt((inefficient_std ** 2 + efficient_std ** 2) / 2)
        #         )

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
        plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "PPC_wider\\" + analyzed_features + ".png", format='png')
        plt.close()

        trace_post = az.extract(idata.posterior)
        # print(az.summary(idata))
        # save the model
        with open(DOUBLE_RESULTS_PATH_TTEST + "model_wider\\" + "idata_" + analyzed_features + ".pkl", 'wb') as handle:
            print("write data into: " + "idata_ttest_" + analyzed_features + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del model
        del idata
