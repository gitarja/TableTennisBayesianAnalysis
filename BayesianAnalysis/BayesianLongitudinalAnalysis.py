import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, features_explanation
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_LONGITUDINAL
from LongitudinalModels import CenteredModel
from sklearn.preprocessing import StandardScaler
import arviz as az
import pickle


ANALYZED_FEATURES = [
    "hitter_p2_cs",
    "receiver_p2_cs",

    "hitter_p1_cs",
    "receiver_p1_cs",
    "receiver_start_fs",

    "receiver_im_racket_ball_wrist",
    "hitter_bouncing_to_partner",
    "receiver_distance_eye_hand",
    "hitter_p2_al_onset",
    "receiver_p2_al_prec",
    "receiver_im_ball_updown",
    "receiver_p1_al_mag",
    "hitter_p1_al_prec",
    "hitter_p1_al_mag",
    "receiver_p2_al_onset",
    "hitter_at_and_after_hit",

]
HITTER_BOOL = [
    True,
    False,

    True,
    False,
    False,

    False,
    True,
    False,
    True,
    False,
    False,
    False,
    True,
    True,
    False,
    True,
]
# Binominal
BINOMINAL = [
    True,
    True,

    True,
    True,
    False,

    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,

]

# ANALYZED_FEATURES = [
#     "hitter_bouncing_to_partner",
#
#
# ]
# HITTER_BOOL = [
#
#     False,
#
# ]
# BINOMINAL = [
#
#     False,
# ]
if __name__ == '__main__':
    n = 10

    lower_group, upper_group = groupLabeling()

    # inefficient group
    lower_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                              include_subjects=lower_group, exclude_failure=True,
                                              exclude_no_pair=False, hmm_probs=True)
    lower_features = lower_reader.getStableUnstableFailureFeatures(group_name="inefficient",
                                                                   success_failure=True,
                                                                   mod="skill_personal_perception_action_impact_ecg_me_other",
                                                                   with_control=True, timepoint=True, min_group_n=n)

    # efficient group
    upper_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                              include_subjects=upper_group, exclude_failure=True,
                                              exclude_no_pair=False, hmm_probs=True)
    upper_features = upper_reader.getStableUnstableFailureFeatures(group_name="efficient", success_failure=True,
                                                                   mod="skill_personal_perception_action_impact_ecg_me_other",
                                                          with_control=True, timepoint=True, min_group_n=n)
    lower_features["group"] = "inefficient"
    upper_features["group"] = "efficient"

    df = pd.concat([lower_features, upper_features])

    # print(df)
    df.loc[:, "inefficient"] = df.group == "inefficient"
    df.loc[:, "efficient"] = df.group == "efficient"

    for feature, hitter, bin in zip(ANALYZED_FEATURES, HITTER_BOOL, BINOMINAL):
        clean_df = df.dropna(subset=[feature])

        if bin != True:
            scaler = StandardScaler()
            average_scaled = scaler.fit_transform(clean_df[feature].values.reshape(-1, 1))
            clean_df[feature] = average_scaled.flatten()
        # az.plot_dist(clean_df[feature])
        # plt.show()

        if hitter:
            clean_df.loc[:, "th_segments"] = clean_df["hitter_timepoint"]/100
        else:

            clean_df.loc[:, "th_segments"] = clean_df["receiver_timepoint"]/100

        hitters = clean_df["hitter"]
        receivers = clean_df["receiver"]
        subjects_idx = np.unique(np.concatenate([clean_df["receiver"].values, clean_df["hitter"].values]))#
        sessions_idx = np.unique(clean_df["session"])

        hitters_idx = np.searchsorted(subjects_idx, hitters)

        receivers_idx = np.searchsorted(subjects_idx, receivers)

        session_idx = np.searchsorted(sessions_idx, clean_df["session"])

        coords = {"subject_idx": subjects_idx,
                  "session_idx": sessions_idx,
                  "obs": range(len(clean_df[feature])),
                  "group": ["inefficient", "efficient"]}

        model = CenteredModel(coords, clean_df, hitters_idx, receivers_idx, session_idx, feature, n, bin, hitter=hitter)

        with model:
            print(model.debug())
            # pm.model_to_graphviz(model).view()
            idata = pm.sample_prior_predictive()

            idata.extend(
                pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                          draws=N_SAMPLES,
                          chains=N_CHAINS, tune=N_TUNE, cores=N_CORE, compile_kwargs=dict(mode="NUMBA"))
            )
            idata.extend(pm.sample_posterior_predictive(idata))

        # save the model
        file_name = "idata_"
        image_name = "r_hat_"

        with open(DOUBLE_RESULTS_PATH_LONGITUDINAL + file_name + feature + "_" + str(n) + ".pkl", 'wb') as handle:
            print("write data into: " + file_name + feature + "_" + str(n) + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # plot rhat
        nc_rhat = az.rhat(idata)
        ax = (nc_rhat.max()
              .to_array()
              .to_series()
              .plot(kind="barh"))
        plt.savefig(DOUBLE_RESULTS_PATH_LONGITUDINAL + image_name + feature + "_" + str(n) + ".png")
        plt.close()

        hierarchical_loo = az.plot_ppc(idata)
        plt.savefig(DOUBLE_RESULTS_PATH_LONGITUDINAL  +  image_name + feature + "_ppc.png", format='png')
        plt.close()

    del model
