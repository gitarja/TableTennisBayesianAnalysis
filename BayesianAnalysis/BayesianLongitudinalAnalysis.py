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

# ANALYZED_FEATURES = [
#
#     # "receiver_start_fs",
#     # "hitter_p1_al_mag",
#     # "receiver_im_racket_dir",
#     "receiver_fixation_racket_latency",
#     "hitter_p2_al_prec",
#     "hitter_p2_al_mag",
#     "receiver_p2_al_mag",
#     "hitter_at_and_after_hit",
#     "hitter_p1_cs",
#     "hitter_p2_al_onset",
#     "hand_movement_sim",
#     "receiver_p1_al_onset",
#     "hitter_p1_al_prec",
#     "receiver_p1_al_mag",
#     "receiver_p2_al_prec",
#     "receiver_p2_al_onset",
#     "hitter_fx_onset",
#     "receiver_distance_eye_hand",
#     "hitter_p1_al_onset",
#     "receiver_p1_al_prec",
#     "hitter_fx_duration",
#     "receiver_p3_fx_onset",
#     "receiver_p1_cs",
#     "receiver_p3_fx_duration"
#
# ]
# HITTER_BOOL = [
#     #
#     # False,
#     # True,
#     # False,
#     False,
#     True,
#     True,
#     False,
#     True,
#     True,
#     True,
#     False,
#     False,
#     True,
#     False,
#     False,
#     False,
#     True,
#     False,
#     True,
#     False,
#     True,
#     False,
#     False,
#     False
# ]
# # Binominal
# BINOMINAL = [
#
#     # False,
#     # False,
#     # False,
#     False,
#     False,
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
#     True,
#     False
#
# ]


ANALYZED_FEATURES = [

    "receiver_im_ball_updown",
    # "receiver_p1_al_prec",


]
HITTER_BOOL = [
    False,
    # False,
]
# Binominal
BINOMINAL = [
    False,
    # False,
]
if __name__ == '__main__':
    n = 5

    lower_group, upper_group = groupLabeling()

    # inefficient group
    lower_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=lower_group, exclude_failure=True,
                                                    exclude_no_pair=False, hmm_probs=True)
    lower_features = lower_reader.getStableUnstableFailureFeatures(group_name="lower",
                                                                               success_failure=True,
                                                                               mod="skill_personal_perception_action_impact",
                                                                               with_control=True, timepoint=True)
    lower_features["group"] = "lower"
    # efficient group
    upper_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=upper_group, exclude_failure=True,
                                                  exclude_no_pair=False, hmm_probs=True)
    upper_features = upper_reader.getStableUnstableFailureFeatures(group_name="higher", success_failure=True,
                                                                           mod="skill_personal_perception_action_impact",
                                                                           with_control=True, timepoint=True)
    upper_features["group"] = "higher"

    df = pd.concat([lower_features, upper_features])

    # print(df)
    df.loc[:, "lower"] = df.group == "lower"
    df.loc[:, "higher"] = df.group == "higher"

    for feature, hitter, bin in zip(ANALYZED_FEATURES, HITTER_BOOL, BINOMINAL):
        clean_df = df.dropna(subset=[feature])

        # scaler = StandardScaler()
        # average_scaled = scaler.fit_transform(clean_df[feature].values.reshape(-1, 1))
        # clean_df[feature] = average_scaled.flatten()
        az.plot_dist(clean_df[feature])
        plt.show()

        if hitter:
            subjects = clean_df["hitter"]
            clean_df.loc[:, "th_segments"] = clean_df["hitter_timepoint"] / 100
        else:
            subjects = clean_df["receiver"]
            clean_df.loc[:, "th_segments"] =  clean_df["receiver_timepoint"]/ 100

        subjects_idx, subjects_unique = pd.factorize(subjects)

        coords = {"subject_idx": subjects_unique, "obs": range(len(clean_df[feature])),
                  "group": ["lower", "higher"]}

        model = CenteredModel(coords, clean_df, subjects_idx, feature, n, bin, hitter=hitter)

        with model:
            print(model.debug())
            # pm.model_to_graphviz(model).view()
            idata = pm.sample_prior_predictive()

            idata.extend(
                pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                          draws=2000,
                          chains=N_CHAINS, tune=3000, cores=N_CORE, compile_kwargs=dict(mode="NUMBA"))
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


    del model

