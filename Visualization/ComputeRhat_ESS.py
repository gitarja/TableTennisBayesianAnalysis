import os.path

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
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

# sns.set_style("white")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20


# plt.rcParams.update({'xtick.labelsize': 25, 'ytick.labelsize': 25})


def gedDF():
    inefficient_group, efficient_group = groupLabeling()

    # inefficient group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=inefficient_group, exclude_failure=True,
                                                    exclude_no_pair=False, hmm_probs=True)
    inefficient_features = inefficient_reader.getStableUnstableFailureFeatures(group_name="inefficient",
                                                                               success_failure=True,
                                                                               mod="skill_personal_perception_action_impact",
                                                                               with_control=True)
    inefficient_features["group"] = "inefficient"
    # efficient group
    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=efficient_group, exclude_failure=True,
                                                  exclude_no_pair=False, hmm_probs=True)
    efficient_features = efficient_reader.getStableUnstableFailureFeatures(group_name="efficient", success_failure=True,
                                                                           mod="skill_personal_perception_action_impact",
                                                                           with_control=True)
    efficient_features["group"] = "efficient"

    df = pd.concat([inefficient_features, efficient_features])

    return df


feature_type = "t-test-mean"
if feature_type == "t-test":
    features = [
        "receiver_im_racket_ball_wrist",
        "hitter_bouncing_to_partner",
        "hitter_p1_al_onset",
        "hitter_p1_cs",
        "receiver_start_fs",
        "hitter_p2_al_mag",
        "receiver_p2_al_onset",
        "receiver_p1_cs",
        "receiver_distance_eye_hand",
        "hitter_p2_al_onset",
        "receiver_p3_fx_duration",
        "hitter_p2_al_prec",
        "hitter_at_and_after_hit",
        "receiver_p1_al_prec",
        "hitter_p2_cs",
        "receiver_p2_al_mag",
        "hitter_p1_al_prec",
        "receiver_p2_cs",
        "receiver_im_ball_updown",

    ]
    height = len(features) * (5 / 10)
if feature_type == "t-test-mean":
    features = [
        "var_hitter_p1_cs",

        "var_hitter_p1_al_onset",
        "var_hitter_p1_al_prec",
        "var_hitter_p1_al_mag",
        "var_receiver_p1_al_onset",
        "var_receiver_p1_al_prec",
        "var_receiver_p1_al_mag",
        "var_receiver_p1_cs",

        "var_start_fs",
        "var_bounce_point",
        "var_spatial_use",

        "var_hitter_p2_al_onset",
        "var_hitter_p2_al_prec",
        "var_hitter_p2_al_mag",
        "var_hitter_p2_cs",

        "var_hitter_p3_fx_onset",
        "var_hitter_p3_fx_duration",

        "var_receiver_p2_al_onset",
        "var_receiver_p2_al_prec",
        "var_receiver_p2_al_mag",
        "var_receiver_p2_cs",

        "var_receiver_p3_fx_onset",
        "var_receiver_p3_fx_duration",

        "var_fixation_racket_latency",
        "var_distance_eye_han",
        "var_im_ball_wrist",
        "var_im_racket_ball_wrist",
        "var_im_racket_ball_angle",
        "var_im_ball_updown",

        "lfhf_sim"
    ]
if feature_type == "t-test-std":
    features = [
        "var_hitter_p1_cs",

        "var_hitter_p1_al_onset",
        "var_hitter_p1_al_prec",
        "var_hitter_p1_al_mag",
        "var_receiver_p1_al_onset",
        "var_receiver_p1_al_prec",
        "var_receiver_p1_al_mag",
        "var_receiver_p1_cs",

        "var_start_fs",
        "var_bounce_point",
        "var_spatial_use",

        "var_hitter_p2_al_onset",
        "var_hitter_p2_al_prec",
        "var_hitter_p2_al_mag",
        "var_hitter_p2_cs",

        "var_hitter_p3_fx_onset",
        "var_hitter_p3_fx_duration",

        "var_receiver_p2_al_onset",
        "var_receiver_p2_al_prec",
        "var_receiver_p2_al_mag",
        "var_receiver_p2_cs",

        "var_receiver_p3_fx_onset",
        "var_receiver_p3_fx_duration",

        "var_fixation_racket_latency",
        "var_distance_eye_han",
        "var_im_ball_wrist",
        "var_im_racket_ball_wrist",
        "var_im_racket_ball_angle",
        "var_im_ball_updown",

        "lfhf_sim"

    ]
if feature_type == "t-test-shap-important":
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
    height = len(features) * (3.95 / 10)

if feature_type == "t-test-insignificant":
    features = [
        "receiver_p1_al_onset",
        "ecg_lfhf_sim",
        "ecg_lfhf_mean",
        "individual_skill_sim",
        "me_whole_sim",
        "me_whole_mean",
        "hitter_fx_duration",
        "receiver_p2_al_prec",
        "hitter_p1_al_mag",
        "relationship",
        "receiver_p1_al_mag",
        "receiver_im_racket_ball_angle",
        "height_sim",
        "individual_skill",

    ]
    height = len(features) * (4.3 / 10)

summary_list = []
for analyzed_features in features:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_supp\\" + "idata_" + analyzed_features + "_mean.pkl", 'rb') as handle:
    # with open(DOUBLE_RESULTS_PATH_TTEST + "\\model\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)

    summary = az.summary(idata, var_names=["efficient_mean", "inefficient_mean", "effect_size"])
    summary["features"] = analyzed_features
    summary_list.append(summary)
df_summary = pd.concat(summary_list)
path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double\\Bayesian-ttest\\"

df_summary.to_csv(os.path.join(path, "convergence_"+feature_type+".csv"))

