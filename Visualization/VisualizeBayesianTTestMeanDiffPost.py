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


# def gedDF():
#     inefficient_group, efficient_group = groupLabeling()
#
#     # inefficient group
#     inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
#                                                     file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
#                                                     include_subjects=inefficient_group, exclude_failure=True,
#                                                     exclude_no_pair=False, hmm_probs=True)
#     inefficient_features = inefficient_reader.getStableUnstableFailureFeatures(group_name="inefficient",
#                                                                                success_failure=True,
#                                                                                mod="skill_personal_perception_action_impact",
#                                                                                with_control=True)
#     inefficient_features["group"] = "inefficient"
#     # efficient group
#     efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
#                                                   file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
#                                                   include_subjects=efficient_group, exclude_failure=True,
#                                                   exclude_no_pair=False, hmm_probs=True)
#     efficient_features = efficient_reader.getStableUnstableFailureFeatures(group_name="efficient", success_failure=True,
#                                                                            mod="skill_personal_perception_action_impact",
#                                                                            with_control=True)
#     efficient_features["group"] = "efficient"
#
#     df = pd.concat([inefficient_features, efficient_features])
#
#     return df


feature_type = "double_q"
if feature_type == "hitter_eye":

    features = [
        "hitter_p1_al_onset",
        "hitter_p1_al_prec",
        "hitter_p1_al_mag",
        "hitter_p1_cs",
        "hitter_p2_al_onset",
        "hitter_p2_al_prec",
        "hitter_p2_al_mag",
        "hitter_p2_cs",
        "hitter_fx_onset",
        "hitter_fx_duration",

    ]
elif feature_type == "receiver_eye":
    features = [
    "receiver_p1_al_onset",
    "receiver_p1_al_prec",
    "receiver_p1_al_mag",
    "receiver_p1_cs",
    "receiver_p2_al_onset",
    "receiver_p2_al_prec",
    "receiver_p2_al_mag",
    "receiver_p2_cs",
    "receiver_p3_fx_onset",
    "receiver_p3_fx_duration",

    ]
elif feature_type == "action":

    features = [
    "receiver_fixation_racket_latency",
    "receiver_start_fs",
    "receiver_distance_eye_hand",

        ]
elif feature_type == "impact":

    features = [
    "receiver_im_ball_wrist",
    "receiver_im_racket_ball_wrist",
    "receiver_im_racket_ball_angle",
    "receiver_im_ball_updown",
        ]

elif feature_type == "personal":
    features = [
    "individual_skill",
    "individual_skill_sim",
    "height_sim",
    "age_sim",
    "relationship",
        ]
elif feature_type == "ecg":
    features = [
    "rr_sim",

        ]
elif feature_type == "double_q":
    features = [

        "double_self_report_score",
        "double_team_score",
        "double_facilitating_skill",
        "double_partner_skill",
        "subject_indv_myskill"
    ]
else:
    features = [
        "receiver_start_fs",
        "hitter_p1_cs",
        "receiver_im_ball_wrist",
        "receiver_im_racket_ball_wrist",
        "receiver_fixation_racket_latency",
        "hitter_p2_al_mag",
        "receiver_p1_cs",
        "receiver_p2_al_mag",
        "receiver_p3_fx_duration",
        "hitter_p1_al_onset",
        "hitter_p1_al_prec",
        "hitter_p2_al_onset",
        "receiver_im_ball_updown",
        "receiver_p1_al_onset",
        "hitter_p2_al_prec",
        "receiver_p3_fx_onset",
        "receiver_p2_al_onset",
        "receiver_distance_eye_hand",
        "receiver_im_racket_ball_angle",
        "receiver_p1_al_prec",
        "receiver_p2_al_prec",
        "age_sim",
        "hitter_fx_duration",
        "hitter_p2_cs",
        "hitter_fx_onset",
        "relationship",
        "hitter_p1_al_mag",
        "individual_skill_sim",
        "height_sim",
        "receiver_p2_cs",
        "individual_skill",
        "receiver_p1_al_mag",



    ]
models = []

for analyzed_features in features:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)
    models.append(idata)

height = len(features) * (3.5 / 10)
# height = len(features) * (4.5 / 10)
axis = az.plot_forest(
    models,
    model_names=features,
    var_names=["difference_of_means1", "difference_of_means2", "difference_of_means3", "difference_of_means4", "difference_of_means5", "difference_of_means6"],
    # var_names=["difference_of_means1",  "difference_of_means4"],
    figsize=(3.3,height),
    combined=True,
    markersize=4,
    hdi_prob=0.95,
    legend=False,
    linewidth=1.5,
    # transform=lambda x: np.abs(x),
    colors="#6d88c3",

)[0]
ymin, ymax = axis.get_ylim()
# axis.set_xlim(-1, 1)
axis.vlines([0.], ymin, ymax, ls='--', colors="#000000")
# plt.minorticks_on()
plt.show()

# plt.savefig(os.path.join(DOUBLE_RESULTS_PATH_TTEST, "MeanDifference", feature_type+".pdf"))


