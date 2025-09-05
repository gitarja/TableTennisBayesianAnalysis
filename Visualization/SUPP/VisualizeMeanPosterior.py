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


feature_type = "t-test"
if feature_type == "t-test":
    features = [
        "receiver_p1_al_prec",
        # "hitter_p2_cs",
        # "receiver_p2_cs",
        # "hitter_p1_cs",
        # "receiver_p1_cs",
        # "receiver_start_fs",
        # "receiver_im_racket_ball_wrist",
        # "hitter_bouncing_to_partner",
        # "receiver_distance_eye_hand",
        # "hitter_p2_al_onset",
        # "receiver_p2_al_prec",
        # "receiver_im_ball_updown",
        # "receiver_p1_al_mag",
        # "hitter_p1_al_prec",
        # "hitter_p1_al_mag",
        # "receiver_p2_al_onset",
        # "hitter_at_and_after_hit",

    ]
    height = len(features) * (5 / 10)
if feature_type == "t-test-mean":
    features = [
        "var_start_fs",
        "var_receiver_p1_al_onset",
        "var_spatial_use",
        "var_hitter_p1_cs",
        "var_hitter_p2_cs",

    ]
if feature_type == "t-test-std":
    features = [
        "var_im_ball_updown",
        "var_im_racket_ball_wrist",
        "var_hitter_p2_al_onset",
        "var_hitter_p1_cs",
        "var_receiver_p1_al_onset",

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
means_effectient_list = []
means_ineffecient_list = []
for analyzed_features in features:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_final\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)

    trace_post = az.extract(idata.posterior)

    mean_effecient = trace_post['efficient_mean'].data.flatten()
    mean_ineffecient = trace_post['inefficient_mean'].data.flatten()

    df_efficient = pd.DataFrame({"posterior":mean_effecient, "group":"efficient" })
    df_inefficient = pd.DataFrame({"posterior": mean_ineffecient, "group": "inefficient"})

    fig, ax = plt.subplots(figsize=(8, 4))  # width=10, height=6 inches
    sns.kdeplot(data=df_efficient, x="posterior",  color="#68a880", fill=True, alpha=.3, linewidth=1,)
    sns.kdeplot(data=df_inefficient, x="posterior", color="#b5202d", fill=True, alpha=.3, linewidth=1,)
    plt.legend().remove()
    sns.despine()

    ax.set_yticks([])

    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    x_ticks = np.arange(xmin, xmax, (xmax - xmin) / 4)
    ax.set_xticks(x_ticks)

    # if "_cs" in analyzed_features:
    #     plt.xlim(0.0, 1.0)
    # else:
    #     plt.xlim(-0.5, 0.5)
    # plt.show()
    results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\all_lower_upper\\"
    plt.savefig(os.path.join(results_path, "MeanGroups", analyzed_features + "_means.pdf"))
    plt.close()


