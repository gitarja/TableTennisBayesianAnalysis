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


feature_type = "t-test-post"
if feature_type == "t-test":
    features = [
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
    height = len(features) * (4.5 / 10)
if feature_type == "t-test-mean":
    features = [
        "var_start_fs",


    ]
if feature_type == "t-test-std":
    features = [
        "var_im_ball_updown",
        "var_im_racket_ball_wrist",


    ]
if feature_type == "t-test-shap-important":
    features = [
        "p1_al_prec_sim",
        "p1_cs_mean",
        "im_racket_ball_wrist_mean",
        "im_racket_ball_angle_sim",
        "p1_al_prec_mean",
        "p2_cs_mean",
        "me_whole_mean",
        "p1_al_onset_mean",
        "p2_cs_sim",
        "im_racket_ball_angle_mean",
        "ec_start_fs_sim",
        "ecg_lfhf_sim",
        "p2_al_onset_mean",
        "im_ball_updown_sim",
        "me_whole_sim"

    ]

if feature_type == "t-test-insignificant":
    features = [
        "hitter_p1_al_onset",
        "age_sim",
        "receiver_p3_fx_duration",
        "hitter_p2_al_mag",
        "ecg_lfhf_sim",
        "ecg_lfhf_mean",
        "receiver_p2_al_mag",
        "me_whole_sim",
        "me_whole_mean",
        "individual_skill_sim",
        "receiver_p1_al_onset",
        "hitter_fx_duration",
        "receiver_im_racket_ball_angle",
        "hitter_p2_al_prec",
        "relationship",
        "height_sim",
        "receiver_p1_al_prec",
        "individual_skill",


    ]
    height = len(features) * (3.5 / 10)
if feature_type == "t-test-post":
    features = [
        "double_self_report_score",
        "double_team_score",
        "double_facilitating_skill",
        "double_partner_skill",
        "subject_indv_myskill",
    ]

if feature_type == "t-test-insignificant-gender":
    features = ["gender_sim"]
    height = 3 * (3.5 / 10)

models = []

for analyzed_features in features:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_supp\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
    # with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_final\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)
    # idata.posterior['effect_size'] = idata.posterior['effect_size'].dropna(dim="draw", how="any")

    models.append(idata)
height = len(features) * (6.5 / 10)
# height = len(features) * (3.5 / 10)

axis = az.plot_forest(
    models,
    model_names=features,
    var_names=["difference_of_means"],
    figsize=(3.3, height),
    combined=True,
    markersize=4,
    hdi_prob=0.95,
    legend=False,
    linewidth=1.5,
    # transform=lambda x: 0 if x == np.nan else x,
    colors="#6d88c3",

)[0]
ymin, ymax = axis.get_ylim()
# axis.set_xticks([-0.5, 0, 0.5])
# axis.set_xlim(-1.5, 1.5)

#axis.set_xticks([-1.5, 0, 1.5])
#axis.set_xlim(-2.5, 2.5)
axis.vlines([0.], ymin, ymax, ls='--', colors="#000000")
# plt.show()

results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\all_lower_upper\\"
plt.savefig(os.path.join(results_path, "MeanDifference", feature_type + ".pdf"))
