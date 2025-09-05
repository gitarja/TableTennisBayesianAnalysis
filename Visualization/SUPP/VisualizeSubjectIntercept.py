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


feature_type = "t-test"
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

models = []

for analyzed_features in features:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
    # with open(DOUBLE_RESULTS_PATH_TTEST + "\\model\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)
    az.plot_forest(idata, var_names=["subjects_intercept"], combined=True)
    plt.show()
    plt.close()




