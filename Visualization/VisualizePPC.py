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


features = [
    # ECG similarity metrics
    "ecg_hf_sim", "ecg_lfhf_sim", "ecg_rmsdd_sim",
    # ECG raw metrics (means)
    "ecg_hf_mean", "ecg_lfhf_mean", "ecg_rmsdd_mean",
    # Hitter-level anticipatory metrics (onset, precision, magnitude, cross-correlation)
    "hitter_p1_al_onset", "hitter_p1_al_prec", "hitter_p1_al_mag", "hitter_p1_cs",
    "hitter_p2_al_onset", "hitter_p2_al_prec", "hitter_p2_al_mag", "hitter_p2_cs",
    "hitter_fx_onset", "hitter_fx_duration",
    # Receiver-level anticipatory metrics
    "receiver_p1_al_onset", "receiver_p1_al_prec", "receiver_p1_al_mag", "receiver_p1_cs",
    "receiver_p2_al_onset", "receiver_p2_al_prec", "receiver_p2_al_mag", "receiver_p2_cs",
    # Receiver fixation and timing features
    "receiver_p3_fx_onset", "receiver_p3_fx_duration",
    "receiver_fixation_racket_latency", "receiver_start_fs",
    # Spatial relationship metrics
    "receiver_distance_eye_hand", "receiver_im_ball_wrist",
    "receiver_im_racket_ball_wrist", "receiver_im_racket_ball_angle",
    "receiver_im_ball_updown",
    # Hitter bouncing behavior measures
    "hitter_bouncing_to_partner", "hitter_bouncing_to_self", "hitter_at_and_after_hit",
    # "Me" (subject self-perception) similarity and raw metrics
    "me_foot_sim", "me_shoulder_arm_sim", "me_whole_sim",
    "me_foot_mean", "me_shoulder_arm_mean", "me_whole_mean",
    # Individual skill estimates and related metrics
    "individual_skill", "individual_skill_sim", "individual_skill_max",
    # Other contextual features
    "height_sim", "relationship", "hitter_bouncing_to_ratio",
]

for analyzed_features in features:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_final\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:

        idata = pickle.load(handle)
    hierarchical_loo = az.plot_ppc(idata)
    plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "PPC_final\\" + analyzed_features + ".png", format='png')
    plt.close()