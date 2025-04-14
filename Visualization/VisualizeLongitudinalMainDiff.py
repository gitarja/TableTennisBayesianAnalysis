import os.path

import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, features_explanation
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_RESULTS_PATH_LONGITUDINAL, \
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



feature_type = "hitter_eye"
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

models = []
n = 5
for analyzed_features in features:

    with open(DOUBLE_RESULTS_PATH_LONGITUDINAL + "idata_" + analyzed_features + "_" + str(n) + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)
    models.append(idata)

height = len(features) * (4.5 / 10)

# height = len(features) * (6 / 10)
axis = az.plot_forest(
    models,
    model_names=features,
    var_names=["global_diff_of_means"],
    figsize=(4.5,height),
    combined=True,
    markersize=4,
    hdi_prob=0.95,
    legend=False,
    linewidth=1.5,
    colors="#6d88c3",

)[0]
ymin, ymax = axis.get_ylim()
axis.set_xlim(-1, 1)
axis.vlines([0.], ymin, ymax, ls='--', colors="#000000")
# plt.minorticks_on()
# plt.show()

plt.savefig(os.path.join(DOUBLE_RESULTS_PATH_LONGITUDINAL, "MeanDifference", feature_type+".pdf"))


