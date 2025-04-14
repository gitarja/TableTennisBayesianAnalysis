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
    "receiver_fixation_racket_latency",
    "receiver_start_fs",
    "receiver_distance_eye_hand",
    "receiver_im_ball_wrist",
    "receiver_im_racket_ball_wrist",
    "receiver_im_racket_ball_angle",
    "receiver_im_ball_updown",
    "individual_skill",
    "individual_skill_sim",
    "height_sim",
    "age_sim",
    "relationship",

]

mean_abs_ef = []
for analyzed_features in features:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)

        trace_post = az.extract(idata.posterior)

        mean_effect = np.abs(np.median(trace_post['effect_size'].data.flatten())) / np.nanstd(trace_post['effect_size'].data.flatten())

        mean_abs_ef.append(mean_effect)


df = pd.DataFrame({"features": features, "mean_abs_ef": mean_abs_ef})

df.to_csv(os.path.join(DOUBLE_RESULTS_PATH_TTEST, "effect_size.csv"))

