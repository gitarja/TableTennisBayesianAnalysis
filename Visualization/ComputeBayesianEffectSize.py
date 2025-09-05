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
    "ecg_lfhf_sim",
    "ecg_lfhf_mean",
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
    "receiver_p3_fx_duration",
    "receiver_start_fs",
    "receiver_distance_eye_hand",
    "receiver_im_racket_ball_wrist",
    "receiver_im_racket_ball_angle",
    "receiver_im_ball_updown",
    "hitter_bouncing_to_partner",
    "hitter_at_and_after_hit",
    "me_whole_sim",
    "me_whole_mean",
    "individual_skill",
    "individual_skill_sim",
    "height_sim",
    "relationship",

    "age_sim",
    "gender_sim",


    # var
    # "var_hitter_p1_cs",
    # "var_hitter_p1_al_onset",
    # "var_hitter_p1_al_prec",
    # "var_hitter_p1_al_mag",
    #
    # "var_receiver_p1_al_onset",
    # "var_receiver_p1_al_prec",
    # "var_receiver_p1_al_mag",
    # "var_receiver_p1_cs",
    #
    # "var_hitter_p2_al_onset",
    # "var_hitter_p2_al_prec",
    # "var_hitter_p2_al_mag",
    # "var_hitter_p2_cs",
    #
    # "var_hitter_p3_fx_duration",
    #
    # "var_receiver_p2_al_onset",
    # "var_receiver_p2_al_prec",
    # "var_receiver_p2_al_mag",
    # "var_receiver_p2_cs",
    #
    # "var_receiver_p3_fx_duration",
    #
    # "var_start_fs",
    # "var_bounce_point",
    # "var_spatial_use",
    #
    # "var_distance_eye_han",
    # "var_im_racket_ball_wrist",
    # "var_im_racket_ball_angle",
    # "var_im_ball_updown",
    #
    # "lfhf_sim"

    # # study 1
    # "subject_skill",
    # "subject_skill_sim",
    # "p1_al_onset_sim",
    # "p1_al_prec_sim",
    # "p1_al_mag_sim",
    # "p2_al_onset_sim",
    # "p2_al_prec_sim",
    # "p2_al_mag_sim",
    # "p3_fx_du_sim",
    # "p1_cs_sim",
    # "p2_cs_sim",
    # "p1_al_onset_mean",
    # "p1_al_prec_mean",
    # "p1_al_mag_mean",
    # "p1_cs_mean",
    # "p2_al_onset_mean",
    # "p2_al_prec_mean",
    # "p2_al_mag_mean",
    # "p2_cs_mean",
    # "p3_fx_du_mean",
    # "ec_start_fs_sim",
    # "distance_eye_hand_sim",
    # "ec_start_fs_mean",
    # "distance_eye_hand_mean",
    # "im_racket_ball_angle_sim",
    # "im_racket_ball_wrist_sim",
    # "im_ball_updown_sim",
    # "im_racket_ball_angle_mean",
    # "im_racket_ball_wrist_mean",
    # "im_ball_updown_mean",
    # "gender_sim",
    # "height_sim",
    # "age_sim",
    # "relationship",
    # "ecg_lfhf_sim",
    # "ecg_lfhf_mean",
    # "me_whole_sim",
    # "me_whole_mean",

]

med_norm_effect_list = []
med_mdiff_list = []
med_effect_list = []
low_effect_list = []
up_effect_list = []

med_efficient_list = []
med_inefficient_list = []
rhat_count = []
rhat_mean = []

for analyzed_features in features:
    # with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_supp\\" + "idata_" + analyzed_features + "_std.pkl",
    #           'rb') as handle:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_final\\" + "idata_" + analyzed_features + ".pkl",
              'rb') as handle:
        idata = pickle.load(handle)

        trace_post = az.extract(idata.posterior)

        med_mdiff = np.abs(np.nanmedian(trace_post['difference_of_means'].data.flatten())) / np.nanstd(
            trace_post['difference_of_means'].data.flatten())

        med_norm_effect = np.abs(np.nanmedian(trace_post['effect_size'].data.flatten())) / np.nanstd(
            trace_post['effect_size'].data.flatten())

        med_effect = np.nanmedian(trace_post['effect_size'].data.flatten(), keepdims=False)

        effet_size_arr = trace_post['effect_size'].data.flatten()
        low_effect, high_effect = az.hdi(effet_size_arr[~np.isnan(effet_size_arr)], hdi_prob=.95)
        med_ineffecient = np.nanmedian(trace_post['inefficient_mean'].data.flatten(), keepdims=False)
        med_effecient = np.nanmedian(trace_post['efficient_mean'].data.flatten(), keepdims=False)


        med_norm_effect_list.append(med_norm_effect)
        med_mdiff_list.append(med_mdiff)
        med_effect_list.append(med_effect)
        low_effect_list.append(low_effect)
        up_effect_list.append(high_effect)

        med_efficient_list.append(med_effecient)
        med_inefficient_list.append(med_ineffecient)

        rhat_count.append(np.sum(az.summary(idata)["r_hat"].values > 1.01))
        rhat_mean.append(np.nanmean(az.summary(idata)["r_hat"].values))


df = pd.DataFrame({"features": features,
                   "med_mdiff": med_mdiff_list,
                   "med_norm_effect": med_norm_effect_list,
                   "med_effect": med_effect_list,
                   "low_effect": low_effect_list,
                   "up_effect": up_effect_list,

                   "med_efficient": med_efficient_list,
                   "med_inefficient": med_inefficient_list,
                   "rhat_count": rhat_count,
                   "rhat_mean": rhat_mean
                   })

df.to_csv(os.path.join(DOUBLE_RESULTS_PATH_TTEST, "effect_size.csv"))
