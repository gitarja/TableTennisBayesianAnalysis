import numpy as np

from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST
import arviz as az
import pickle
import pandas as pd

feature_type = "individual_characteristic"
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
else:
    features = [
        "individual_skill",
        "individual_skill_sim",
        "height_sim",
        "age_sim",
        "relationship",
    ]
models = []

mean_diff_median_list = []
efficient_median_list = []
inefficient_median_list = []

mean_diff_hdi_list = []
efficient_hdi_list = []
inefficient_hdi_list = []
for analyzed_features in features:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)

    trace_post = az.extract(idata.posterior)

    mean_difference = trace_post['difference_of_means'].data.flatten()
    mean_effecient = trace_post['efficient_mean'].data.flatten()
    mean_ineffecient = trace_post['inefficient_mean'].data.flatten()

    # hdi
    mean_differece_hdi = [round(num, 2) for num in az.hdi(mean_difference, hdi_prob=.95)]
    efficient_hdi = [round(num, 2) for num in az.hdi(mean_effecient, hdi_prob=.95)]
    inefficient_hdi = [round(num, 2) for num in az.hdi(mean_ineffecient, hdi_prob=.95)]

    # median
    mean_differece_median = np.median(mean_difference)
    efficient_median = np.median(mean_effecient)
    inefficient_median = np.median(mean_ineffecient)

    mean_diff_median_list.append(mean_differece_median)
    efficient_median_list.append(efficient_median)
    inefficient_median_list.append(inefficient_median)

    mean_diff_hdi_list.append(mean_differece_hdi)
    efficient_hdi_list.append(efficient_hdi)
    inefficient_hdi_list.append(inefficient_hdi)

summary_df = pd.DataFrame(
    {"features": features,
     "mean_diff_median": mean_diff_median_list,
     "efficient_median": efficient_median_list,
     "inefficient_median": inefficient_median_list,
     "mean_diff_hdi": mean_diff_hdi_list,
     "efficient_hdi": efficient_hdi_list,
     "inefficient_hdi": inefficient_hdi_list})
summary_df = summary_df.round(
    {'mean_diff_median': 2,
     'efficient_median': 2,
     'inefficient_median': 2,
     'mean_diff_hdi': 2,
     'efficient_hdi': 2,
     'inefficient_hdi': 2})
summary_df.to_csv(DOUBLE_RESULTS_PATH_TTEST + feature_type + "_summary_bayesian.csv")
