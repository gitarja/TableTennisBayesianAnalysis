import os.path

import matplotlib.pyplot as plt


from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST
import arviz as az
import pickle
import pandas as pd

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



]

loo_efficient_skew_list = []
loo_inefficient_skew_list = []
loo_efficient_list = []
loo_inefficient_list = []
for analyzed_features in features:
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_wider\\" + "idata_" + analyzed_features + ".pkl",
              'rb') as handle:
        idata_skew = pickle.load(handle)
    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model\\" + "idata_" + analyzed_features + ".pkl",
              'rb') as handle:
        idata = pickle.load(handle)

    loo_efficient_skew_list.append(az.loo(idata_skew, var_name="efficient").elpd_loo)
    loo_inefficient_skew_list.append(az.loo(idata_skew, var_name="inefficient").elpd_loo)

    loo_efficient_list.append(az.loo(idata, var_name="efficient").elpd_loo)
    loo_inefficient_list.append(az.loo(idata, var_name="inefficient").elpd_loo)


df = pd.DataFrame({"features": features,
                   "loo_efficient_skew": loo_efficient_skew_list,
                   "loo_inefficient_skew": loo_inefficient_skew_list,
                   "loo_efficient": loo_efficient_list,
                   "loo_inefficient": loo_inefficient_list

                   })

df.to_csv(os.path.join(DOUBLE_RESULTS_PATH_TTEST, "model_comparison_wider.csv"))


