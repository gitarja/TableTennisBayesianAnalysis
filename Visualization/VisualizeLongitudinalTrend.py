from Utils.Conf import DOUBLE_RESULTS_PATH_LONGITUDINAL, features_explanation
import pickle
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from Utils.GroupClassification import groupLabeling
from Utils.Conf import  DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
# sns.set_style("white")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams.update({'xtick.labelsize': 25, 'ytick.labelsize': 25})
# save the model
n = 10

def gedDF():
    inefficient_group, efficient_group = groupLabeling()

    # inefficient group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=inefficient_group, exclude_failure=True,
                                                    exclude_no_pair=False, hmm_probs=True)
    inefficient_features = inefficient_reader.getStableUnstableFailureFeatures(group_name="inefficient",
                                                                               success_failure=True,
                                                                               mod="skill_personal_perception_action_impact_other",
                                                                               with_control=True)
    inefficient_features["group"] = "inefficient"
    # efficient group
    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=efficient_group, exclude_failure=True,
                                                  exclude_no_pair=False, hmm_probs=True)
    efficient_features = efficient_reader.getStableUnstableFailureFeatures(group_name="efficient", success_failure=True,
                                                                           mod="skill_personal_perception_action_impact_other",
                                                                           with_control=True)
    efficient_features["group"] = "efficient"

    df = pd.concat([inefficient_features, efficient_features])

    return df
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

# features = ["hitter_bouncing_to_partner"]
for analyzed_features in features:
    print(analyzed_features)
    df = gedDF()
    print(analyzed_features)
    clean_df = df.dropna(subset=[analyzed_features])
    mean_ori = np.nanmean(clean_df[analyzed_features].values.reshape(-1, 1))
    std_ori = np.nanstd(clean_df[analyzed_features].values.reshape(-1, 1))

    with open(DOUBLE_RESULTS_PATH_LONGITUDINAL + "idata_" + analyzed_features + "_" + str(n) + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)


    # print(az.summary(
    #     idata,
    #     var_names=[
    #         "global_efficient", "global_inefficient",
    #         "global_efficient_seg", "global_inefficient_seg",
    #         "global_th_segment",
    #         "global_skill_slope",
    #         "global_diff_of_means",
    #         "global_seg_diff_of_means"
    #     ],
    # ))

    # axes = az.plot_forest(idata,
    #                       kind='forestplot',
    #                       var_names=["global_efficient", "global_efficient"],
    #                       combined=True,
    #                       figsize=(9, 7))
    # plt.show()
    # global_efficient = posterior["global_efficient"].mean(dim="sample")
    # global_efficient_seg = posterior["global_efficient_seg"].mean(dim="sample")
    #
    # global_inefficient = posterior["global_inefficient"].mean(dim="sample")
    # global_inefficient_seg = posterior["global_inefficient_seg"].mean(dim="sample")
    #
    # global_skill_slope = posterior["global_skill_slope"].mean(dim="sample")

    posterior = az.extract(idata.posterior)
    global_efficient = posterior["global_efficient"]
    global_efficient_seg = posterior["global_efficient_seg"]

    global_inefficient = posterior["global_inefficient"]
    global_inefficient_seg = posterior["global_inefficient_seg"]



    subjects_intercept = posterior["subjects_intercept"].mean(dim="subject_idx")
    sessions_intercept = posterior["sessions_intercept"].mean(dim="session_idx")





    global_intercept =  posterior["global_intercept"] + subjects_intercept + sessions_intercept
    global_th_segment = posterior["global_th_segment"]




    time = 100
    time_xi = xr.DataArray(np.arange(time) / 100)
    # plot line
    fig, ax = plt.subplots(figsize=(8, 8))

    y_efficient = global_intercept + global_efficient + global_efficient_seg * (
        time_xi) + global_th_segment * time_xi

    y_efficient_mean = global_intercept.mean()   + global_efficient.mean() + global_efficient_seg.mean() * (
        time_xi) + global_th_segment.mean() * time_xi

    y_inefficient = global_intercept + global_inefficient  + global_inefficient_seg * (
        time_xi) + global_th_segment * time_xi

    y_inefficient_mean = global_intercept.mean() + global_inefficient.mean() + global_inefficient_seg.mean() * (
        time_xi) + global_th_segment.mean() * time_xi



    az.plot_hdi(
        time_xi,
        y_efficient.values.reshape(4, 1000, time),
        hdi_prob=0.95,
        fill_kwargs={"alpha": 0.1, "linewidth": 0.1},
        color="#69A87F",
    )

    az.plot_hdi(
        time_xi,
        y_inefficient.values.reshape(4, 1000, time),
        hdi_prob=0.95,
        fill_kwargs={"alpha": 0.1, "linewidth": 0.1},
        color="#B5152C",
    )

    ax.plot(
        time_xi,
        y_efficient_mean,
        color="#69A87F",
        lw=3,
        linestyle='dashed'
    )

    ax.plot(
        time_xi,
        y_inefficient_mean,
        color="#B5152C",
        lw=3,
        linestyle='dashed'
    )
    ax.set_ylabel(features_explanation[analyzed_features], fontsize=28)
    ax.set_xlabel(r"T of episode / 100", fontsize=28)
    ax.set_frame_on(False)
    # ax.set_ylim([0.19, 0.25])

    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
    ax.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # plt.show()
    plt.savefig(DOUBLE_RESULTS_PATH_LONGITUDINAL + "trends\\"+analyzed_features+".pdf", format='pdf', transparent=True, bbox_inches='tight')
    plt.close()

