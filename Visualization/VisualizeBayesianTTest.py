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

sns.set_style("white")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams.update({'xtick.labelsize': 25, 'ytick.labelsize': 25})


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


# features = [
#
#     "receiver_im_ball_updown",
#
#     "receiver_im_ball_wrist",
#     "receiver_start_fs",
#     "receiver_im_racket_ball_wrist",
#     "hitter_p1_al_mag",
#     "receiver_im_racket_dir",
#     "receiver_fixation_racket_latency",
#     "hitter_p2_al_prec",
#     "hitter_p2_al_mag",
#     "receiver_p2_al_mag",
#     "hitter_at_and_after_hit",
#     "hitter_p1_cs",
#     "hitter_p2_al_onset",
#     "hand_movement_sim",
#     "receiver_p1_al_onset",
#     "hitter_p1_al_prec",
#     "receiver_p1_al_mag",
#     "hitter_p1_cs", "hitter_p2_cs", "receiver_p1_cs", "receiver_p2_cs",
#     "receiver_p2_al_prec",
#     "receiver_p2_al_onset",
#     "hitter_fx_onset",
#     "receiver_distance_eye_hand",
#     "hitter_p1_al_onset",
#     "receiver_p1_al_prec",
#     "hitter_fx_duration",
#     "receiver_p3_fx_onset",
#     "height_sim",
#     "age_sim",
#     "relationship",
#     "receiver_p1_cs",
#     "receiver_p3_fx_duration",
#     "gender_sim",
#     "receiver_p3_fx",
#     "hitter_fx",
#     "hitter_p2_al",
#     "receiver_p2_al",
#     "receiver_p2_cs",
#     "hitter_p2_cs",
#     "receiver_p1_al",
#     "hitter_p1_al",
#     # "individual_skill", "individual_skill_sim"
#
# ]

features = ["height_sim"]

for analyzed_features in features:
    df = gedDF()
    print(analyzed_features)
    clean_df = df.dropna(subset=[analyzed_features])
    mean_ori = np.nanmean(clean_df[analyzed_features].values.reshape(-1, 1))
    std_ori = np.nanstd(clean_df[analyzed_features].values.reshape(-1, 1))

    with open(DOUBLE_RESULTS_PATH_TTEST + "\\model\\" + "idata_" + analyzed_features + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)
    trace_post = az.extract(idata.posterior)

    alpha = 0.05
    l = len(trace_post['efficient_mean'].data.flatten())
    low_bound = int(alpha / 2 * l)
    high_bound = int((1 - (alpha / 2)) * l)

    fig, ax = plt.subplots(figsize=(12, 8))
    for group, color in zip(['efficient_mean', 'inefficient_mean'],
                            ['#69A87F', '#B5152C']):
        data = (trace_post[group].data.flatten() * std_ori) + mean_ori

        # Estimate KDE
        kde = stats.gaussian_kde(data)
        # plot complete kde curve as line
        pos = np.linspace(np.min(data), np.max(data), 101)
        plt.plot(pos, kde(pos), color=color, label='{0} KDE'.format(group), linewidth=3.0)
        # Set shading bounds
        low = np.sort(data)[low_bound]
        high = np.sort(data)[high_bound]
        # plt.hlines(y=0, xmin=low, xmax=high, colors='#000000',  lw=4)
        # plot shaded kde
        shade = np.linspace(low, high, 101)
        plt.fill_between(shade, kde(shade), alpha=0.1, color=color, label="{0} 95% HPD Score".format(group))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        xmin, xmax = ax.get_xaxis().get_view_interval()
        if xmax > 100:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ymin,ymax = ax.get_yaxis().get_view_interval()
        ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))

    # plt.legend()
    ax.set_xlabel(features_explanation[analyzed_features], fontsize=28)
    #
    # plt.show()
    plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "MU\\" + analyzed_features + ".pdf", format='pdf', transparent=True)
    plt.close()
