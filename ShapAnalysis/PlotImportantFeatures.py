import os.path

from SHAPPlots import plotSHAP, plotShapSummary, plotShapInteraction, plotShapAbsoulte
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import shap
from Utils.Conf import features_explanation
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.ticker import FormatStrFormatter

# sns.set_theme()
# sns.set(font_scale=5)
# sns.set(font="Arial")
# sns.set_style("white")
plt.rcParams["text.usetex"] = True
plt.rcParams['font.sans-serif'] = "Comic Sans MS"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams.update({'xtick.labelsize': 25, 'ytick.labelsize': 25})

# Direct input
# Options
# params = {'text.usetex' : True,
#           'font.size' : 18,
#           'font.family' : 'lmodern',
#
#           }
# plt.rcParams.update(params)

def plotSHAP(shap_values, x, all_columns, columns, results_path="", prefix="", alpha=0.15, dot_size=10, axes=None):
    # plt.rcParams["font.family"] = "Arial"
    # plt.rcParams["font.size"] = 2
    # combing shap and x
    shap_columns = ["shap_" + c for c in all_columns]
    summary_df = pd.DataFrame(np.concatenate([x.values, shap_values], axis=-1), columns=all_columns + shap_columns)
    explanation = shap.Explanation(values=shap_values, data=x.values, feature_names=all_columns)

    for i in range(len(columns)):
        for j in range(len(columns[i])):
            c = columns[i, j]
            ax = axes[i, j]

            ref_value = explanation[:, c]

            xmin = np.nanpercentile(ref_value.data, 0.5)
            xmax = np.nanpercentile(ref_value.data, 99.5)
            ymin = np.min(ref_value.values)
            ymax = np.max(ref_value.values)
            y_abs_max = np.max(np.abs(ref_value.values))

            if c == "receiver_skill":
                xmax = 1
            elif c == "receiver_p3_fx_onset":
                xmin = 0

            shap.plots.scatter(ref_value, show=False, alpha=alpha, xmin=xmin, xmax=xmax, dot_size=dot_size, ax=ax,
                               hist=False)
            snipset_summary = summary_df.loc[(summary_df[c] >= xmin) & (summary_df[c] <= xmax)]
            # if (c != "receiver_p1_al") & (c != "receiver_p2_al") & (c != "receiver_p3_fx") & (c != "hitter_p1_al") & (
            #         c != "hitter_p2_al") & (c != "hitter_fx") & (c != "hitter_p1_cs") & (c != "hitter_p2_cs") & (
            #         c != "receiver_p1_cs") & (c != "receiver_p2_cs"):
                # a = sns.regplot(data=snipset_summary, x=c, y="shap_" + c, order=2, color="#81b1d3",
                #                 line_kws=dict(color="#252525"), scatter=False, ax=ax)

            ax.set_xlabel(features_explanation[c], fontsize=28)
            # ax.set_xlabel("a")
            ax.set_ylabel("")
            ax.axhline(y=0., color="#525252", linestyle=":")
            if c == "individual_skill":
                ax.set_xlim(0.7, 1.01)
            if c == "individual_skill_sim":
                ax.set_xlim(0., 0.5)

            ax.set_ylim(-1 * y_abs_max, y_abs_max)
            # set ticks
            xmin, xmax = ax.get_xaxis().get_view_interval()
            if xmax > 10:
                x_ticks = np.arange(xmin, xmax, int((xmax - xmin) / 4)).astype(int)[1:]
            else:
                x_ticks = np.arange(xmin, xmax, (xmax - xmin) / 4)[1:]
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_xticks(x_ticks)

            ymin, ymax = ax.get_yaxis().get_view_interval()
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # y_ticks = np.arange(ymin, ymax, (ymax - ymin) / 3)
            # ax.set_yticks(y_ticks)

            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)

            # plt.show()


label = "all_lower_upper"
shap_results = np.load("Results\\Final2\\Full-model\\" + label + "_shap.npy")
xval_results = pd.read_pickle("Results\\Final2\\Full-model\\" + label + "_xval.pkl")

results_path = os.path.join(
    "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final\\", label)

# important_features = np.asarray([
#     "receiver_im_ball_updown",
#     "receiver_im_racket_ball_angle",
#     "individual_skill",
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
#     "individual_skill_sim",
# ]).reshape((4, 5))


important_features = np.asarray([
    "p1_al_prec_sim",
    "p1_al_prec_mean",
    "distance_eye_hand_sim",
    "ec_start_fs_sim",
    "p2_cs_sim",
    "p3_fx_du_sim",
    "p2_cs_mean",
    "p1_cs_mean",
    "height_sim",
    "im_racket_ball_wrist_mean",
    "p2_al_mag_sim",
    "p3_fx_onset_sim",
    "p2_al_mag_mean",
    "p1_al_mag_mean",
    "im_ball_wrist_mean",
    "distance_eye_hand_mean",
    "im_ball_updown_mean",
    "p1_cs_sim",
    "im_ball_wrist_sim",
    "p2_al_prec_mean",

]).reshape((4, 5))



# important_features = np.asarray([
# "hitter_p2_cs",
# "receiver_p2_cs",
# "receiver_p2_al",
# "hitter_p2_al",
# "hitter_fx",
# "receiver_p3_fx",
# "gender_sim",
# "receiver_p3_fx_duration",
# "receiver_p1_cs",
# "relationship",
# "age_sim",
# "height_sim",
# "receiver_p3_fx_onset",
# "hitter_fx_duration",
# "receiver_p1_al_prec",
# "hitter_p1_al_onset",
# "receiver_distance_eye_hand",
# "hitter_fx_onset",
# "receiver_p2_al_onset",
# "receiver_p2_al_prec",
#
# ]).reshape((4, 5))

fig, axs = plt.subplots(nrows=4, ncols=5, constrained_layout=True)
plotSHAP(shap_values=shap_results, x=xval_results, all_columns=xval_results.columns.values.tolist(),
         columns=important_features, results_path=results_path, alpha=0.5, dot_size=60, axes=axs)
sns.despine(fig=fig)

fig.set_size_inches(30.5, 22.5)
plt.savefig(results_path + "\\important_features2.pdf", format='pdf', transparent=True, bbox_inches='tight')
plt.close()
