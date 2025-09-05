import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH
from Double.GlobalFeaturesReader import ImpressionFeatures
import pymc as pm
import pandas as pd
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST
import seaborn as sns
import os

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (4, 4)
np.random.seed(1945)  # For Replicability

if __name__ == '__main__':

    lower_group, upper_group = groupLabeling()

    # lower group
    lower_reader = ImpressionFeatures(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                      file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                      include_subjects=lower_group, exclude_failure=False,
                                      exclude_no_pair=True)
    # upper group
    upper_reader = ImpressionFeatures(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                      file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                      include_subjects=upper_group, exclude_failure=False,
                                      exclude_no_pair=True)

    label = "all_lower_upper"
    # mod = "skill_personal_perception_action_impact_ecg_me"
    mod = "skill_personal_perception_action_impact_ecg_me"
    lower_features, skill_lower = lower_reader.getImpressionFeatures(group="lower",
                                                                     mod=mod,
                                                                     return_group_skill=True, return_control=True)

    upper_features, skill_upper = upper_reader.getImpressionFeatures(group="upper",
                                                                     mod=mod,
                                                                     return_group_skill=True, return_control=True)

    X_lower = lower_features.loc[:, lower_features.columns != 'labels']
    y_lower = lower_features["labels"].values

    X_upper = upper_features.loc[:, upper_features.columns != 'labels']
    y_upper = upper_features["labels"].values

    X_upper["group"] = "efficient"
    X_lower["group"] = "inefficient"
    X = pd.concat([X_lower, X_upper])



    features = [
        "p1_al_prec_sim",
        "p1_cs_mean",
        "im_racket_ball_wrist_mean",
        "im_racket_ball_angle_sim",
        "p1_al_prec_mean",
        "p2_cs_mean",


    ]

    palette = {
        'efficient': '#67A77F',
        'inefficient': '#B5152C',

    }

    for f in features:
        fig, ax = plt.subplots()
        sns.stripplot(
            data=X, x="group", y=f, hue="group",
            dodge=True, alpha=.2, legend=False, palette=palette,
        )
        sns.pointplot(
            data=X, x="group",  y=f, hue="group",
            dodge=.4, linestyle="none", errorbar=('ci', 95), capsize=.1,
             markersize=5, markeredgewidth=1,palette=palette, estimator="median"
        )
        plt.legend([], [], frameon=False)
        results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\all_lower_upper\\Distribution\\"


        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plt.show()
        plt.savefig(os.path.join(results_path, "SHAP-important_" +  f + ".pdf"))
        plt.close()
