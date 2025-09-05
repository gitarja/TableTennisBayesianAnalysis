from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH
from Double.GlobalFeaturesReader import ImpressionFeatures
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
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
    mod = "skill_personal_perception_action_impact_ecg_me"
    # mod = "top-5"
    lower_features, skill_lower = lower_reader.getImpressionFeatures(group="lower",
                                                                     mod=mod,
                                                                     return_group_skill=True)

    upper_features, skill_upper = upper_reader.getImpressionFeatures(group="upper",
                                                                     mod=mod,
                                                                     return_group_skill=True)


    X_lower = lower_features.loc[:, lower_features.columns != 'labels']


    X_upper = upper_features.loc[:, upper_features.columns != 'labels']


    df = pd.concat([X_lower, X_upper])

    fig, axes = plt.subplots(8, 5, figsize=(8.27, 11.69))
    features = df.columns
    n_cols = len(features)

    i = 0
    for f in features:
        row, col_idx = divmod(i, 5)
        ax = axes[row, col_idx]
        sns.histplot(df[f], ax=ax, fill=True, edgecolor=None, element="step")
        ax.set_title(f, fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=8)
        i += 1

    # Hide any unused axes (in case < 48 columns)
    for j in range(len(df.columns), 24):
        row, col_idx = divmod(j, 8)
        axes[row, col_idx].axis('off')

    plt.tight_layout()

    results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\Figures\\Material\\"
    plt.savefig(os.path.join(results_path, "analysis1_distribution.pdf"))
