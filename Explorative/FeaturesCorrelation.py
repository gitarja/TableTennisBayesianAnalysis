import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, DOUBLE_RESULTS_PATH_TTEST
from Double.GlobalFeaturesReader import ImpressionFeatures
import pandas as pd
from scipy import stats
import os.path


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
                                                                 return_group_skill=True)

upper_features, skill_upper = upper_reader.getImpressionFeatures(group="upper",
                                                                 mod=mod,
                                                                 return_group_skill=True)

X_lower = lower_features.loc[:, lower_features.columns != 'labels']
y_lower = lower_features["labels"].values

X_upper = upper_features.loc[:, upper_features.columns != 'labels']
y_upper = upper_features["labels"].values

X = pd.concat([X_lower, X_upper])

columns = X.columns
print(columns)
N = len(X.columns)
print(N)
pairs = []
pearson_score = []

correlation_mat = np.zeros((N, N))
for i in range(N):
    feature_i = X[columns[i]]
    for j in range(i+1, N, 1):
        feature_j = X[columns[j]]
        if (not("sim" in columns[i])) & (not("sim" in columns[j])):
            res = stats.pearsonr(feature_i, feature_j)
            correlation_mat[i, j] = np.abs(res.statistic)

            pairs.append([columns[i], columns[j]])
            pearson_score.append(np.abs(res.statistic))





corr_df = pd.DataFrame({"pairs": pairs, "pearson_score": pearson_score})

corr_df.to_csv(os.path.join(DOUBLE_RESULTS_PATH_TTEST, "pearson.csv"))

# sns.heatmap(correlation_mat, annot=False, cmap=sns.color_palette("Blues", as_cmap=True), fmt=".2f")

# plt.show()