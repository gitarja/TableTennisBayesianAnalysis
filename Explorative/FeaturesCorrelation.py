import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
import numpy as np
from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
from Utils.Conf import DOUBLE_SUMMARY_FILE_PATH
from Utils.Conf import DOUBLE_FEATURES_FILE_PATH, DOUBLE_RESULTS_PATH_EXPLORATION

from Utils.GroupClassification import groupClassifcation
from scipy import stats
import seaborn as sns
avg_group, ineff_group, eff_group = groupClassifcation()

all_groups = np.concatenate([avg_group, ineff_group, eff_group])
all_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                        file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                        exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                        include_subjects=all_groups)

all_features = all_reader.getStableUnstableFailureFeatures(group_name="all", success_failure=True)

all_X = all_features.loc[:, all_features.columns != 'labels']

ordered_features = [
    "receiver_p1_al",
    "receiver_p2_al",
    "receiver_p3_fx",
    "receiver_p1_cs",
    "receiver_p2_cs",
    "receiver_p1_al_onset",
    "receiver_p1_al_prec",
    "receiver_p1_al_mag",
    "receiver_p2_al_mag",
    "receiver_p2_al_onset",
    "receiver_p2_al_prec",
    "receiver_p3_fx_onset",
    "receiver_p3_fx_duration",
    "receiver_im_racket_dir",
    "receiver_im_ball_updown",
    "receiver_im_racket_ball_angle",
    "receiver_im_racket_ball_wrist",
    "receiver_im_ball_wrist",
    "receiver_start_fs",
    "hand_movement_sim",
    "receiver_fixation_racket_latency",
    "receiver_distance_eye_hand",
    "hitter_p1_al",
    "hitter_p1_al_onset",
    "hitter_p1_al_prec",
    "hitter_p1_al_mag",
    "hitter_p1_cs",
    "hitter_p2_al",
    "hitter_p2_al_onset",
    "hitter_p2_al_prec",
    "hitter_p2_al_mag",
    "hitter_p2_cs",
    "hitter_fx",
    "hitter_fx_onset",
    "hitter_fx_duration",
    "hitter_at_and_after_hit",
    "receiver_skill"
]

corr_matrix = np.zeros(shape=(len(ordered_features), len(ordered_features)))
for i in range(len(ordered_features)):
    for j in range(len(ordered_features)):
        X = all_X.loc[:, [ordered_features[i], ordered_features[j]]].values
        is_nan = np.sum(np.isnan(X), axis=-1)
        X = X[is_nan==0]

        corr = stats.spearmanr(X[:, 0], X[:, 1], nan_policy="omit")

        if (np.isnan(corr[0]) & ((np.std(X[:, 0]) == 0) |  (np.std(X[:, 1]) == 0))):
            corr_matrix[i, j] = 1
        else:
            corr_matrix[i, j] = corr[0]
    #print(corr)

plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.show()
np.save(os.path.join(DOUBLE_RESULTS_PATH_EXPLORATION, "features_correlation.npy"), corr)