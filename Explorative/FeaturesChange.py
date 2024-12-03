import os

import matplotlib.pyplot as plt

os.environ['OMP_NUM_THREADS'] = '8'
# This must happen before pymc is imported, so you might
# need to restart the kernel for it to take effect.
import pandas as pd

import sys

sys.path.append(os.path.dirname(__file__))
import numpy as np

from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
from Utils.Conf import DOUBLE_SUMMARY_FILE_PATH
from Utils.Conf import DOUBLE_FEATURES_FILE_PATH


from Utils.GroupClassification import groupClassifcation, groupLabeling

import seaborn as sns
np.random.seed(1945)  # For Replicability
inefficient_group, efficient_group = groupLabeling()
all_groups = np.concatenate([inefficient_group, efficient_group])

efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                          file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                          exclude_failure=False, exclude_no_pair=False, hmm_probs=True,
                                          include_subjects=efficient_group)

efficient_features = efficient_reader.getStableUnstableFailureFeatures(group_name="train_subjects", success_failure=True,
                                                               mod="full_mode", timepoint=True)


efficient_features["group_name"] = "top"

inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                         file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                         exclude_failure=False, exclude_no_pair=False, hmm_probs=True,
                                         include_subjects=inefficient_group)

inefficient_features = inefficient_reader.getStableUnstableFailureFeatures(group_name="test_subjects",
                                                             success_failure=True,
                                                             mod="full_mode", timepoint=True)

inefficient_features["group_name"] = "low"


features = pd.concat([efficient_features, inefficient_features], ignore_index=True)
features = features.loc[features["hitter_timepoint"] <=20]

sns.lineplot(data=features, x="hitter_timepoint", y="hitter_p1_al_prec", hue="group_name")
# sns.lineplot(data=features, x="hitter_timepoint", y="hitter_p1_al_prec", hue="group_name", units="hitter", estimator=None, lw=1,)
plt.show()
