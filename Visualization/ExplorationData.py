import os

# This must happen before pymc is imported, so you might
# need to restart the kernel for it to take effect.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys

sys.path.append(os.path.dirname(__file__))
import numpy as np
from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, TARGET_ACC, DOUBLE_SUMMARY_FILE_PATH
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from GroupClassification import outliersDetection
import arviz as az
from Utils.Conf import DOUBLE_RESULTS_PATH, DOUBLE_FEATURES_FILE_PATH
from LongitudinalModels import CenteredModel, CenteredPolyModel
import pickle
from sklearn.preprocessing import StandardScaler

np.random.seed(1945)  # For Replicability

# ANALYZED_FEATURES
ANALYZED_FEATURES = ["receiver_al_p1_prec", "receiver_al_p2_prec"]

if __name__ == '__main__':
    n = 10

    # load single and double data
    single_fr = SubjectCrossValidation()
    double_fr = DoubleSubjectCrossValidation()
    fr = GlobalFeaturesReader(single_fr.getSummary(), double_fr.getSummary())
    X, y, group_label = fr.getSingleDoubleFeatures(col="skill", log_scale=False)

    X = np.average(X, axis=-1, keepdims=False)

    labels = outliersDetection(X, y)
    inlier_idx = np.argwhere(labels == 1).flatten()
    over_idx = np.argwhere(labels == 2).flatten()
    under_idx = np.argwhere(labels == 3).flatten()

    inlier_group = group_label[inlier_idx]
    over_group = group_label[over_idx]
    under_group = group_label[under_idx]

    # load data

    # control group
    control_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                                file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                include_subjects=inlier_group,
                                                exclude_failure=True, exclude_no_pair=True, hmm_probs=True)
    control_features = control_reader.getSegmentateFeatures(group_label="control", n_segment=n)

    # control group
    over_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                             file_summary_path=DOUBLE_SUMMARY_FILE_PATH, include_subjects=over_group,
                                             exclude_failure=True, exclude_no_pair=True, hmm_probs=True)
    over_features = over_reader.getSegmentateFeatures(group_label="inefficient", n_segment=n)

    # control group
    under_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH, include_subjects=under_group,
                                              exclude_failure=True, exclude_no_pair=True, hmm_probs=True)
    under_features = under_reader.getSegmentateFeatures(group_label="efficient", n_segment=n)


    for f in ANALYZED_FEATURES:

        plt.plot(control_features.groupby('th_segments')["th_segments"].median(),
                 control_features.groupby('th_segments')[f].median(), color="#377eb8")
        plt.plot(over_features.groupby('th_segments')["th_segments"].median(),
                 over_features.groupby('th_segments')[f].median(), color="#e41a1c")
        plt.plot(under_features.groupby('th_segments')["th_segments"].median(),
                 under_features.groupby('th_segments')[f].median(), color="#4daf4a")

        plt.show()
        plt.close()
