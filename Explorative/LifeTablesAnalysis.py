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
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, BINOMINAL, ANALYZED_FEATURES, TARGET_ACC, EXCLUDE_NO_PAIR, \
    CENTERED, DOUBLE_SUMMARY_FILE_PATH
import pandas as pd
import matplotlib.pyplot as plt
from GroupClassification import outliersDetection
import arviz as az
from Utils.Conf import DOUBLE_RESULTS_PATH, DOUBLE_FEATURES_FILE_PATH
import seaborn as sns

np.random.seed(1945)  # For Replicability

if __name__ == '__main__':
    interval_length = 10

    analyzed_features = "hitter_pf_rate"

    print("Analysis of " + analyzed_features + " with n_segment: " + str(interval_length))

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
                                                exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR, hmm_probs=True)
    control_features = control_reader.getSegmentateFeatures(group_label="control", n_segment=interval_length)

    # control group
    over_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                             file_summary_path=DOUBLE_SUMMARY_FILE_PATH, include_subjects=over_group,
                                             exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR, hmm_probs=True)
    over_features = over_reader.getSegmentateFeatures(group_label="inefficient", n_segment=interval_length)

    # control group
    under_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH, include_subjects=under_group,
                                              exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR, hmm_probs=True)
    under_features = under_reader.getSegmentateFeatures(group_label="efficient", n_segment=interval_length)

    indv = pd.concat([control_features, over_features, under_features]).reset_index()

    # One Hot Encode Data
    dummies = pd.get_dummies(indv.group)
    # dummies.columns = ['control','over','under']
    df = indv.join(dummies)

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the observation of segments

    interval_length = 10
    interval_bounds = np.arange(0, df["th_segments"].max() + interval_length + 1, interval_length)
    n_intervals = interval_bounds.size - 1
    intervals = np.arange(n_intervals)

    ax.hist(
        df[df.group == "efficient"]["th_segments"].values,
        bins=interval_bounds,
        lw=0,
        color="C3",
        alpha=0.5,
        label="Efficient",
    )

    ax.hist(
        df[df.group == "inefficient"]["th_segments"].values,
        bins=interval_bounds,
        lw=0,
        color="C7",
        alpha=0.5,
        label="Inefficient",
    )

    ax.set_xlim(0, interval_bounds[-1])
    ax.set_xlabel("#Episodes")

    ax.set_ylabel("Number of observations")

    ax.legend()

    plt.show()

    # plot the observation of segments

    # overall
    # Plot the responses for different events and regions
    # sns.displot(x=analyzed_features,
    #              hue="group",
    #              data=df, kind="kde")
    #
    # plt.show()
