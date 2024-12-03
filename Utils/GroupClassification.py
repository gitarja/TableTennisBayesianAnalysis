import os

from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader
from BayesianAnalysis.GroupClassification import outliersDetection, outliersLabeling
import numpy as np
from Utils.Conf import DOUBLE_GROUPS_FILE_PATH
import os

rng = np.random.default_rng(seed=42)
def groupClassifcation():
    # # load single and double data
    # single_fr = SubjectCrossValidation()
    # double_fr = DoubleSubjectCrossValidation()
    # fr = GlobalFeaturesReader(single_fr.getSummary(), double_fr.getSummary())
    # X, y, group_label = fr.getSingleDoubleFeatures(col="skill", log_scale=False)
    #
    # X = np.average(X, axis=-1, keepdims=False)
    #
    # labels = outliersDetection(X, y)
    # inlier_idx = np.argwhere(labels == 1).flatten()
    # inefficient_idx = np.argwhere(labels == 2).flatten()
    # efficient_idx = np.argwhere(labels == 3).flatten()
    #
    # inlier_group = group_label[inlier_idx]
    # inefficient_group = group_label[inefficient_idx]
    # efficient_group = group_label[efficient_idx]

    inlier_group = np.load(os.path.join(DOUBLE_GROUPS_FILE_PATH, "inlier_group_5std.npy"))
    inefficient_group = np.load(os.path.join(DOUBLE_GROUPS_FILE_PATH, "inefficient_group_5std.npy"))
    efficient_group = np.load(os.path.join(DOUBLE_GROUPS_FILE_PATH, "efficient_group_5std.npy"))
    return inlier_group, inefficient_group, efficient_group


def groupLabeling():
    # # load single and double data
    # single_fr = SubjectCrossValidation()
    # double_fr = DoubleSubjectCrossValidation()
    # fr = GlobalFeaturesReader(single_fr.getSummary(), double_fr.getSummary())
    # X, y, group_label = fr.getSingleDoubleFeatures(col="skill", log_scale=False)
    #
    # X = np.average(X, axis=-1, keepdims=False)
    #
    # labels = outliersLabeling(X, y)
    # upper_idx = np.argwhere(labels == 1).flatten()
    # lower_idx = np.argwhere(labels == 0).flatten()
    #
    # upper_group = group_label[upper_idx]
    # lower_group = group_label[lower_idx]

    lower_group = np.load(os.path.join(DOUBLE_GROUPS_FILE_PATH, "lower_group_5std.npy"))
    upper_group = np.load(os.path.join(DOUBLE_GROUPS_FILE_PATH, "upper_group_5std.npy"))

    print(len(lower_group))
    print(len(upper_group))
    return lower_group, upper_group
