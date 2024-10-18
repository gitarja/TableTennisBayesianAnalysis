import numpy as np
import arviz as az
from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import scipy.stats as stats
from GroupClassification import outliersDetection
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC
import pickle
from sklearn.preprocessing import StandardScaler

np.random.seed(1945)  # For Replicability

if __name__ == '__main__':

    rng = np.random.default_rng(seed=42)

    # load single and double data
    single_fr = SubjectCrossValidation()
    double_fr = DoubleSubjectCrossValidation()
    fr = GlobalFeaturesReader(single_fr.getSummary(), double_fr.getSummary())
    X, y, group_label = fr.getSingleDoubleFeatures(col="skill", log_scale=False)

    X = np.average(X, axis=-1, keepdims=False)

    labels = outliersDetection(X, y)
    inlier_idx = np.argwhere(labels == 1).flatten()
    inefficient_idx = np.argwhere(labels == 2).flatten()
    efficient_idx = np.argwhere(labels == 3).flatten()

    inlier_group = group_label[inlier_idx]
    inefficient_group = group_label[inefficient_idx]
    efficient_group = group_label[efficient_idx]

    # features = ["receiver_p1_al",
    #             "receiver_p2_al",
    #             "receiver_pursuit",
    #             "receiver_pursuit_duration",
    #             "receiver_p1_al_prec",
    #             "receiver_p1_al_onset",
    #             "receiver_p2_al_prec",
    #             "receiver_p2_al_onset",
    #             "receiver_p1_cs",
    #             "receiver_p2_cs",
    #             "hitter_p1_al",
    #             "hitter_p2_al",
    #             "hitter_pursuit",
    #             "hitter_pursuit_duration",
    #             "hitter_p1_al_prec",
    #             "hitter_p1_al_onset",
    #             "hitter_p2_al_prec",
    #             "hitter_p2_al_onset",
    #             "hitter_p1_cs",
    #             "hitter_p2_cs",
    #             "receiver_start_fs_std",
    #             "receiver_racket_to_root_std",
    #             "receiver_fs_ball_racket_dir_std",
    #             "hand_mov_sim",
    #             "single_mov_sim",
    #             "stable_percentage", "bouncing_point_var_p1", "bouncing_point_var_p2"]

    features = ["hitter_pursuit"]

    # control group
    control_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                include_subjects=inlier_group, exclude_failure=True,
                                                exclude_no_pair=True, hmm_probs=True)
    control_features = control_reader.getGlobalFeatures(group_label="control")

    # inefficientestimated group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=inefficient_group, exclude_failure=True,
                                                    exclude_no_pair=True, hmm_probs=True)
    inefficient_features = inefficient_reader.getGlobalFeatures(group_label="inefficient")

    # efficientestimated group
    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=efficient_group, exclude_failure=True,
                                                  exclude_no_pair=True, hmm_probs=True)
    efficient_features = efficient_reader.getGlobalFeatures(group_label="efficient")

    print(control_features.shape)
    print(inefficient_features.shape)
    print(efficient_features.shape)
    indv = pd.concat([control_features, inefficient_features, efficient_features]).reset_index()

    # One Hot Encode Data
    dummies = pd.get_dummies(indv.group)
    # dummies.columns = ['control','inefficient','efficient']
    df = indv.join(dummies)

    # scaled control
    scaler = StandardScaler()

    control_skill_scaled = scaler.fit(df["subject_skill"].values.reshape(-1, 1))
    df["subject_skill"] = scaler.transform(df["subject_skill"].values.reshape(-1, 1))

    subjects_idx, subjects_unique = pd.factorize(df["subject"])

