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
import pymc as pm
import matplotlib.pyplot as plt
from GroupClassification import outliersDetection
import arviz as az
from Utils.Conf import DOUBLE_RESULTS_PATH, DOUBLE_FEATURES_FILE_PATH
from LongitudinalModels import NonCenteredModel, CenteredModel
import pickle
from sklearn.preprocessing import StandardScaler

np.random.seed(1945)  # For Replicability

if __name__ == '__main__':
    n = 10

    analyzed_features = ANALYZED_FEATURES

    print("Analysis of " + analyzed_features + " with n_segment: " + str(n))

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
    control_features = control_reader.getSegmentateFeatures(group_label="control", n_segment=n)

    # control group
    over_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                             file_summary_path=DOUBLE_SUMMARY_FILE_PATH, include_subjects=over_group,
                                             exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR, hmm_probs=True)
    over_features = over_reader.getSegmentateFeatures(group_label="inefficient", n_segment=n)

    # control group
    under_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH, include_subjects=under_group,
                                              exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR, hmm_probs=True)
    under_features = under_reader.getSegmentateFeatures(group_label="efficient", n_segment=n)

    print(control_features.shape)
    print(over_features.shape)
    print(under_features.shape)
    indv = pd.concat([control_features, over_features, under_features]).reset_index()

    # One Hot Encode Data
    dummies = pd.get_dummies(indv.group)
    # dummies.columns = ['control','over','under']
    df = indv.join(dummies)

    # scaled control
    # scaler = StandardScaler()
    # control_skill_scaled = scaler.fit(df["subject_skill"].values.reshape(-1, 1))
    # df["subject_skill"] = scaler.transform(df["subject_skill"].values.reshape(-1, 1))



    # # standarized features
    # scaler = StandardScaler()
    # control_scaled = scaler.fit_transform(df[analyzed_features].values.reshape(-1, 1))
    # df[analyzed_features] = control_scaled.flatten()



    subjects_idx, subjects_unique = pd.factorize(df["subject"])

    coords = {"subject_idx": subjects_unique, "obs": range(len(df[analyzed_features])), "group": ["control", "inefficient", "efficient"]}
    if CENTERED:
        model = CenteredModel(coords, df, BINOMINAL, subjects_idx, analyzed_features, n)
    else:
        model = NonCenteredModel(coords, df, BINOMINAL, subjects_idx, analyzed_features, n)

    print("Run: Start")
    with model:
        print(model.debug())
        pm.model_to_graphviz(model).view()
        idata_m3 = pm.sample_prior_predictive()
        idata_m3.extend(
            pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True}, draws=N_SAMPLES,
                      chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
        )
        idata_m3.extend(pm.sample_posterior_predictive(idata_m3))

    # save the model
    with open(DOUBLE_RESULTS_PATH + "idata_m3_" + analyzed_features + "_" + str(n) + ".pkl", 'wb') as handle:
        print("write data into: " + "idata_m3_" + analyzed_features + "_" + str(n) + ".pkl")
        pickle.dump(idata_m3, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plot rhat
    nc_rhat = az.rhat(idata_m3)
    ax = (nc_rhat.max()
          .to_array()
          .to_series()
          .plot(kind="barh"))
    print(az.summary(idata_m3))
    plt.savefig(DOUBLE_RESULTS_PATH + "r_hat_" + analyzed_features + "_" + str(n) + ".png")
    plt.show()

