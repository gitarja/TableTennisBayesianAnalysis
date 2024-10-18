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
# poly model or not
POLY_BOOL = [False, True]
# ANALYZED_FEATURES
ANALYZED_FEATURES = [ "receiver_al_p1_prec", "receiver_al_p2_prec", "hand_mov_sim", "receiver_cs_p1", "receiver_cs_p2"]
HITTER_BOOL = [False, False, False, False, False]
# Binominal
BINOMINAL = [ False, False, False, False, False]
# Centered
CENTERED = True
EXCLUDE_NO_PAIR = True

if __name__ == '__main__':
    n = 5





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
    for feature, hitter, bin in zip(ANALYZED_FEATURES, HITTER_BOOL, BINOMINAL):
        for poly in POLY_BOOL:
            analyzed_features = feature

            print("Analysis of " + analyzed_features + " with n_segment: " + str(n))

            # az.plot_dist(df[analyzed_features].values.astype(int), kind="hist")
            # plt.show()
            # az.plot_dist(control_features[analyzed_features].values, kind="hist")
            # plt.show()
            # az.plot_dist(over_features[analyzed_features].values, kind="hist")
            # plt.show()
            # az.plot_dist(under_features[analyzed_features].values, kind="hist")
            # plt.show()

            if hitter:
                subjects = df["hitter"]
            else:
                subjects = df["receiver"]
            subjects_idx, subjects_unique = pd.factorize(subjects)

            coords = {"subject_idx": subjects_unique, "obs": range(len(df[analyzed_features])),
                      "group": ["control", "inefficient", "efficient"]}

            if poly:
                model = CenteredPolyModel(coords, df, bin, subjects_idx, analyzed_features, n, hitter=hitter)
            else:
                model = CenteredModel(coords, df, bin, subjects_idx, analyzed_features, n, hitter=hitter)

            print("Run: Start")
            with model:
                print(model.debug())
                # pm.model_to_graphviz(model).view()
                idata_m3 = pm.sample_prior_predictive()


                idata_m3.extend(
                    pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                              draws=N_SAMPLES,
                              chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
                )
                idata_m3.extend(pm.sample_posterior_predictive(idata_m3))

            # save the model
            file_name = "idata_m3_"
            image_name = "r_hat_"

            if poly:
                file_name = "idata_m3_poly_"
                image_name = "r_hat_poly_"
            with open(DOUBLE_RESULTS_PATH + file_name + analyzed_features + "_" + str(n) + ".pkl", 'wb') as handle:
                print("write data into: " + file_name + analyzed_features + "_" + str(n) + ".pkl")
                pickle.dump(idata_m3, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # plot rhat
            nc_rhat = az.rhat(idata_m3)
            ax = (nc_rhat.max()
                  .to_array()
                  .to_series()
                  .plot(kind="barh"))
            print(az.summary(idata_m3))
            plt.savefig(DOUBLE_RESULTS_PATH + image_name + analyzed_features + "_" + str(n) + ".png")
            # plt.show()
