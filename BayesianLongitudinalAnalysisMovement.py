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
    CENTERED
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from GroupClassification import outliersDetection, movementSimilarity
import arviz as az
from Utils.Conf import DOUBLE_RESULTS_PATH, DOUBLE_FEATURES_FILE_PATH
from LongitudinalModels import CenteredModelSim
import pickle

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

    # get similarity group
    _, _, inner_labels_sim = movementSimilarity(inlier_group)
    _, _, over_labels_sim = movementSimilarity(over_group)
    _, _, under_labels_sim = movementSimilarity(under_group)

    # get group combination
    inlier_sim_group = inlier_group[inner_labels_sim == 1]
    inlier_dis_group = inlier_group[inner_labels_sim == 0]

    over_sim_group = over_group[over_labels_sim == 1]
    over_dis_group = over_group[over_labels_sim == 0]

    under_sim_group = under_group[under_labels_sim == 1]
    under_dis_group = under_group[under_labels_sim == 0]

    # control group
    control_sim_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                                    include_subjects=inlier_sim_group,
                                                    exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR)
    control_sim_features = control_sim_reader.getSegmentateFeatures(group_label="control_sim", n_segment=n)

    control_dis_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                                    include_subjects=inlier_dis_group,
                                                    exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR)
    control_dis_features = control_dis_reader.getSegmentateFeatures(group_label="control_dis", n_segment=n)

    # control group
    over_sim_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH, include_subjects=over_sim_group,
                                             exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR)
    over_sim_features = over_sim_reader.getSegmentateFeatures(group_label="over_sim", n_segment=n)

    over_dis_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH, include_subjects=over_dis_group,
                                                 exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR)
    over_dis_features = over_dis_reader.getSegmentateFeatures(group_label="over_dis", n_segment=n)

    # control group
    under_sim_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH, include_subjects=under_sim_group,
                                              exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR)
    under_sim_features = under_sim_reader.getSegmentateFeatures(group_label="under_sim", n_segment=n)


    under_dis_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH, include_subjects=under_dis_group,
                                              exclude_failure=True, exclude_no_pair=EXCLUDE_NO_PAIR)
    under_dis_features = under_dis_reader.getSegmentateFeatures(group_label="under_dis", n_segment=n)

    print(control_sim_features.shape)
    print(control_dis_features.shape)

    print(over_sim_features.shape)
    print(over_dis_features.shape)

    print(under_sim_features.shape)
    print(under_dis_features.shape)

    indv = pd.concat([control_sim_features, control_dis_features, over_sim_features, over_dis_features, under_sim_features, under_dis_features]).reset_index()

    # One Hot Encode Data
    dummies = pd.get_dummies(indv.group)
    # dummies.columns = ['control','over','under']
    df = indv.join(dummies)


    # az.plot_dist(df[analyzed_features].values)
    # plt.show()


    mu = df[analyzed_features].mean()
    sigma = df[analyzed_features].std() * 2
    session_id_idx, unique_ids = pd.factorize(df["session_id"])

    coords = {"ids": unique_ids, "obs": range(len(df[analyzed_features]))}

    model = CenteredModelSim(coords, df, BINOMINAL, session_id_idx, analyzed_features, n)

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
    plt.savefig(DOUBLE_RESULTS_PATH + "r_hat_" + analyzed_features + "_" + str(n) + ".png")
