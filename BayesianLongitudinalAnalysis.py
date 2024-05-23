import os
# # This must happen before pymc is imported, so you might
# # need to restart the kernel for it to take effect.
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, TARGET_ACC, ANALYZED_FEATURES
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from OutliersLib import OutliersDetection
import arviz as az
from Utils.Conf import DOUBLE_RESULTS_PATH, DOUBLE_FEATURES_FILE_PATH
import pickle
np.random.seed(1945)  # For Replicability

if __name__ == '__main__':
    n = 10

    analyzed_features = ANALYZED_FEATURES
    # load single and double data
    single_fr = SubjectCrossValidation()
    double_fr = DoubleSubjectCrossValidation()
    fr = GlobalFeaturesReader(single_fr.getSummary(), double_fr.getSummary())
    X, y, group_label = fr.getSingleDoubleFeatures(col="skill", log_scale=False)

    X = np.average(X, axis=-1, keepdims=False)

    labels = OutliersDetection(X, y)
    inlier_idx = np.argwhere(labels == 1).flatten()
    over_idx = np.argwhere(labels == 2).flatten()
    under_idx = np.argwhere(labels == 3).flatten()


    inlier_group = group_label[inlier_idx]
    over_group = group_label[over_idx]
    under_group = group_label[under_idx]

    # load data


    # control group
    control_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH, include_subjects=inlier_group, exclude_failure=False, exclude_no_pair=False)
    control_features = control_reader.getSegmentateFeatures(group_label="control", n_segment=n)

    # control group
    over_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH, include_subjects=over_group, exclude_failure=False, exclude_no_pair=False)
    over_features = over_reader.getSegmentateFeatures(group_label="over", n_segment=n)

    # control group
    under_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH, include_subjects=under_group, exclude_failure=False, exclude_no_pair=False)
    under_features = under_reader.getSegmentateFeatures(group_label="under", n_segment=n)

    print(control_features.shape)
    print(over_features.shape)
    print(under_features.shape)
    indv = pd.concat([control_features, over_features, under_features]).reset_index()

    # One Hot Encode Data
    dummies = pd.get_dummies(indv.group)
    # dummies.columns = ['control','over','under']
    df = indv.join(dummies)



    mu = df[analyzed_features].mean()
    sigma = df[analyzed_features].std() * 2
    session_id_idx, unique_ids = pd.factorize(df["session_id"])

    coords = {"ids": session_id_idx, "obs": range(len(df[analyzed_features]))}
    with pm.Model(coords=coords) as model:
        th_segments = pm.Data("th_segments", df["th_segments"].values)

        control = pm.Data("control", df["control"].values.astype(float))
        over = pm.Data("over", df["control"].values.astype(float))
        under = pm.Data("under", df["control"].values.astype(float))


        # level 1
        global_intercept_raw = pm.Normal("global_intercept_raw ", 0, 0.5)
        global_th_segment_raw  = pm.Normal("global_th_segment_raw ", 0, 0.5)

        global_intercept_mu =  pm.Normal("global_intercept_mu", 0, 1)
        global_intercept_sigma = pm.HalfNormal("global_intercept_sigma", 1)


        global_th_segment_mu = pm.Normal("global_th_segment_mu", 0, 1)
        global_th_segment_sigma = pm.HalfNormal("global_th_segment_sigma", 1)

        global_intercept = pm.Deterministic("global_intercept ", global_intercept_mu + global_intercept_raw * global_intercept_sigma)
        global_th_segment = pm.Deterministic("global_th_segment", global_th_segment_mu + global_th_segment_raw * global_th_segment_sigma)



        global_control =  pm.Normal("global_control", 0, 1)
        global_under = pm.Normal("global_under", 0, 1)
        global_over = pm.Normal("global_over", 0, 1)

        global_control_seg = pm.Normal("global_control_seg", 0, 1)
        global_under_seg =  pm.Normal("global_under_seg", 0, 1)
        global_over_seg = pm.Normal("global_over_seg", 0, 1)

        # level 2
        group_intercept_raw = pm.Normal("group_intercept_raw", 0, 0.5, dims="ids")
        group_th_segments_raw = pm.Normal("group_th_segments_raw", 0, 0.5, dims="ids")

        group_intercept_mu = pm.Normal("group_intercept_mu", 0, 1)
        group_intercept_sigma = pm.HalfNormal("group_intercept_sigma", 1)

        group_th_segments_mu = pm.Normal("group_th_segments_mu", 0, 1)
        group_th_segments_sigma = pm.HalfNormal("group_th_segments_sigma", 1)

        group_intercept = pm.Deterministic("group_intercept",
                                           group_intercept_mu + group_intercept_raw * group_intercept_sigma)
        group_th_segments = pm.Deterministic("group_th_segments",
                                             group_th_segments_mu + group_th_segments_raw * group_th_segments_sigma)

        growth_model = pm.Deterministic(
                "growth_model",
            pm.math.invlogit(
            (global_intercept + group_intercept[session_id_idx])
            +global_control * control
            +global_under * under
            +global_over * over
            +global_control_seg * (control * th_segments)
            +global_over_seg * (over * th_segments)
            +global_under_seg * (under * th_segments)
            +(global_th_segment + group_th_segments[session_id_idx]) * th_segments,
            )

            )

        # likelihood
        outcome = pm.Binomial("y", n=n, p=growth_model, observed=df[analyzed_features].values, dims="obs")

        # pm.model_to_graphviz(model).view()
        idata_m3 = pm.sample_prior_predictive()
        idata_m3.extend(
            pm.sample(random_seed=100, target_accept=N_TUNE, idata_kwargs={"log_likelihood": True}, draws=N_SAMPLES, chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
        )
        idata_m3.extend(pm.sample_posterior_predictive(idata_m3))



    # save the model
    with open(DOUBLE_RESULTS_PATH + "idata_m3.pkl", 'wb') as handle:
        pickle.dump(idata_m3, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # print rhat
    # az.summary(idata_m3).to_pickle(DOUBLE_RESULTS_PATH + "idata_m3_summary.pkl")

    # plot rhat
    nc_rhat = az.rhat(idata_m3)
    ax = (nc_rhat.max()
          .to_array()
          .to_series()
          .plot(kind="barh"))
    plt.savefig(DOUBLE_RESULTS_PATH + "r_hat.png")

