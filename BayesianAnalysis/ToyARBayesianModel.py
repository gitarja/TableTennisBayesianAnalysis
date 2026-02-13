import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, features_explanation
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FE_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST
import arviz as az
import pickle
from sklearn.preprocessing import StandardScaler
from BayesianRegressionModels import BayesianHirSegmentModel
from sklearn.preprocessing import StandardScaler

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20

if __name__ == '__main__':
    inefficient_group, efficient_group = groupLabeling()

    # inefficient group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=inefficient_group, exclude_failure=True,
                                                    exclude_no_pair=False, hmm_probs=True)

    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=efficient_group, exclude_failure=True,
                                                  exclude_no_pair=False, hmm_probs=True)

    inefficient_features = inefficient_reader.getFEFeatures()
    inefficient_features["group"] = "inefficient"
    # efficient group

    efficient_features = efficient_reader.getFEFeatures()
    efficient_features["group"] = "efficient"

    df = pd.concat([inefficient_features, efficient_features])

    clean_df = df.dropna()

    subjects_idx =  np.unique(clean_df["hitter"].values)
    sessions_idx = np.unique(clean_df["session"].values)
    group_idx = np.unique(clean_df["group"].values)
    event_idx = np.unique(clean_df["event_seg"].values)


    prior_mean_group = clean_df.groupby("hitter")["priors_mean"].mean().reset_index()
    priors_mean = prior_mean_group.loc[np.argwhere(prior_mean_group["hitter"].values == subjects_idx).flatten()]["priors_mean"].values

    prior_std_group = clean_df.groupby("hitter")["priors_std"].mean().reset_index()
    priors_std = prior_std_group.loc[np.argwhere(prior_std_group["hitter"].values == subjects_idx).flatten()]["priors_std"].values

    # normalize the predictors
    session_v_idx = np.searchsorted(sessions_idx, clean_df["session"].values)
    actor_v_idx = np.searchsorted(subjects_idx, clean_df["hitter"].values)
    partner_v_idx = np.searchsorted(subjects_idx, clean_df["receiver"].values)
    group_v_idx = np.searchsorted(group_idx, clean_df["group"].values)
    event_v_idx =  np.searchsorted(event_idx, clean_df["event_seg"].values)


    y_f = clean_df["post_visual_angle_error"].values
    X = clean_df[["sense_p1_visual_angle_error",
                  "sense_p2_visual_angle_error",
                  "sense_p3_pursuit_duration",
                  "sense_swing_onset",
                  "sense_distance_eye_hand",
                  "sense_racket_ball_angle",
                  "sense_racket_ball_wrist",
                  "sense_ball_updown"]].values

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    y_f_std = scaler.fit_transform(y_f.reshape((-1, 1))).flatten()

    # az.plot_dist(x)
    # plt.show()
    coords = {
        # subjects id
        "obs": range(len(y_f)),
        "subject_idx": subjects_idx,
        "session_idx": sessions_idx,
        "group_idx": group_idx,
        "axis": range(8),
        "event_idx": event_idx,
        "feature": range(4),

    }

    type_model = "ARFE"
    model = BayesianHirSegmentModel(coords, X_std, y_f_std, priors_mean, priors_std,  session_v_idx, actor_v_idx, partner_v_idx, group_v_idx, event_v_idx)

    with model:
        # debug the model
        # pm.model_to_graphviz(model).view()
        print(model.debug())

        # Inference!
        idata = pm.sample_prior_predictive()
        idata.extend(
            pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                      draws=N_SAMPLES,
                      chains=N_CHAINS, tune=N_TUNE, cores=N_CORE, compile_kwargs=dict(mode="NUMBA"),
                      init="jitter+adapt_diag")
        )
        idata.extend(pm.sample_posterior_predictive(idata))

    # print loo
    hierarchical_loo = az.plot_ppc(idata)
    plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "PPC_regression\\minimize_"+type_model+".png", format='png')
    plt.close()

    trace_post = az.extract(idata.posterior)
    # print(az.summary(idata))
    # save the model
    with open(DOUBLE_RESULTS_PATH_TTEST + "model_regression\\" + "idata_"+type_model+".pkl", 'wb') as handle:
        pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)
