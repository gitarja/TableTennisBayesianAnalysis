import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, features_explanation
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST
import arviz as az
import pickle
import scipy.stats as stats
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

    inefficient_features = inefficient_reader.getSingleDoubleFeatures()
    inefficient_features["group"] = "inefficient"
    # efficient group

    efficient_features = efficient_reader.getSingleDoubleFeatures()
    efficient_features["group"] = "efficient"

    df = pd.concat([inefficient_features, efficient_features])

    features = [

        "v_hitter_p1_cs",
        "v_hitter_p1_al_onset",
        "v_hitter_p1_al_prec",
        "v_hitter_p1_al_mag",
        "v_receiver_p1_al_onset",
        "v_receiver_p1_al_prec",
        "v_receiver_p1_al_mag",
        "v_receiver_p1_cs",
        "v_start_fs",
        "v_bounce_point",
        "v_spatial_use",
        "v_hitter_p2_al_onset",
        "v_hitter_p2_al_prec",
        "v_hitter_p2_al_mag",
        "v_hitter_p2_cs",
        "v_hitter_p3_fx_onset",
        "v_hitter_p3_fx_duration",
        "v_receiver_p2_al_onset",
        "v_receiver_p2_al_prec",
        "v_receiver_p2_al_mag",
        "v_receiver_p2_cs",
        "v_receiver_p3_fx_onset",
        "v_receiver_p3_fx_duration",
        "v_fixation_racket_latency",
        "v_distance_eye_han",
        "v_im_ball_wrist",
        "v_im_racket_ball_wrist",
        "v_im_racket_ball_angle",
        "v_im_ball_updown",
        "lfhf_sim"

    ]
    for f in features:
        print(f)
        analyzed_features = f
        clean_df = df.dropna(subset=[analyzed_features])

        # mean and std ori
        mean_ori = np.nanmean(clean_df[analyzed_features].values.reshape(-1, 1))
        std_ori = np.nanstd(clean_df[analyzed_features].values.reshape(-1, 1))

        # standarized features (do not do it for numbers)
        scaler = StandardScaler()
        average_scaled = scaler.fit_transform(clean_df[analyzed_features].values.reshape(-1, 1))
        clean_df[analyzed_features] = average_scaled.flatten()

        # az.plot_dist(clean_df[analyzed_features].values)
        # plt.show()

        subjects_idx = np.unique(np.concatenate([clean_df["id_subject"].values, clean_df["id_partner"].values]))
        # factorize subjects
        ineff_subjects_idx = np.searchsorted(subjects_idx,
                                               clean_df.loc[clean_df["group"] == "inefficient"]["id_subject"].values)
        eff_subjects_idx = np.searchsorted(subjects_idx,
                                               clean_df.loc[clean_df["group"] == "efficient"]["id_subject"].values)



        # factorize partner
        ineff_partners_idx = np.searchsorted(subjects_idx,
                                               clean_df.loc[clean_df["group"] == "inefficient"]["id_partner"].values)


        eff_partners_idx =np.searchsorted(subjects_idx,
                                               clean_df.loc[clean_df["group"] == "efficient"]["id_partner"].values)




        mu_m = clean_df[analyzed_features].mean()
        mu_s = clean_df[analyzed_features].std() * 2

        # for others predictor
        inefficient_obv = clean_df.loc[clean_df["group"] == "inefficient"][analyzed_features].values
        efficient_obv = clean_df.loc[clean_df["group"] == "efficient"][analyzed_features].values

        coords = {
            # subjects id
            "subject_idx": subjects_idx,

                  "components": range(2)}


        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            # continous

            # random intercept
            subjects_subject_intercept = pm.Normal("subjects_subject_intercept", 0, 0.1, dims="subject_idx")




            inefficient_mean = pm.Normal('inefficient_mean', mu=mu_m, sigma=1)
            efficient_mean = pm.Normal('efficient_mean', mu=mu_m, sigma=1)

            inefficient_std = pm.HalfCauchy("inefficient_std", 1.0)
            efficient_std = pm.HalfCauchy("efficient_std", 1.0)
            inefficient_latent = pm.Deterministic("iefficient_latent", inefficient_mean
                                                  + (
                                                          subjects_subject_intercept[ineff_subjects_idx] +
                                                          subjects_subject_intercept[ineff_partners_idx]) / 2)

            efficient_latent = pm.Deterministic("efficient_latent", efficient_mean
                                                + (
                                                        subjects_subject_intercept[eff_subjects_idx] +
                                                        subjects_subject_intercept[eff_partners_idx]) / 2)

            nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            nu = pm.Deterministic("nu", nu_minus_one + 1)
            nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

            lambda_1 = efficient_std ** -2
            lambda_2 = inefficient_std ** -2
            inefficient = pm.StudentT("inefficient", nu=nu,
                                      mu=inefficient_latent, lam=lambda_2,
                                      observed=inefficient_obv)
            efficient = pm.StudentT("efficient", nu=nu, mu=efficient_latent,
                                    lam=lambda_1, observed=efficient_obv)

            # means difference and others
            diff_of_means = pm.Deterministic("difference_of_means", efficient_mean - inefficient_mean)
            diff_of_stds = pm.Deterministic("difference_of_stds", efficient_std - inefficient_std)
            effect_size = pm.Deterministic(
                "effect_size", diff_of_means / np.sqrt((inefficient_std ** 2 + efficient_std ** 2) / 2)
            )

            diff_of_eff_subs = pm.Deterministic("diff_of_eff_subs", subjects_subject_intercept[eff_subjects_idx] - subjects_subject_intercept[eff_partners_idx])
            diff_of_ineff_subs = pm.Deterministic("diff_of_ineff_subs", subjects_subject_intercept[ineff_subjects_idx] -
                                                subjects_subject_intercept[ineff_partners_idx])

            # debug and sampling
        with model:
            # debug the model
            print(model.debug())
            # pm.model_to_graphviz(model).view()
            # Inference!
            idata = pm.sample_prior_predictive()
            idata.extend(
                pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                          draws=N_SAMPLES,
                          chains=N_CHAINS, tune=N_TUNE, cores=N_CORE, compile_kwargs=dict(mode="NUMBA"))
            )
            idata.extend(pm.sample_posterior_predictive(idata))

        # print loo
        hierarchical_loo = az.plot_ppc(idata)
        plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "PPC_supp\\" + analyzed_features + "_mean.png", format='png')
        plt.close()

        trace_post = az.extract(idata.posterior)

        # save the model
        with open(DOUBLE_RESULTS_PATH_TTEST + "model_supp\\" + "idata_" + analyzed_features + "_mean.pkl", 'wb') as handle:
            print("write data into: " + "idata_ttest_" + analyzed_features + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del model
        del idata
