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




    inefficient_features = inefficient_reader.getStableUnstableFailureFeatures(group_name="inefficient",
                                                                               success_failure=True,
                                                                               mod="skill_personal_perception_action_impact",
                                                                               with_control=True)
    inefficient_features["group"] = "inefficient"
    # efficient group

    efficient_features = efficient_reader.getStableUnstableFailureFeatures(group_name="efficient", success_failure=True,
                                                                           mod="skill_personal_perception_action_impact",
                                                                           with_control=True)
    efficient_features["group"] = "efficient"

    df = pd.concat([inefficient_features, efficient_features])

    # features = [
    #
    #     "receiver_im_ball_updown",
    #
    #     "receiver_im_ball_wrist",
    #     "receiver_start_fs",
    #     "receiver_im_racket_ball_wrist",
    #     "hitter_p1_al_mag",
    #     "receiver_im_racket_dir",
    #     "receiver_fixation_racket_latency",
    #     "hitter_p2_al_prec",
    #     "hitter_p2_al_mag",
    #     "receiver_p2_al_mag",
    #     "hitter_at_and_after_hit",
    #     "hitter_p1_cs",
    #     "hitter_p2_al_onset",
    #     "hand_movement_sim",
    #     "receiver_p1_al_onset",
    #     "hitter_p1_al_prec",
    #     "receiver_p1_al_mag",
    #
    #     "receiver_p2_al_prec",
    #     "receiver_p2_al_onset",
    #     "hitter_fx_onset",
    #     "receiver_distance_eye_hand",
    #     "hitter_p1_al_onset",
    #     "receiver_p1_al_prec",
    #     "hitter_fx_duration",
    #     "receiver_p3_fx_onset",
    #     "height_sim",
    #     "age_sim",
    #     "relationship",
    #     "receiver_p1_cs",
    #     "receiver_p3_fx_duration",
    #     "gender_sim",
    #     "receiver_p3_fx",
    #     "hitter_fx",
    #     "hitter_p2_al",
    #     "receiver_p2_al",
    #     "receiver_p2_cs",
    #     "hitter_p2_cs",
    #     "receiver_p1_al",
    #     "hitter_p1_al",
    #
    # ]

    # features = ["hitter_p1_cs", "hitter_p2_cs", "receiver_p1_cs", "receiver_p2_cs"]
    # features = ["individual_skill", "individual_skill_sim" ]
    features = ["relationship", "height_sim"]

    efficient_mean_list = []
    inefficient_mean_list = []
    efficient_hdi_list = []
    inefficient_hdi_list = []
    effect_size_mean_list = []
    effect_size_hdi_list = []
    features_list = []

    for f in features:

        print(f)
        analyzed_features = f
        clean_df = df.dropna(subset=[analyzed_features])

        # mean and std ori
        mean_ori = np.nanmean(clean_df[analyzed_features].values.reshape(-1, 1))
        std_ori = np.nanstd(clean_df[analyzed_features].values.reshape(-1, 1))

        # # standarized features
        scaler = StandardScaler()
        average_scaled = scaler.fit_transform(clean_df[analyzed_features].values.reshape(-1, 1))
        clean_df[analyzed_features] = average_scaled.flatten()

        #
        # az.plot_dist(clean_df[analyzed_features], rug=True)
        # plt.show()

        # factorize receiver
        ineff_subjects_idx, ineff_subjects_unique = pd.factorize(
            clean_df.loc[clean_df["group"] == "inefficient"]["receiver"].values)
        eff_subjects_idx, eff_subjects_unique = pd.factorize(
            clean_df.loc[clean_df["group"] == "efficient"]["receiver"].values)

        # ineff_subjects_idx, ineff_subjects_unique = pd.factorize(
        #     clean_df.loc[clean_df["group"] == "inefficient"].groupby("session")["receiver"].last())
        # eff_subjects_idx, eff_subjects_unique = pd.factorize(
        #     clean_df.loc[clean_df["group"] == "efficient"].groupby("session")["receiver"].last())

        coords = {"ineff_subject_idx": ineff_subjects_unique, "eff_subject_idx": eff_subjects_unique, "components": range(2)}

        mu_m = clean_df[analyzed_features].mean()
        mu_s = clean_df[analyzed_features].std() * 2

        inefficient_obv = clean_df.loc[clean_df["group"] == "inefficient"][analyzed_features].values
        efficient_obv = clean_df.loc[clean_df["group"] == "efficient"][analyzed_features].values

        # inefficient_obv = clean_df.loc[clean_df["group"] == "inefficient"].groupby("session")[analyzed_features].mean().values
        # efficient_obv = clean_df.loc[clean_df["group"] == "efficient"].groupby("session")[analyzed_features].mean().values

        print(np.average(inefficient_obv))
        print(np.average(efficient_obv))

        print(np.std(inefficient_obv))
        print(np.std(efficient_obv))

        sigma_low = 10 ** -1
        sigma_high = 10
        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            # number
            #
            # ineff_subjects_intercept = pm.Normal("ineff_subjects_intercept", 0, 0.1, dims="ineff_subject_idx")
            # eff_subjects_intercept = pm.Normal("eff_subjects_intercept", 0, 0.1, dims="eff_subject_idx")
            # inefficient_mean = pm.TruncatedNormal('inefficient_mean', mu=mu_m, sigma=mu_s, lower=0)
            # efficient_mean = pm.TruncatedNormal('efficient_mean', mu=mu_m, sigma=mu_s, lower=0)
            # inefficient_std = inefficient_mean
            # efficient_std = efficient_mean
            #
            # inefficient = pm.Poisson("inefficient",
            #                          mu=pm.math.invlogit(inefficient_mean + ineff_subjects_intercept[ineff_subjects_idx]),
            #                          observed=inefficient_obv)
            # efficient = pm.Poisson("efficient",
            #                        mu=pm.math.invlogit(efficient_mean + eff_subjects_intercept[eff_subjects_idx]),
            #                        observed=efficient_obv)

            # continous

            # random intercept
            ineff_subjects_intercept = pm.Normal("ineff_subjects_intercept", 0, 0.1, dims="ineff_subject_idx")
            eff_subjects_intercept = pm.Normal("eff_subjects_intercept", 0, 0.1, dims="eff_subject_idx")
            inefficient_mean = pm.Normal('inefficient_mean', mu=mu_m, sigma=mu_s)
            efficient_mean = pm.Normal('efficient_mean', mu=mu_m, sigma=mu_s)

            inefficient_std = pm.Uniform("inefficient_std", lower=sigma_low, upper=sigma_high)
            efficient_std = pm.Uniform("efficient_std", lower=sigma_low, upper=sigma_high)

            nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            nu = pm.Deterministic("nu", nu_minus_one + 1)
            nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

            lambda_1 = efficient_std ** -2
            lambda_2 = inefficient_std ** -2
            inefficient = pm.StudentT("inefficient", nu=nu,
                                      mu=inefficient_mean + ineff_subjects_intercept[ineff_subjects_idx], lam=lambda_2,
                                      observed=inefficient_obv)
            efficient = pm.StudentT("efficient", nu=nu, mu=efficient_mean + eff_subjects_intercept[eff_subjects_idx],
                                    lam=lambda_1, observed=efficient_obv)

            # mixture
            # ineff_subjects_intercept = pm.Normal("ineff_subjects_intercept", 0, 0.01, dims=("ineff_subject_idx", "components"))
            # eff_subjects_intercept = pm.Normal("eff_subjects_intercept", 0, 0.01, dims=("ineff_subject_idx", "components"))
            # inefficient_mean = pm.Normal('inefficient_mean', mu=mu_m, sigma=0.5, shape=2)
            # efficient_mean = pm.Normal('efficient_mean', mu=mu_m, sigma=0.5, shape=2)
            #
            # inefficient_std = pm.Uniform("inefficient_std", lower=sigma_low, upper=sigma_high, shape=2)
            # efficient_std = pm.Uniform("efficient_std", lower=sigma_low, upper=sigma_high, shape=2)
            # weights = pm.Dirichlet("w", np.ones(2))
            #
            # inefficient = pm.NormalMixture("inefficient", w=weights, mu=inefficient_mean + ineff_subjects_intercept[ineff_subjects_idx], sigma=inefficient_std,
            #                            observed=inefficient_obv)
            # efficient = pm.NormalMixture("efficient", w=weights,
            #                                mu=efficient_mean + ineff_subjects_intercept[ineff_subjects_idx],
            #                                sigma=inefficient_std,
            #                                observed=inefficient_obv)


            diff_of_means = pm.Deterministic("difference_of_means", efficient_mean - inefficient_mean)
            diff_of_stds = pm.Deterministic("difference_of_stds", efficient_std - inefficient_std)
            effect_size = pm.Deterministic(
                "effect_size", diff_of_means / np.sqrt((inefficient_std ** 2 + efficient_std ** 2) / 2)
            )

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

        # print(az.summary(idata, round_to=2))
        # plt.show()
        # az.plot_posterior(
        #     idata,
        #     var_names=["difference_of_means", "difference_of_stds", "efficient_std", "inefficient_std", "effect_size"],
        #     color="#87ceeb",
        # )
        # plt.show()
        # print loo
        hierarchical_loo = az.plot_ppc(idata, kind='cumulative')
        plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "PPC\\" + analyzed_features + ".png", format='png')
        plt.close()
        # plt.show()
        # print(hierarchical_loo)

        # az.plot_posterior(idata, var_names=["effect_size"], ref_val=0)
        # plt.xlabel(analyzed_features)
        # # plt.show()
        # plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "EffectSize\\" + analyzed_features + ".pdf", format='pdf')
        # plt.close()

        trace_post = az.extract(idata.posterior)

        # compute mean and HDI 95
        efficient_mean_data = (trace_post["efficient_mean"].data * std_ori) + mean_ori
        inefficient_mean_data = (trace_post["inefficient_mean"].data * std_ori) + mean_ori

        # efficient_mean_data = (trace_post["efficient_mean"].data)
        # inefficient_mean_data = (trace_post["inefficient_mean"].data)
        effect_size_data = trace_post["effect_size"].data
        efficient_avg = np.mean(efficient_mean_data)
        inefficient_avg = np.mean(inefficient_mean_data)
        efficient_hdi = az.hdi(efficient_mean_data, hdi_prob=.95)
        inefficient_hdi = az.hdi(inefficient_mean_data, hdi_prob=.95)

        effect_size_mean = np.mean(effect_size_data)
        effect_size_hdi = az.hdi(effect_size_data, hdi_prob=.95)

        efficient_mean_list.append(efficient_avg)
        inefficient_mean_list.append(inefficient_avg)
        efficient_hdi_list.append(efficient_hdi)
        inefficient_hdi_list.append(inefficient_hdi)
        effect_size_mean_list.append(effect_size_mean)
        effect_size_hdi_list.append(effect_size_hdi)
        features_list.append(f)

        alpha = 0.05
        l = len(trace_post['efficient_mean'].data.flatten())
        low_bound = int(alpha / 2 * l)
        high_bound = int((1 - (alpha / 2)) * l)

        fig, ax = plt.subplots(figsize=(12, 8))
        for group, color in zip(['efficient_mean', 'inefficient_mean'],
                                ['#69a87f', '#b5152c']):
            data = (trace_post[group].data.flatten() * std_ori) + mean_ori
            # data = (trace_post[group].data.flatten())
            # Estimate KDE
            kde = stats.gaussian_kde(data)
            # plot complete kde curve as line
            pos = np.linspace(np.min(data), np.max(data), 101)
            plt.plot(pos, kde(pos), color=color, label='{0} KDE'.format(group))
            # Set shading bounds
            low = np.sort(data)[low_bound]
            high = np.sort(data)[high_bound]
            # plot shaded kde
            shade = np.linspace(low, high, 101)
            plt.fill_between(shade, kde(shade), alpha=0.3, color=color, label="{0} 95% HPD Score".format(group))
        # plt.legend()
        plt.xlabel(features_explanation[analyzed_features])

        # plt.show()

        # plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "MU\\" + analyzed_features + ".pdf", format='pdf')
        # plt.close()

        # save the model
        with open(DOUBLE_RESULTS_PATH_TTEST + "model\\" + "idata_" + analyzed_features + ".pkl", 'wb') as handle:
            print("write data into: " + "idata_ttest_" + analyzed_features + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        summary_df = pd.DataFrame(
            {"features": features_list, "efficient_mean": efficient_mean_list, "efficient_hdi": efficient_hdi_list,
             "inefficient_mean": inefficient_mean_list, "inefficient_hdi": inefficient_hdi_list,
             "effect_size_mean": effect_size_mean_list, "effect_size_hdi": effect_size_hdi_list})

        summary_df.to_csv(DOUBLE_RESULTS_PATH_TTEST + "summary_prec.csv")

        del model
        del idata
