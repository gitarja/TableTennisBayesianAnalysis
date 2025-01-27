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
from Utils.GroupClassification import groupClassifcation

np.random.seed(1945)  # For Replicability

if __name__ == '__main__':

    rng = np.random.default_rng(seed=42)

    avg_group, ineff_group, eff_group = groupClassifcation()
    # features = [
    #     # "receiver_p1_al",
    #     #     "receiver_p2_al",
    #     #     "receiver_pursuit",
    #     #     "receiver_p1_cs",
    #     #     "receiver_p2_cs",
    #     #     "receiver_p1_al_onset",
    #     #     "receiver_p1_al_prec",
    #         "receiver_p1_al_mag",
    #         "receiver_p2_al_mag",
    #         "receiver_p2_al_onset",
    #         "receiver_p2_al_prec",
    #         "receiver_pursuit_onset",
    #         "receiver_pursuit_duration",
    #         "receiver_im_racket_dir",
    #         "receiver_im_ball_updown",
    #
    #        "receiver_start_fs",
    #         "hand_movement_sim",
    #         "receiver_fixation_racket_latency",
    #         "receiver_distance_eye_hand",
    #         "hitter_p1_al",
    #         "hitter_p1_al_onset",
    #         "hitter_p1_al_prec",
    #         "hitter_p1_al_mag",
    #         "hitter_p1_cs",
    #         "hitter_p2_al",
    #         "hitter_p2_al_onset",
    #         "hitter_p2_al_prec",
    #         "hitter_p2_al_mag",
    #         "hitter_p2_cs",
    #         "hitter_pursuit",
    #         "hitter_pursuit_onset",
    #         "hitter_pursuit_duration",
    #
    #         "hitter_at_and_after_hit",
    #         "receiver_skill"
    #             ]

    features = ["hitter_pursuit_onset"]

    # average group
    average_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                include_subjects=avg_group, exclude_failure=True,
                                                exclude_no_pair=True, hmm_probs=True)
    average_features = average_reader.getGlobalFeatures(group_label="average")

    # inefficientestimated group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=ineff_group, exclude_failure=True,
                                                    exclude_no_pair=True, hmm_probs=True)
    inefficient_features = inefficient_reader.getGlobalFeatures(group_label="inefficient")

    # efficientestimated group
    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=eff_group, exclude_failure=True,
                                                  exclude_no_pair=True, hmm_probs=True)
    efficient_features = efficient_reader.getGlobalFeatures(group_label="efficient")

    # print(inefficient_features.shape)
    # print(average_features.shape)
    # print(efficient_features.shape)
    indv = pd.concat([average_features, inefficient_features, efficient_features]).reset_index()

    # One Hot Encode Data
    dummies = pd.get_dummies(indv.group)
    # dummies.columns = ['average','inefficient','efficient']
    df = indv.join(dummies)

    # scaled average
    scaler = StandardScaler()

    average_skill_scaled = scaler.fit(df["subject_skill"].values.reshape(-1, 1))
    df["subject_skill"] = scaler.transform(df["subject_skill"].values.reshape(-1, 1))



    for f in features:
        analyzed_features = f
        clean_df = df.dropna(subset=[analyzed_features])

        subjects_idx, subjects_unique = pd.factorize(clean_df["subject"])

        average_values = clean_df.loc[clean_df["group"] == "average"][analyzed_features].values
        efficient_values = clean_df.loc[clean_df["group"] == "efficient"][analyzed_features].values
        inefficient_values = clean_df.loc[clean_df["group"] == "inefficient"][analyzed_features].values

        # average_values = average_values[~np.isnan(average_values)]
        # efficient_values = efficient_values[~np.isnan(efficient_values)]
        # inefficient_values = inefficient_values[~np.isnan(inefficient_values)]
        # observed_values = df[analyzed_features].values

        print(np.mean(inefficient_values))
        print(np.mean(average_values))
        print(np.mean(efficient_values))


        # mean and std ori
        mean = np.nanmean(clean_df[analyzed_features].values.reshape(-1, 1))
        std = np.nanstd(clean_df[analyzed_features].values.reshape(-1, 1))

        # # standarized features


        scaler = StandardScaler()
        average_scaled = scaler.fit_transform(clean_df[analyzed_features].values.reshape(-1, 1))
        clean_df[analyzed_features] = average_scaled.flatten()

        # az.plot_dist(subjects_unique)
        # plt.show()

        coords = {"subject_idx": subjects_unique, "obs": range(len(clean_df[analyzed_features])), "group": range(3)}

        # prior
        mu_m = clean_df[analyzed_features].mean()
        mu_s = clean_df[analyzed_features].std() * 2



        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            groups = pm.Data("groups", clean_df[["average", "inefficient", "efficient"]].values.astype(float), mutable=True,
                             dims=("obs", "group"))

            # skilled
            average_skill = pm.Data("average_skill",
                                        clean_df[["subject_skill", "subject_skill", "subject_skill"]].values.astype(float),
                                        mutable=True, dims=("obs", "group"))


            observed = pm.Data("observed", clean_df[analyzed_features].values, mutable=True, dims="obs")

            # subject_idx
            subject_idx = pm.Data("subject_idx", subjects_idx, mutable=True)

            # # Define priors
            # sigma_low = 0.01
            # sigma_high = 10.
            # # sigma_high = 10. ** 3
            # global_sigma = pm.Uniform("global_sigma", lower=sigma_low, upper=sigma_high, dims="group")
            # lambda1 = global_sigma ** -2
            global_sigma = pm.HalfNormal("global_sigma", sigma=1, dims="group")


            # mean
            global_mean = pm.Normal('global_mean', mu=mu_m, sigma=mu_s, dims="group")

            subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=0.1, dims=("subject_idx", "group"))

            skill_slope = pm.Normal('skill_slope', mu=0, sigma=0.1)


            average_variable = pm.Deterministic("average_variable", subjects_intercept[subject_idx] + pm.math.dot(average_skill, skill_slope))
            mu = pm.Deterministic("mu", global_mean + average_variable)

            # Define likelihood
            likelihood = pm.NormalMixture('likelihood',
                                          w=groups,
                                          mu=mu,
                                          sigma=global_sigma,
                                          observed=observed, dims="obs")


            # compute diff means



            # compute diference of mean and std, and effect size
            # average: 0
            # inefficient: 1
            # efficient: 2

            diff_means_efficient_average = pm.Deterministic('diff_means_efficient_average',
                                                            global_mean[2] - global_mean[0])
            diff_means_inefficient_average = pm.Deterministic('diff_means_inefficient_average',
                                                            global_mean[1] - global_mean[0])
            diff_means_efficient_inefficient = pm.Deterministic('diff_means_efficient_inefficient',
                                                            global_mean[2] - global_mean[1])

            diff_sigma_efficient_average = pm.Deterministic('diff_sigma_efficient_average',
                                                            global_sigma[2] - global_sigma[0])
            diff_sigma_inefficient_average = pm.Deterministic('diff_sigma_inefficient_average',
                                                              global_sigma[1] - global_sigma[0])
            diff_sigma_efficient_inefficient = pm.Deterministic('diff_sigma_efficient_inefficient',
                                                                global_sigma[2] - global_sigma[1])

            # compute effect size
            effect_efficient_average = pm.Deterministic(
                "effect_efficient_average",
                diff_means_efficient_average / np.sqrt((global_sigma[2] ** 2 + global_sigma[0] ** 2) / 2)
            )

            effect_efficient_inefficient = pm.Deterministic(
                "effect_efficient_inefficient",
                diff_means_efficient_inefficient / np.sqrt((global_sigma[2] ** 2 + global_sigma[1] ** 2) / 2)
            )

            effect_inefficient_average = pm.Deterministic(
                "effect_inefficient_average",
                diff_means_inefficient_average / np.sqrt((global_sigma[1] ** 2 + global_sigma[0] ** 2) / 2)
            )

        # debug and sampling
        with model:
            # debug the model
            print(model.debug())
            # pm.model_to_graphviz(model).view()
            # Inference!
            idata_m3 = pm.sample_prior_predictive()
            idata_m3.extend(
                pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                          draws=N_SAMPLES,
                          chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
            )
            idata_m3.extend(pm.sample_posterior_predictive(idata_m3))

        # print loo
        hierarchical_loo = az.loo(idata_m3)

        print(hierarchical_loo)
        trace_post = az.extract(idata_m3.posterior)
        print(trace_post['global_mean'].data.shape)
        # Get posterior samples for the parameter of interest
        print("Features: " + analyzed_features)
        average_posterior_samples = np.concatenate([trace_post['global_mean'].data[0].flatten()])
        average_credible_interval = np.percentile(average_posterior_samples, [5, 95.0])

        inefficient_posterior_samples = np.concatenate([trace_post['global_mean'].data[1].flatten()])
        inefficient_credible_interval = np.percentile(inefficient_posterior_samples, [5, 95.0])

        efficient_posterior_samples = np.concatenate([trace_post['global_mean'].data[2].flatten()])
        efficient_credible_interval = np.percentile(efficient_posterior_samples, [5, 95.0])

        print("average Credible Interval (95%):", average_credible_interval)
        print("efficient Credible Interval (95%):", efficient_credible_interval)
        print("inefficient Credible Interval (95%):", inefficient_credible_interval)

        alpha = 0.05
        l = len(trace_post['global_mean'].data[0].flatten())
        low_bound = int(alpha / 2 * l)
        high_bound = int((1 - (alpha / 2)) * l)

        fig, ax = plt.subplots(figsize=(12, 8))
        for idx, group, color in zip([0, 1, 2], ['average_mean', 'inefficient_mean', 'efficient_mean'],
                                     ['#377eb8', '#e41a1c', '#4daf4a']):
            data = (trace_post["global_mean"].data[idx].flatten() * std) + mean
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
        plt.legend()
        plt.xlabel(analyzed_features)

        # plt.show()
        plt.savefig(DOUBLE_RESULTS_PATH_ANOVA + "posterior\\" + analyzed_features + ".eps", format="eps")
        plt.savefig(DOUBLE_RESULTS_PATH_ANOVA + "posterior\\" + analyzed_features + ".png", format="png")
        plt.close()

        # save the model
        # with open(DOUBLE_RESULTS_PATH_ANOVA + "model\\" + "idata_m3_" + analyzed_features + ".pkl", 'wb') as handle:
        #     print("write data into: " + "idata_m3_" + analyzed_features + ".pkl")
        #     pickle.dump(idata_m3, handle, protocol=pickle.HIGHEST_PROTOCOL)
