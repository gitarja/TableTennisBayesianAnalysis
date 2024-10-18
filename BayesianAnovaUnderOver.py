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

    features = ["stable_percentage"]

    # control group
    control_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                include_subjects=avg_group, exclude_failure=False,
                                                exclude_no_pair=True, hmm_probs=True)
    control_features = control_reader.getGlobalFeatures(group_label="control")

    # inefficientestimated group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=ineff_group, exclude_failure=False,
                                                    exclude_no_pair=True, hmm_probs=True)
    inefficient_features = inefficient_reader.getGlobalFeatures(group_label="inefficient")

    # efficientestimated group
    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=eff_group, exclude_failure=False,
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


    for f in features:
        analyzed_features = f

        control_values = df.loc[df["group"] == "control"][analyzed_features].values
        efficient_values = df.loc[df["group"] == "efficient"][analyzed_features].values
        inefficient_values = df.loc[df["group"] == "inefficient"][analyzed_features].values

        print(np.mean(control_values))
        print(np.mean(inefficient_values))
        print(np.mean(efficient_values))


        # mean and std ori
        mean = np.nanmean(df[analyzed_features].values.reshape(-1, 1))
        std = np.nanstd(df[analyzed_features].values.reshape(-1, 1))

        # # standarized features
        scaler = StandardScaler()
        control_scaled = scaler.fit_transform(df[analyzed_features].values.reshape(-1, 1))
        df[analyzed_features] = control_scaled.flatten()

        # az.plot_dist(subjects_unique)
        # plt.show()

        coords = {"subject_idx": subjects_unique, "obs": range(len(df[analyzed_features])), "group": range(3)}

        # prior
        mu_m = df[analyzed_features].mean()
        mu_s = df[analyzed_features].std() * 2



        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            groups = pm.Data("groups", df[["control", "inefficient", "efficient"]].values.astype(float), mutable=True,
                             dims=("obs", "group"))

            # skilled
            controlled_skill = pm.Data("controlled_skill",
                                        df[["subject_skill", "subject_skill", "subject_skill"]].values.astype(float),
                                        mutable=True, dims=("obs", "group"))

            observed = pm.Data("observed", df[analyzed_features].values, mutable=True, dims="obs")

            # subject_idx
            subject_idx = pm.Data("subject_idx", subjects_idx, mutable=True)

            # Define priors
            sigma_low = 0.01
            # sigma_high = 10.
            sigma_high = 10. ** 3
            global_sigma = pm.Uniform("global_sigma", lower=sigma_low, upper=sigma_high, dims="group")

            # mean
            global_mean = pm.Normal('global_mean', mu=mu_m, sigma=mu_s, dims="group")

            subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=0.1, dims=("subject_idx", "group"))

            skill_slope = pm.Normal('skill_slope', mu=0, sigma=0.1)


            control_variable = pm.Deterministic("control_variable",  (subjects_intercept[subject_idx]) + pm.math.dot(controlled_skill, skill_slope))
            mu = pm.Deterministic("mu", global_mean + control_variable)

            # Define likelihood
            likelihood = pm.NormalMixture('likelihood',
                                          w=groups,
                                          mu=mu,
                                          sigma=global_sigma,
                                          observed=observed, dims="obs")

            # compute diff means



            # compute diference of mean and std, and effect size
            # control: 0
            # inefficient: 1
            # efficient: 2

            diff_means_efficient_control = pm.Deterministic('diff_means_efficient_control',
                                                            global_mean[2] - global_mean[0])
            diff_means_inefficient_control = pm.Deterministic('diff_means_inefficient_control',
                                                            global_mean[1] - global_mean[0])
            diff_means_efficient_inefficient = pm.Deterministic('diff_means_efficient_inefficient',
                                                            global_mean[2] - global_mean[1])

            diff_sigma_efficient_control = pm.Deterministic('diff_sigma_efficient_control',
                                                            global_sigma[2] - global_sigma[0])
            diff_sigma_inefficient_control = pm.Deterministic('diff_sigma_inefficient_control',
                                                              global_sigma[1] - global_sigma[0])
            diff_sigma_efficient_inefficient = pm.Deterministic('diff_sigma_efficient_inefficient',
                                                                global_sigma[2] - global_sigma[1])

            # compute effect size
            effect_efficient_control = pm.Deterministic(
                "effect_efficient_control",
                diff_means_efficient_control / np.sqrt((global_sigma[2] ** 2 + global_sigma[0] ** 2) / 2)
            )

            effect_efficient_inefficient = pm.Deterministic(
                "effect_efficient_inefficient",
                diff_means_efficient_inefficient / np.sqrt((global_sigma[2] ** 2 + global_sigma[1] ** 2) / 2)
            )

            effect_inefficient_control = pm.Deterministic(
                "effect_inefficient_control",
                diff_means_inefficient_control / np.sqrt((global_sigma[1] ** 2 + global_sigma[0] ** 2) / 2)
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
                          draws=3000,
                          chains=N_CHAINS, tune=3000, cores=N_CORE)
            )
            idata_m3.extend(pm.sample_posterior_predictive(idata_m3))

        # print loo
        hierarchical_loo = az.loo(idata_m3)

        print(hierarchical_loo)
        trace_post = az.extract(idata_m3.posterior)
        print(trace_post['global_mean'].data.shape)
        # Get posterior samples for the parameter of interest
        print("Features: " + analyzed_features)
        control_posterior_samples = np.concatenate([trace_post['global_mean'].data[0].flatten()])
        control_credible_interval = np.percentile(control_posterior_samples, [5, 95.0])

        inefficient_posterior_samples = np.concatenate([trace_post['global_mean'].data[1].flatten()])
        inefficient_credible_interval = np.percentile(inefficient_posterior_samples, [5, 95.0])

        efficient_posterior_samples = np.concatenate([trace_post['global_mean'].data[2].flatten()])
        efficient_credible_interval = np.percentile(efficient_posterior_samples, [5, 95.0])

        print("Control Credible Interval (95%):", control_credible_interval)
        print("efficient Credible Interval (95%):", efficient_credible_interval)
        print("inefficient Credible Interval (95%):", inefficient_credible_interval)

        alpha = 0.05
        l = len(trace_post['global_mean'].data[0].flatten())
        low_bound = int(alpha / 2 * l)
        high_bound = int((1 - (alpha / 2)) * l)

        fig, ax = plt.subplots(figsize=(12, 8))
        for idx, group, color in zip([0, 1, 2], ['control_mean', 'inefficient_mean', 'efficient_mean'],
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
        with open(DOUBLE_RESULTS_PATH_ANOVA + "model\\" + "idata_m3_" + analyzed_features + ".pkl", 'wb') as handle:
            print("write data into: " + "idata_m3_" + analyzed_features + ".pkl")
            pickle.dump(idata_m3, handle, protocol=pickle.HIGHEST_PROTOCOL)
