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

    prefix = ["stable_TO_stable", "unstable_TO_stable"]

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

    # features = ["recover_pr_p1_al", "recover_pr_p1_al_onset", "recover_pr_p1_cs", "recover_pr_p1_al_prec",
    #             "recover_pr_p2_al", "recover_pr_p2_al_onset", "recover_pr_p2_cs", "recover_pr_p2_al_prec",
    #             "recover_hitter_pursuit", "recover_hitter_pursuit_duration",
    #             "recover_movement_sim", "recover_bouncing_point_var_p1"]

    features = ["recover_hitter_pursuit_duration"]

    # control group
    control_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                include_subjects=inlier_group, exclude_failure=True,
                                                exclude_no_pair=True, hmm_probs=True)
    control_features = control_reader.getGlobalStableUnstableFeatures(group_label="control", prefix=prefix)

    # inefficientestimated group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=inefficient_group, exclude_failure=True,
                                                    exclude_no_pair=True, hmm_probs=True)
    inefficient_features = inefficient_reader.getGlobalStableUnstableFeatures(group_label="inefficient", prefix=prefix)

    # efficientestimated group
    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=efficient_group, exclude_failure=True,
                                                  exclude_no_pair=True, hmm_probs=True)
    efficient_features = efficient_reader.getGlobalStableUnstableFeatures(group_label="efficient", prefix=prefix)

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



        control_stable_values = df.loc[df["group"] == "control_" + prefix[0]][analyzed_features].values
        control_unstable_values = df.loc[df["group"] == "control_" + prefix[1]][analyzed_features].values
        efficient_stable_values = df.loc[df["group"] == "efficient_" + prefix[0]][analyzed_features].values
        efficient_unstable_values = df.loc[df["group"] == "efficient_" + prefix[1]][analyzed_features].values
        inefficient_stable_values = df.loc[df["group"] == "inefficient_" + prefix[0]][analyzed_features].values
        inefficient_unstable_values = df.loc[df["group"] == "inefficient_" + prefix[1]][analyzed_features].values

        print(np.nanmean(control_stable_values))
        print(np.nanmean(control_unstable_values))
        print(np.nanmean(inefficient_stable_values))
        print(np.nanmean(inefficient_unstable_values))
        print(np.nanmean(efficient_stable_values))
        print(np.nanmean(efficient_unstable_values))

        # mean and std ori
        mean = np.nanmean(df[analyzed_features].values.reshape(-1, 1))
        std = np.nanstd(df[analyzed_features].values.reshape(-1, 1))



        # plot features
        # az.plot_dist(df[analyzed_features].values)
        # plt.show()

        # # standarized features
        scaler = StandardScaler()
        control_scaled = scaler.fit_transform(df[analyzed_features].values.reshape(-1, 1))
        df[analyzed_features] = control_scaled.flatten()

        # az.plot_dist(subjects_unique)
        # plt.show()

        coords = {"subject_idx": subjects_unique, "obs": range(len(df[analyzed_features])), "group": range(6)}

        # prior
        mu_m = df[analyzed_features].mean()
        mu_s = df[analyzed_features].std() * 2

        df[analyzed_features].fillna(mu_m, inplace=True)

        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            groups = pm.Data("groups", df[
                ["control_"+prefix[0], "control_"+prefix[1], "inefficient_"+prefix[0], "inefficient_"+prefix[1], "efficient_"+prefix[0],
                 "efficient_"+prefix[1]]].values.astype(float), mutable=True,
                             dims=("obs", "group"))

            # skilled
            controlled_skill = pm.Data("controlled_skill",
                                       df[["subject_skill", "subject_skill", "subject_skill", "subject_skill",
                                           "subject_skill", "subject_skill"]].values.astype(float),
                                       mutable=True, dims=("obs", "group"))

            observed = pm.Data("observed", df[analyzed_features].values, mutable=True, dims="obs")

            # subject_idx
            subject_idx = pm.Data("subject_idx", subjects_idx, mutable=True)

            # Define priors
            # sigma_low = 0.8
            # sigma_high = 1.
            sigma_low = 0.01
            sigma_high = 10. ** 3
            global_sigma = pm.Uniform("global_sigma", lower=sigma_low, upper=sigma_high, dims="group")
            # global_sigma = pm.HalfNormal("global_sigma", 10, dims="group")

            # mean
            global_mean = pm.Normal('global_mean', mu=mu_m, sigma=mu_s, dims="group")

            subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=0.1, dims=("subject_idx", "group"))

            skill_slope = pm.Normal('skill_slope', mu=0, sigma=0.1)

            control_variable = pm.Deterministic("control_variable",
                                                (subjects_intercept[subject_idx]) + pm.math.dot(controlled_skill,
                                                                                                skill_slope))
            mu = pm.Deterministic("mu", global_mean + control_variable)

            # Define likelihood
            likelihood = pm.NormalMixture('likelihood',
                                          w=groups,
                                          mu=mu,
                                          sigma=global_sigma,
                                          observed=observed, dims="obs")

            # compute diference of mean and std, and effect size

            diff_meansdiff_control = pm.Deterministic('diff_meansdiff_control', global_mean[1] - global_mean[0])
            diff_meansdiff_inefficient = pm.Deterministic('diff_meansdiff_inefficient',
                                                          global_mean[3] - global_mean[2])
            diff_meansdiff_efficient = pm.Deterministic('diff_meansdiff_efficient',
                                                        global_mean[5] - global_mean[4])

            diff_sigmadiff_control = pm.Deterministic('diff_sigmadiff_control', global_sigma[1] - global_sigma[0])
            diff_sigmadiff_inefficient = pm.Deterministic('diff_sigmadiff_inefficient',
                                                          global_sigma[3] - global_sigma[2])
            diff_sigmadiff_efficient = pm.Deterministic('diff_sigmadiff_efficient',
                                                        global_sigma[5] - global_sigma[4])

            # compute effect size
            effect_control = pm.Deterministic(
                "effect_control",
                diff_meansdiff_control / np.sqrt((global_sigma[1] ** 2 + global_sigma[0] ** 2) / 2)
            )
            effect_inefficient = pm.Deterministic(
                "effect_inefficient",
                diff_meansdiff_inefficient / np.sqrt((global_sigma[3] ** 2 + global_sigma[2] ** 2) / 2)
            )
            effect_efficient = pm.Deterministic(
                "effect_efficient",
                diff_meansdiff_efficient / np.sqrt((global_sigma[5] ** 2 + global_sigma[4] ** 2) / 2)
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

        alpha = 0.05
        l = len(trace_post['global_mean'].data[0].flatten())
        low_bound = int(alpha / 2 * l)
        high_bound = int((1 - (alpha / 2)) * l)

        fig, ax = plt.subplots(figsize=(12, 8))
        for idx, group, color in zip([0, 1, 2, 3, 4, 5],
                                     ["control_"+prefix[0]+"_mean", "control_"+prefix[1]+"_mean", "inefficient_"+prefix[0]+"_mean",
                                      "inefficient_"+prefix[1]+"_mean", "efficient_"+prefix[0]+"_mean",
                                      "efficient_"+prefix[1]+"_mean"],
                                     ['#08519c', '#3182bd', '#b30000', "#e34a33", '#006d2c', "#31a354"]):
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

        plt.savefig(DOUBLE_RESULTS_PATH_ANOVA + "posterior\\transition\\" + analyzed_features + ".eps", format="eps")
        plt.savefig(DOUBLE_RESULTS_PATH_ANOVA + "posterior\\transition\\" + analyzed_features + ".png", format="png")
        plt.close()

        # save the model
        with open(DOUBLE_RESULTS_PATH_ANOVA + "model\\" + "idata_m3_transition" + analyzed_features + ".pkl",
                  'wb') as handle:
            print("write data into: " + "idata_m3_" + analyzed_features + ".pkl")
            pickle.dump(idata_m3, handle, protocol=pickle.HIGHEST_PROTOCOL)
