import numpy as np

from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
import pandas as pd
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
from GroupClassification import outliersDetection
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST, \
    DOUBLE_SUMMARY_FILE_PATH, DOUBLE_SUMMARY_FEATURES_PATH
import pickle
from sklearn.preprocessing import StandardScaler
import xarray as xr

np.random.seed(1945)  # For Replicability
from scipy.special import expit

if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)

    # load single and double data
    single_fr = SubjectCrossValidation()
    double_fr = DoubleSubjectCrossValidation()
    fr = GlobalFeaturesReader(single_fr.getSummary(), double_fr.getSummary())
    X, y, group_label = fr.getSingleDoubleFeatures(log_scale=False, col="skill")

    X = np.average(X, axis=-1, keepdims=False)

    labels = outliersDetection(X, y)
    inlier_idx = np.argwhere(labels == 1).flatten()
    inefficient_idx = np.argwhere(labels == 2).flatten()
    efficient_idx = np.argwhere(labels == 3).flatten()

    inlier_group = group_label[inlier_idx]
    inefficient_group = group_label[inefficient_idx]
    efficient_group = group_label[efficient_idx]

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
    #             "bouncing_point_var",
    #             "s1_bouncing_point_var",
    #             "s2_bouncing_point_var",
    #             "receiver_racket_ball_force_ratio_std"]

    features = ["stable_percentage"]

    # load data

    # control group
    control_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                include_subjects=inlier_group, exclude_failure=True,
                                                exclude_no_pair=False, hmm_probs=True)
    control_features = control_reader.getGlobalFeatures(group_label="control")

    # inefficientestimated group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                             file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                             include_subjects=inefficient_group, exclude_failure=True,
                                             exclude_no_pair=False, hmm_probs=True)
    inefficient_features = inefficient_reader.getGlobalFeatures(group_label="inefficient")

    # efficientestimated group
    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                              include_subjects=efficient_group, exclude_failure=True,
                                              exclude_no_pair=False, hmm_probs=True)
    efficient_features = efficient_reader.getGlobalFeatures(group_label="efficient")

    print(control_features.shape)
    print(inefficient_features.shape)
    print(efficient_features.shape)
    indv = pd.concat([control_features, inefficient_features, efficient_features]).reset_index()

    # One Hot Encode Data
    dummies = pd.get_dummies(indv.group)
    df = indv.join(dummies)

    # scaled control
    scaler = StandardScaler()

    control_scaled = scaler.fit_transform(df["group_skill"].values.reshape(-1, 1))
    df["group_skill"] = control_scaled.flatten()

    for f in features:
        print(f)
        analyzed_features = f

        mu_m = df[analyzed_features].mean()
        mu_s = df[analyzed_features].std() * 2

        control_values = df.loc[df["group"] == "control"][analyzed_features].values
        efficient_values = df.loc[df["group"] == "efficient"][analyzed_features].values
        inefficient_values = df.loc[df["group"] == "inefficient"][analyzed_features].values

        control_skill = df.loc[df["group"] == "control"]["group_skill"].values
        efficient_skill = df.loc[df["group"] == "efficient"]["group_skill"].values
        inefficient_skill = df.loc[df["group"] == "inefficient"]["group_skill"].values

        # # plot observed data
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 8))

        print(np.average(control_values))
        print(np.average(efficient_values))
        print(np.average(inefficient_values))

        az.plot_violin(control_values, ax=ax1)
        az.plot_violin(efficient_values, ax=ax2)
        az.plot_violin(inefficient_values, ax=ax3)

        plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "data_dist\\" + analyzed_features + ".png")
        # plt.show()

        with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
            # Define priors

            control_mean = pm.Normal('control_mean', mu=mu_m, sigma=mu_s)
            inefficient_mean = pm.Normal('inefficient_mean', mu=mu_m, sigma=mu_s)
            efficient_mean = pm.Normal('efficient_mean', mu=mu_m, sigma=mu_s)

            # beta fixed effects
            beta_control = pm.Normal('beta_control', mu=0, sigma=10)

            # Define STD
            sigma_low = 0.01
            sigma_high = 10 ** 4 if mu_m > 1 else 10. ** 2

            control_sigma = pm.Uniform("control_sigma", lower=sigma_low, upper=sigma_high)
            efficient_sigma = pm.Uniform("efficient_sigma", lower=sigma_low, upper=sigma_high)
            inefficient_sigma = pm.Uniform("inefficient_sigma", lower=sigma_low, upper=sigma_high)

            # Define nu
            nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            nu = pm.Deterministic("nu", nu_minus_one + 0.1)
            nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

            control = pm.StudentT("control", nu=nu, mu=control_mean + (beta_control * control_skill),
                                  sigma=control_sigma, observed=control_values)
            efficient = pm.StudentT("efficient", nu=nu, mu=efficient_mean + (beta_control * efficient_skill), sigma=efficient_sigma,
                                observed=efficient_values)
            inefficient = pm.StudentT("inefficient", nu=nu, mu=inefficient_mean + (beta_control * inefficient_skill), sigma=inefficient_sigma,
                               observed=inefficient_values)

            # compute diff means
            diff_means_efficient_control = pm.Deterministic('diff_means_efficient_control', efficient_mean - control_mean)
            diff_means_inefficient_control = pm.Deterministic('diff_means_inefficient_control', inefficient_mean - control_mean)
            diff_means_efficient_inefficient = pm.Deterministic('diff_means_efficient_inefficient', efficient_mean - inefficient_mean)

            # compute diff std
            diff_stds_efficient_control = pm.Deterministic("diff_stds_efficient_control", efficient_sigma - control_sigma)
            diff_stds_inefficient_control = pm.Deterministic("diff_stds_inefficient_control", inefficient_sigma - control_sigma)
            diff_stds_efficient_inefficient = pm.Deterministic("diff_stds_efficient_inefficient", efficient_sigma - inefficient_sigma)

            # compute effect size
            effect_efficient_control = pm.Deterministic(
                "effect_efficient_control", diff_means_efficient_control / np.sqrt((control_sigma ** 2 + efficient_sigma ** 2) / 2)
            )
            effect_inefficient_control = pm.Deterministic(
                "effect_inefficient_control", diff_means_inefficient_control / np.sqrt((control_sigma ** 2 + inefficient_sigma ** 2) / 2)
            )
            effect_efficient_inefficient = pm.Deterministic(
                "effect_efficient_inefficient", diff_means_efficient_inefficient / np.sqrt((inefficient_sigma ** 2 + efficient_sigma ** 2) / 2)
            )

            # check the model
            # pm.model_to_graphviz(model).view()
            # Inference!
            idata_m3 = pm.sample_prior_predictive(samples=50, random_seed=100)

        # prior predictive check

        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
        x = xr.DataArray(np.linspace(-1, 1, 50), dims=["plot_dim"])
        prior = idata_m3.prior

        # control
        ax1.plot(x, ((prior["control_mean"] + (prior["beta_control"] * x)).stack(sample=("chain", "draw"))), c="k",
                 alpha=0.4)
        # efficient
        ax2.plot(x, ((prior["efficient_mean"] + (prior["beta_control"] * x)).stack(sample=("chain", "draw"))), c="k",
                 alpha=0.4)
        # inefficient
        ax3.plot(x, ((prior["inefficient_mean"] + (prior["beta_control"] * x)).stack(sample=("chain", "draw"))), c="k",
                 alpha=0.4)

        ax1.set_xlabel("Predictor (stdz)")
        ax1.set_ylabel("Mean Outcome (stdz)")
        plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "prior\\" + analyzed_features + "_prior.png")
        plt.close()

        # sampling

        with model:
            idata_m3.extend(
                pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                          draws=N_SAMPLES,
                          chains=N_CHAINS, tune=N_TUNE, cores=N_CORE)
            )
            idata_m3.extend(pm.sample_posterior_predictive(idata_m3))

        az.plot_forest(idata_m3, var_names=["control_mean", "inefficient_mean", "efficient_mean"])
        plt.xlabel(analyzed_features)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 10)
        plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "posterior\\" + analyzed_features + ".png")
        plt.close()
        # plt.show()

        # save the model
        with open(DOUBLE_RESULTS_PATH_TTEST + "model\\" + "idata_m3_" + analyzed_features + ".pkl", 'wb') as handle:
            print("write data into: " + "idata_m3_" + analyzed_features + ".pkl")
            pickle.dump(idata_m3, handle, protocol=pickle.HIGHEST_PROTOCOL)
