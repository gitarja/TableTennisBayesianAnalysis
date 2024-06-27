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

    rng = np.random.default_rng(seed=42)

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
    #             "s2_bouncing_point_var"]


    features = ["hitter_pursuit"]

    # load data

    # control group
    control_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                include_subjects=inlier_group, exclude_failure=True,
                                                exclude_no_pair=False)
    control_features = control_reader.getGlobalFeatures(group_label="control")

    # overestimated group
    over_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                             file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                             include_subjects=over_group, exclude_failure=True,
                                             exclude_no_pair=False)
    over_features = over_reader.getGlobalFeatures(group_label="over")

    # underestimated group
    under_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                              include_subjects=under_group, exclude_failure=True,
                                              exclude_no_pair=False)
    under_features = under_reader.getGlobalFeatures(group_label="under")

    print(control_features.shape)
    print(over_features.shape)
    print(under_features.shape)
    indv = pd.concat([control_features, over_features, under_features]).reset_index()

    # One Hot Encode Data
    dummies = pd.get_dummies(indv.group)
    # dummies.columns = ['control','over','under']
    df = indv.join(dummies)

    # scaled control
    scaler = StandardScaler()

    control_scaled = scaler.fit_transform(df["group_skill"].values.reshape(-1, 1))
    df["group_skill"] = control_scaled.flatten()

    for f in features:
        analyzed_features = f

        # prior
        mu_m = df[analyzed_features].mean()
        mu_s = df[analyzed_features].std() * 2


        with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
            # Define priors
            global_sigma = pm.HalfCauchy("sigma", beta=10)

            # mean
            control_mean = pm.Normal('control_mean', mu=mu_m, sigma=mu_s)
            over_mean = pm.Normal('over_mean', mu=mu_m, sigma=mu_s)
            under_mean = pm.Normal('under_mean', mu=mu_m, sigma=mu_s)

            control = pm.Data("control", df["control"].values.astype(float), mutable=True)
            over = pm.Data("over", df["over"].values.astype(float), mutable=True)
            under = pm.Data("under", df["under"].values.astype(float), mutable=True)

            controlled_skill = pm.Data("group_skill", df["group_skill"].values.astype(float), mutable=True)

            beta_control = pm.Normal('beta_control', mu=0, sigma=10)

            global_mu = pm.Deterministic("global_mu",
                                         (control_mean * control + under_mean * under + over_mean * over) + (beta_control * controlled_skill) )
            # Define likelihood
            likelihood = pm.Normal('likelihood',
                                   mu=global_mu,
                                   sigma=global_sigma,
                                   observed=df[analyzed_features].values)

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

        # az.plot_trace(idata_m3)
        # plt.show()

        hierarchical_loo = az.loo(idata_m3)

        print(hierarchical_loo)
        # pm.model_to_graphviz(model).view()
        trace_post = az.extract(idata_m3.posterior)

        # Get posterior samples for the parameter of interest
        print("Features: " + analyzed_features)
        control_posterior_samples = np.concatenate([trace_post['control_mean'].data.flatten()])
        control_credible_interval = np.percentile(control_posterior_samples, [5, 95.0])

        under_posterior_samples = np.concatenate([trace_post['under_mean'].data.flatten()])
        under_credible_interval = np.percentile(under_posterior_samples, [5, 95.0])

        over_posterior_samples = np.concatenate([trace_post['over_mean'].data.flatten()])
        over_credible_interval = np.percentile(over_posterior_samples, [5, 95.0])

        print("Control Credible Interval (95%):", control_credible_interval)
        print("Under Credible Interval (95%):", under_credible_interval)
        print("Over Credible Interval (95%):", over_credible_interval)

        alpha = 0.05
        l = len(trace_post['control_mean'].data.flatten())
        low_bound = int(alpha / 2 * l)
        high_bound = int((1 - (alpha / 2)) * l)

        fig, ax = plt.subplots(figsize=(12, 8))
        for group, color in zip(['control_mean', 'over_mean', 'under_mean'], ['#377eb8', '#e41a1c', '#4daf4a']):
            # Estimate KDE
            kde = stats.gaussian_kde(trace_post[group].data.flatten())
            # plot complete kde curve as line
            pos = np.linspace(trace_post[group].min(), trace_post[group].max(), 101)
            plt.plot(pos, kde(pos), color=color, label='{0} KDE'.format(group))
            # Set shading bounds
            low = np.sort(trace_post[group].data.flatten())[low_bound]
            high = np.sort(trace_post[group].data.flatten())[high_bound]
            # plot shaded kde
            shade = np.linspace(low, high, 101)
            plt.fill_between(shade, kde(shade), alpha=0.3, color=color, label="{0} 95% HPD Score".format(group))
        plt.legend()
        plt.xlabel(analyzed_features)

        plt.savefig(DOUBLE_RESULTS_PATH_ANOVA + "posterior\\" + analyzed_features + ".png")
        plt.close()

        # save the model
        with open(DOUBLE_RESULTS_PATH_ANOVA + "model\\"  + "idata_m3_" + analyzed_features + ".pkl", 'wb') as handle:
            print("write data into: " + "idata_m3_" + analyzed_features + ".pkl")
            pickle.dump(idata_m3, handle, protocol=pickle.HIGHEST_PROTOCOL)
