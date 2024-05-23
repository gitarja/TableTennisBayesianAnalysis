import numpy as np
import arviz as az
from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import scipy.stats as stats
from OutliersLib import OutliersDetection

np.random.seed(1945)  # For Replicability

if __name__ == '__main__':
    results_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double\\Bayesian-anova\\"

    rng = np.random.default_rng(seed=42)

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

    features = ["receiver_p1_al", "receiver_p2_al", "receiver_pursuit", "receiver_pursuit_duration", "hitter_p1_al", "hitter_p2_al", "hitter_pursuit",
                "hitter_pursuit_duration", "receiver_fs_ball_racket_dir_std"]

    # features = ["receiver_fs_ball_racket_dir_std"]

    for f in features:
        analyzed_features = f
        # load data
        path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double_episode_features.pkl"

        # control group
        control_reader = GlobalDoubleFeaturesReader(file_path=path, include_subjects=inlier_group, exclude_failure=False)
        control_features = control_reader.getGlobalFeatures(group_label="control")

        # overestimated group
        over_reader = GlobalDoubleFeaturesReader(file_path=path, include_subjects=over_group, exclude_failure=False)
        over_features = over_reader.getGlobalFeatures(group_label="over")

        # underestimated group
        under_reader = GlobalDoubleFeaturesReader(file_path=path, include_subjects=under_group, exclude_failure=False)
        under_features = under_reader.getGlobalFeatures(group_label="under")

        print(control_features.shape)
        print(over_features.shape)
        print(under_features.shape)
        indv = pd.concat([control_features, over_features, under_features]).reset_index()

        # One Hot Encode Data
        dummies = pd.get_dummies(indv.group)
        # dummies.columns = ['control','over','under']
        df = indv.join(dummies)


        group_id_idx, _ = pd.factorize(df["group"])
        coords = {"group_id_idx": group_id_idx, "obs": range(len(df[analyzed_features]))}
        with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
            # Define priors
            sigma = pm.HalfCauchy("sigma", beta=10)
            control = pm.Normal('control', mu=control_features[analyzed_features].mean(), sigma=control_features[analyzed_features].std())
            over = pm.Normal('over', mu=over_features[analyzed_features].mean(), sigma=over_features[analyzed_features].std())
            under = pm.Normal('under', mu=under_features[analyzed_features].mean(), sigma=under_features[analyzed_features].std())

            global_mu = pm.Deterministic("global_mu", control * df['control'] + under * df['under'] + over * df['over'])
            # Define likelihood
            likelihood = pm.Normal('likelihood',
                                   mu=global_mu,
                                   sigma=sigma,
                                   observed=df[analyzed_features].values)

            # likelihood = pm.StudentT("likelihood", nu=7, mu=global_mu, sigma=sigma, observed=df[analyzed_features].values)

            # Inference!
            trace = pm.sample(4000, cores=4, chains=4, random_seed=100, tune=1000,  target_accept=1.0)  # draw 4000 posterior samples using NUTS sampling

        # pm.model_to_graphviz(model).view()
        trace_post = az.extract(trace.posterior)

        # Get posterior samples for the parameter of interest
        print("Features: "+ analyzed_features)
        control_posterior_samples = np.concatenate([trace_post['control'].data.flatten()])
        control_credible_interval = np.percentile(control_posterior_samples, [5, 95.0])

        under_posterior_samples = np.concatenate([trace_post['under'].data.flatten()])
        under_credible_interval = np.percentile(under_posterior_samples, [5, 95.0])

        over_posterior_samples = np.concatenate([trace_post['over'].data.flatten()])
        over_credible_interval = np.percentile(over_posterior_samples, [5, 95.0])
        print("Control Credible Interval (95%):", control_credible_interval)
        print("Under Credible Interval (95%):", under_credible_interval)
        print("Over Credible Interval (95%):", over_credible_interval)

        alpha = 0.05
        l = len(trace_post['control'].data.flatten())
        low_bound = int(alpha / 2 * l)
        high_bound = int((1 - (alpha / 2)) * l)

        fig, ax = plt.subplots(figsize=(12, 8))
        for group, color in zip(['control', 'over', 'under'], ['#377eb8', '#e41a1c', '#4daf4a']):
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
        plt.savefig(results_path  + analyzed_features+".png")
        plt.close()
