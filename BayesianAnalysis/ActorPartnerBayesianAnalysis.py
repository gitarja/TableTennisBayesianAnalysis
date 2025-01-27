import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, features_explanation
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_APM_TTEST
import arviz as az
import pickle
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams['font.size'] = 20

if __name__ == '__main__':

    lower_group, upper_group = groupLabeling()

    # inefficient group
    lower_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                              include_subjects=lower_group, exclude_failure=True,
                                              exclude_no_pair=False, hmm_probs=True)

    higher_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                               file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                               include_subjects=upper_group, exclude_failure=True,
                                               exclude_no_pair=False, hmm_probs=True)

    higher_features = higher_reader.getCoupledFeatures(group_name="efficient",
                                                       success_failure=True,
                                                       mod="skill_personal_perception_action_impact",
                                                       with_control=True)

    lower_features = lower_reader.getCoupledFeatures(group_name="efficient",
                                                     success_failure=True,
                                                     mod="skill_personal_perception_action_impact",
                                                     with_control=True)

    lower_features["group"] = "lower"

    higher_features["group"] = "upper"

    df = pd.concat([lower_features, higher_features])

    features = ["onset_forward_swing", "al_prec_p1", "al_prec_HP_p1"]
    # features = ["ball_updown"]

    for f in features:

        print(f)
        analyzed_features = f
        clean_df = df.dropna(subset=[analyzed_features + "_prev", analyzed_features + "_next"])

        # az.plot_dist(clean_df[analyzed_features + "_next"])
        # plt.show()

        # standarize features
        # scaler = StandardScaler()
        # feature = np.concatenate([clean_df[analyzed_features + "_prev"].values, clean_df[analyzed_features + "_next"].values]).reshape(-1, 1)
        # scaler.fit(feature)
        # clean_df[analyzed_features + "_prev"] = scaler.transform(clean_df[analyzed_features + "_prev"].values.reshape(-1, 1))
        # clean_df[analyzed_features + "_next"] = scaler.transform(clean_df[analyzed_features + "_next"].values.reshape(-1, 1))

        # factorize receiver
        lower_partner_idx, lower_partner_unique = pd.factorize(
            clean_df.loc[clean_df["group"] == "lower"]["receiver"].values)
        upper_partner_idx, upper_partner_unique = pd.factorize(
            clean_df.loc[clean_df["group"] == "upper"]["receiver"].values)

        lower_actor_idx, lower_actor_unique = pd.factorize(
            clean_df.loc[clean_df["group"] == "lower"]["hitter"].values)
        upper_actor_idx, upper_actor_unique = pd.factorize(
            clean_df.loc[clean_df["group"] == "upper"]["hitter"].values)

        lower_hitter_idx, lower_hitter_unique = pd.factorize(
            clean_df.loc[clean_df["group"] == "lower"]["hitter_idx"].values
        )
        upper_hitter_idx, upper_hitter_unique = pd.factorize(
            clean_df.loc[clean_df["group"] == "upper"]["hitter_idx"].values
        )

        coords = {"lower_partner_idx": lower_partner_unique, "upper_partner_idx": upper_partner_unique,
                  "lower_actor_idx": lower_actor_unique, "upper_actor_idx": upper_actor_unique,
                  "hitter_idx": range(2)}

        lower_obv = clean_df.loc[clean_df["group"] == "lower"][analyzed_features + "_next"].values
        upper_obv = clean_df.loc[clean_df["group"] == "upper"][analyzed_features + "_next"].values

        with pm.Model(coords=coords) as model:  # model specifications in PyMC3 are wrapped in a with-statement

            # data
            upper_hitter_features = pm.Data("upper_hitter_features", clean_df.loc[clean_df["group"] == "upper"][
                analyzed_features + "_prev"].values.astype(float))
            lower_hitter_features = pm.Data("lower_hitter_features", clean_df.loc[clean_df["group"] == "lower"][
                analyzed_features + "_prev"].values.astype(float))

            # global intercept
            global_intercept = pm.Normal("global_intercept", 0, 1)
            # random intercept
            lower_actor_intercept = pm.Normal("lower_actor_intercept", 0, 1, dims="lower_actor_idx")
            lower_partner_intercept = pm.Normal("lower_partner_intercept", 0, 1, dims="lower_partner_idx")
            upper_actor_intercept = pm.Normal("upper_actor_intercept", 0, 1, dims="upper_actor_idx")
            upper_partner_intercept = pm.Normal("upper_partner_intercept", 0, 1, dims="upper_partner_idx")

            lower_subjects_intercept = pm.Deterministic("lower_subjects_intercept",
                                                        global_intercept + lower_actor_intercept[lower_actor_idx] +
                                                        lower_partner_intercept[lower_partner_idx])

            upper_subjects_intercept = pm.Deterministic("upper_subjects_intercept",
                                                        global_intercept + upper_actor_intercept[upper_actor_idx] +
                                                        upper_partner_intercept[upper_partner_idx])

            upper_W = pm.Normal('upper_W', mu=0, sigma=1, dims="hitter_idx")

            lower_W = pm.Normal('lower_W', mu=0, sigma=1, dims="hitter_idx")

            upper_mu = pm.Deterministic("upper_mu",
                                        (upper_W[upper_hitter_idx] * upper_hitter_features) + upper_subjects_intercept)
            lower_mu = pm.Deterministic("lower_mu",
                                        (lower_W[lower_hitter_idx] * lower_hitter_features) + lower_subjects_intercept)

            global_std = pm.HalfStudentT("global_std", 1, 3)

            # likelihood
            upper_next = pm.Normal(
                "upper_next",
                mu=upper_mu,
                sigma=global_std,
                observed=upper_obv,

            )
            lower_next = pm.Normal(
                "lower_next",
                mu=lower_mu,
                sigma=global_std,
                observed=lower_obv,

            )

            upper_diff = pm.Deterministic("upper_diff", upper_W[0] - upper_W[1])
            lower_diff = pm.Deterministic("lower_diff", lower_W[0] - lower_W[1])

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
        hierarchical_loo = az.plot_ppc(idata, kind='cumulative')
        plt.savefig(DOUBLE_RESULTS_PATH_APM_TTEST + "PPC\\" + analyzed_features + ".png", format='png')
        plt.close()

        # az.plot_posterior(
        #     idata, var_names=["global_intercept", "upper_W", "lower_W", "upper_diff", "lower_diff"], figsize=(15, 4)
        # )
        # plt.show()

        trace_post = az.extract(idata.posterior)

        alpha = 0.05
        l = len(trace_post['upper_diff'].data.flatten())
        low_bound = int(alpha / 2 * l)
        high_bound = int((1 - (alpha / 2)) * l)

        fig, ax = plt.subplots(figsize=(12, 8))
        for group, color in zip(['upper_diff', 'lower_diff'],
                                ['#69a87f', '#b5152c']):
            data = (trace_post[group].data.flatten())
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
        # plt.xlabel(features_explanation[analyzed_features])
        # plt.show()

        plt.savefig(DOUBLE_RESULTS_PATH_APM_TTEST + "MU\\" + analyzed_features + ".pdf", format='pdf')
        plt.close()
        # save the model
        with open(DOUBLE_RESULTS_PATH_APM_TTEST + "model\\" + "idata_" + analyzed_features + ".pkl", 'wb') as handle:
            print("write data into: " + "idata_ttest_" + analyzed_features + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del model
        del idata
