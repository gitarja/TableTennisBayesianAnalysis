import pymc as pm
import pytensor.tensor as pt
import numpy as np


def BayesianTTestModel(coords, observation, session_idx, hitter_receiver_idx, analyzed_features,  truncated_features, mu_m, y_min, BINOMINAL=False, PER_INDIVIDUAL=False):
    ineff_session_idx, eff_session_idx = session_idx
    ineff_hitter_idx, ineff_receiver_idx, eff_hitter_idx, eff_receiver_idx = hitter_receiver_idx
    inefficient_obv,  efficient_obv= observation
    with pm.Model(coords=coords) as model:
        if BINOMINAL:
            if analyzed_features == "gender_sim":
                # id
                subjects_intercept = pm.Normal("subjects_intercept", 0, 0.1, dims=("subject_idx", "gender_idx"))
                sessions_intercept = pm.Normal("sessions_intercept", 0, 0.1, dims=("session_idx", "gender_idx"))

                inefficient_mean = pm.Normal('inefficient_mean', 0, sigma=1, dims="gender_idx")
                efficient_mean = pm.Normal('efficient_mean', 0, sigma=1, dims="gender_idx")
                inefficient_std = inefficient_mean
                efficient_std = efficient_mean

                inefficient = pm.Categorical("inefficient",
                                             logit_p=inefficient_mean
                                                     + sessions_intercept[ineff_session_idx] * (
                                                             subjects_intercept[ineff_hitter_idx] +
                                                             subjects_intercept[ineff_receiver_idx]) / 2

                                             ,
                                             observed=inefficient_obv)
                efficient = pm.Categorical("efficient",
                                           logit_p=efficient_mean
                                                   + sessions_intercept[eff_session_idx] + (
                                                           subjects_intercept[eff_hitter_idx] +
                                                           subjects_intercept[eff_receiver_idx]) / 2

                                           ,
                                           observed=efficient_obv)
            else:
                # number
                print("I am number")
                # id
                subjects_intercept = pm.HalfNormal("subjects_intercept", 0.1, dims="subject_idx")
                sessions_intercept = pm.HalfNormal("sessions_intercept", 0.1, dims="session_idx")

                inefficient_mean = pm.HalfNormal('inefficient_mean', sigma=1)
                efficient_mean = pm.HalfNormal('efficient_mean', sigma=1)
                inefficient_std = inefficient_mean
                efficient_std = efficient_mean

                inefficient = pm.Poisson("inefficient",
                                         mu=inefficient_mean
                                            + sessions_intercept[ineff_session_idx] * (
                                                    subjects_intercept[ineff_hitter_idx] +
                                                    subjects_intercept[ineff_receiver_idx]) / 2

                                         ,
                                         observed=inefficient_obv)
                efficient = pm.Poisson("efficient",
                                       mu=efficient_mean
                                          + sessions_intercept[eff_session_idx] + (
                                                  subjects_intercept[eff_hitter_idx] +
                                                  subjects_intercept[eff_receiver_idx]) / 2

                                       ,
                                       observed=efficient_obv)

        else:

            # continous
            # centered for subjects
            # 0.5 for wide, 0.1 for narrow. Narrow produces better results
            subjects_intercept = pm.Normal("subjects_intercept", mu=0, sigma=0.1, dims="subject_idx")
            # centered for sessions
            sessions_intercept = pm.Normal("sessions_intercept", mu=0, sigma=0.1, dims="session_idx")
            inefficient_std = pm.HalfCauchy("inefficient_std", 1.0)
            efficient_std = pm.HalfCauchy("efficient_std", 1.0)
            inefficient_mean = pm.Normal('inefficient_mean', mu=mu_m, sigma=1)
            efficient_mean = pm.Normal('efficient_mean', mu=mu_m, sigma=1)

            nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            nu = pm.Deterministic("nu", nu_minus_one + 1)

            a = pm.Gamma("a", alpha=2, beta=0.5)  # controls tails, higher = lighter tail
            b = pm.Gamma("b", alpha=2, beta=0.5)  # controls skewness

            lambda_1 = efficient_std ** -2
            lambda_2 = inefficient_std ** -2

            if PER_INDIVIDUAL:
                inefficient_latent = pm.Deterministic("iefficient_latent", inefficient_mean
                                                      + (
                                                              subjects_intercept[ineff_hitter_idx] +
                                                              subjects_intercept[ineff_receiver_idx]) / 2)

                efficient_latent = pm.Deterministic("efficient_latent", efficient_mean
                                                    + (
                                                            subjects_intercept[eff_hitter_idx] +
                                                            subjects_intercept[eff_receiver_idx]) / 2)
            else:
                inefficient_latent = pm.Deterministic("iefficient_latent", inefficient_mean
                                                      + sessions_intercept[ineff_session_idx] + (
                                                              subjects_intercept[ineff_hitter_idx] +
                                                              subjects_intercept[ineff_receiver_idx]) / 2)

                efficient_latent = pm.Deterministic("efficient_latent", efficient_mean
                                                    + sessions_intercept[eff_session_idx] + (
                                                            subjects_intercept[eff_hitter_idx] +
                                                            subjects_intercept[eff_receiver_idx]) / 2)

            if analyzed_features in truncated_features:
                inefficient = pm.TruncatedNormal("inefficient",
                                                 lower=y_min,
                                                 mu=inefficient_latent,
                                                 sigma=inefficient_std,

                                                 observed=inefficient_obv)
                efficient = pm.TruncatedNormal("efficient",
                                               lower=y_min,
                                               mu=efficient_latent,
                                               sigma=efficient_std, observed=efficient_obv)

            else:

                # inefficient = pm.StudentT("inefficient", nu=nu,
                #                           mu=inefficient_latent, lam=lambda_2,
                #                           observed=inefficient_obv)
                # efficient = pm.StudentT("efficient", nu=nu, mu=efficient_latent,
                #                         lam=lambda_1, observed=efficient_obv)

                inefficient = pm.SkewStudentT("inefficient",
                                              a=a,
                                              b=b,
                                              mu=inefficient_latent,
                                              sigma=inefficient_std,

                                              observed=inefficient_obv)
                efficient = pm.SkewStudentT("efficient",
                                            a=a,
                                            b=b,
                                            mu=efficient_latent,
                                            sigma=efficient_std, observed=efficient_obv)

        # means difference and others

        diff_of_means = pm.Deterministic("difference_of_means", efficient_mean - inefficient_mean)
        diff_of_stds = pm.Deterministic("difference_of_stds", efficient_std - inefficient_std)

        if (BINOMINAL) or (analyzed_features == "gender_sim"):
            print("effect_size")
            effect_size = pm.Deterministic(
                "effect_size", diff_of_means / np.sqrt((inefficient_std + efficient_std) / 2)
            )
        else:
            effect_size = pm.Deterministic(
                "effect_size", diff_of_means / np.sqrt((inefficient_std ** 2 + efficient_std ** 2) / 2)
            )

    return model