import pymc as pm
import pytensor.tensor as pt
import numpy as np

eps = np.finfo(float).eps


def CenteredModel(coords, df, subject_idx, analyzed_features, n=0, BINOMINAL=False, hitter=False):
    print(np.nanmean(df.loc[df.efficient == True, analyzed_features].values))
    print(np.nanmean(df.loc[df.efficient == False, analyzed_features].values))

    mu_m = np.nanmean(df[analyzed_features].values)
    mu_s = np.nanstd(df[analyzed_features].values) * 2
    mu_change_m = np.nanmean(np.abs(np.diff(df[analyzed_features].values)))
    mu_change_s = np.nanstd(np.abs(np.diff(df[analyzed_features].values))) * 2

    print(mu_change_m)
    print(mu_change_s)

    with pm.Model(coords=coords) as model:
        th_segments = pm.Data("th_segments", df["th_segments"].values)
        efficient = pm.Data("efficient", df["efficient"].values.astype(float))
        inefficient = pm.Data("inefficient", df["inefficient"].values.astype(float))
        observed = pm.Data("observed", df[analyzed_features].values)

        # skilled
        if hitter:
            controlled_skill = pm.Data("controlled_skill",
                                       df["hitter_skill"].values.astype(float),
                                       mutable=True, dims=("obs"))
        else:
            controlled_skill = pm.Data("controlled_skill",
                                       df["receiver_skill"].values.astype(float),
                                       mutable=True, dims=("obs"))

        # Define priors
        # level 1
        if BINOMINAL:
            global_intercept = pm.TruncatedNormal("global_intercept", 0, 1, lower=0)
            global_th_segment = pm.TruncatedNormal("global_th_segment", 0, 1, lower=0)
            global_efficient = pm.TruncatedNormal("global_efficient", 0, 1, lower=0)
            global_efficient_seg = pm.TruncatedNormal("global_efficient_seg", 0, 1, lower=0)

            global_inefficient = pm.TruncatedNormal("global_inefficient", 0, 1, lower=0)
            global_inefficient_seg = pm.TruncatedNormal("global_inefficient_seg", 0, 1, lower=0)

            global_skill_slope = pm.TruncatedNormal('global_skill_slope', 0, 0.01, lower=0)

            # level 2
            # fixed effect params
            subjects_intercept = pm.TruncatedNormal('subjects_intercept', mu=0, sigma=0.01, dims=("subject_idx"),
                                                    lower=0)
            subjects_intercept_seg = pm.TruncatedNormal('subjects_intercept_seg', mu=0, sigma=0.01,
                                                        dims=("subject_idx"), lower=0)
        else:
            global_intercept = pm.Normal("global_intercept", 0, 1)
            global_th_segment = pm.Normal("global_th_segment", 0, 1)
            global_efficient = pm.Normal("global_efficient", mu_m, 1)
            global_efficient_seg = pm.Normal("global_efficient_seg", mu_change_m, 1)

            global_inefficient = pm.Normal("global_inefficient", mu_m, 1)
            global_inefficient_seg = pm.Normal("global_inefficient_seg", mu_change_m, 1)

            global_skill_slope = pm.Normal('global_skill_slope', 0, 0.01)

            # level 2
            # fixed effect params
            subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=0.01, dims=("subject_idx"))
            subjects_intercept_seg = pm.Normal('subjects_intercept_seg', mu=0, sigma=0.01,
                                               dims=("subject_idx"))

        control_skilled_variable = pm.Deterministic("control_skilled_variable", controlled_skill * global_skill_slope)
        
        # difference
        global_diff_of_means = pm.Deterministic("global_diff_of_means", global_efficient - global_inefficient)
        global_seg_diff_of_means = pm.Deterministic("global_seg_diff_of_means", global_efficient_seg - global_inefficient_seg)

        mu = pm.Deterministic("mu",
                              (global_intercept + subjects_intercept[subject_idx])
                              # efficient * seg
                               + (global_efficient * efficient)
                              + (global_efficient_seg * (efficient * th_segments))

                              # inefficient * seg

                              + (global_inefficient * inefficient)
                              + (global_inefficient_seg * (inefficient * th_segments))


                              + ((global_th_segment + subjects_intercept_seg[subject_idx]) * th_segments)

                                +control_skilled_variable
                              )

        if BINOMINAL:
            growth_model = pm.Poisson(
                "growth_model",
                mu=mu,
                observed=observed,
                dims="obs"

            )
        else:

            # likelihood
            global_sigma = pm.HalfStudentT("global_sigma", 1, 3)
            # growth_model = pm.TruncatedNormal(
            #     "growth_model",
            #     mu=mu,
            #     sigma=global_sigma,
            #     observed=observed,
            #     dims="obs",
            #
            #
            # )
            nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            nu = pm.Deterministic("nu", nu_minus_one + 1)
            nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))
            growth_model = pm.StudentT("growth_model", nu=nu, mu=mu,
                                    lam=global_sigma, observed=observed, dims="obs",)

            # growth_model = pm.Normal(
            #     "growth_model",
            #     mu=mu,
            #     sigma=global_sigma,
            #     observed=observed,
            #     dims="obs",
            #
            #
            # )

    return model



