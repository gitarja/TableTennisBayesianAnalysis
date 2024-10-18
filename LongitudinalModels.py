import pymc as pm
import pytensor.tensor as pt
import numpy as np

eps = np.finfo(float).eps



def CenteredModel(coords, df, BINOMINAL, subject_idx, analyzed_features, n, hitter=False):
    mu = df[analyzed_features].mean()
    with pm.Model(coords=coords) as model:
        th_segments = pm.Data("th_segments", df["th_segments"].values, mutable=True,
                              dims=("obs"))

        groups = pm.Data("groups", df[["control", "inefficient", "efficient"]].values.astype(float), mutable=True,
                         dims=("obs", "group"))

        # skilled
        if hitter:
            controlled_skill = pm.Data("controlled_skill",
                                       df["hitter_skill"].values.astype(float),
                                       mutable=True, dims=("obs"))
        else:
            controlled_skill = pm.Data("controlled_skill",
                                       df["receiver_skill"].values.astype(float),
                                       mutable=True, dims=("obs"))

        observed = pm.Data("observed", df[analyzed_features].values, mutable=True, dims="obs")

        # Define priors
        # level 1
        global_intercept = pm.Normal("global_intercept", 0, 1)
        global_th_segment = pm.Normal("global_th_segment", 0, 1)
        global_group = pm.Normal("global_group", 0, 1, dims="group")
        global_group_seg = pm.Normal("global_group_seg", 0, 1, dims="group")



        # level 2
        # fixed effect params
        global_skill_slope = pm.Normal('global_skill_slope', mu=0, sigma=1)
        subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=1, dims=("subject_idx"))
        subjects_intercept_seg = pm.Normal('subjects_intercept_seg', mu=0, sigma=1, dims=("subject_idx"))

        control_skilled_variable = pm.Deterministic("control_skilled_variable", controlled_skill * global_skill_slope)

        mu = pm.Deterministic("mu", (global_intercept + subjects_intercept[subject_idx])
                              #  control
                              + (global_group_seg[0] * th_segments * groups[:, 0])
                              + (global_group[0] * groups[:, 0])
                              #  inefficient
                              + (global_group_seg[1] * th_segments * groups[:, 1])
                              + (global_group[1] * groups[:, 1])
                              #  efficient
                              + (global_group_seg[2] * th_segments * groups[:, 2])
                              + (global_group[2] * groups[:, 2])

                              + ((global_th_segment + subjects_intercept_seg[subject_idx]) * th_segments)
                              + (control_skilled_variable)
                              )

        if BINOMINAL:
            growth_model = pm.Binomial(
                "growth_model",
                n=n,
                p=pm.math.invlogit(mu),
                observed = observed,
                dims = "obs"

            )
        else:

            # likelihood
            global_sigma = pm.HalfNormal("global_sigma", 0.5)
            growth_model = pm.Normal(
                "growth_model",
                mu=mu,
                sigma=global_sigma,
                observed=observed,
                dims="obs"
            )

    return model


def CenteredPolyModel(coords, df, BINOMINAL, subject_idx, analyzed_features, n, hitter=False):
    with pm.Model(coords=coords) as model:
        th_segments = pm.Data("th_segments", df["th_segments"].values, mutable=True,
                              dims=("obs"))

        th_segments2 = pm.Data("th_segments2", df["th_segments"].values ** 2, mutable=True,
                              dims=("obs"))

        groups = pm.Data("groups", df[["control", "inefficient", "efficient"]].values.astype(float), mutable=True,
                         dims=("obs", "group"))

        # skilled
        if hitter:
            controlled_skill = pm.Data("controlled_skill",
                                       df["hitter_skill"].values.astype(float),
                                       mutable=True, dims=("obs"))
        else:
            controlled_skill = pm.Data("controlled_skill",
                                       df["receiver_skill"].values.astype(float),
                                       mutable=True, dims=("obs"))

        observed = pm.Data("observed", df[analyzed_features].values, mutable=True, dims="obs")

        # Define priors
        # level 1
        global_intercept = pm.Normal("global_intercept", 0, 1)
        global_th_segment = pm.Normal("global_th_segment", 0, 1)
        global_th_segment2 = pm.Normal("global_th_segment2", 0, 1)

        global_group = pm.Normal("global_group", 0, 1, dims="group")
        global_group_seg = pm.Normal("global_group_seg", 0, 1, dims="group")
        global_group_seg2 = pm.Normal("global_group_seg2", 0, 1, dims="group")

        # level 2
        # fixed effect params
        global_skill_slope = pm.Normal('global_skill_slope', mu=0, sigma=1)
        subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=1, dims=("subject_idx"))
        subjects_intercept_seg = pm.Normal('subjects_intercept_seg', mu=0, sigma=1, dims=("subject_idx"))
        subjects_intercept_seg2 = pm.Normal('subjects_intercept_seg2', mu=0, sigma=1, dims=("subject_idx"))

        control_skilled_variable = pm.Deterministic("control_skilled_variable",
                                                    pm.math.dot(controlled_skill, global_skill_slope))

        mu = pm.Deterministic("mu", (global_intercept + subjects_intercept[subject_idx])
                              #  control
                              + (global_group_seg2[0]  * (th_segments2 * groups[:, 0]))
                              + (global_group_seg[0] * th_segments * groups[:, 0])
                              + (global_group[0] * groups[:, 0])
                              #  inefficient
                              + (global_group_seg2[1] * (th_segments2 * groups[:, 1]))
                              + (global_group_seg[1] * th_segments * groups[:, 1])
                              + (global_group[1] * groups[:, 1])
                              #  efficient
                              + (global_group_seg2[2] * (th_segments2 * groups[:, 2]))
                              + (global_group_seg[2] * th_segments * groups[:, 2])
                              + (global_group[2] * groups[:, 2])

                              + ((global_th_segment + subjects_intercept_seg[subject_idx]) * th_segments)
                              + ((global_th_segment2 + subjects_intercept_seg2[subject_idx]) * th_segments2)
                              + (control_skilled_variable)
                              )

        if BINOMINAL:
            growth_model = pm.Binomial(
                "growth_model",
                n=n,
                p=pm.math.invlogit(mu),
                observed=observed,
                dims="obs"
            )
            # global_sigma = pm.HalfNormal("global_sigma", 0.5)
            # outcome_latent = pm.Gumbel.dist(mu, global_sigma)
            # growth_model = pm.Censored(
            #     "growth_model", outcome_latent, lower=0, upper=n, observed=observed, dims="obs"
            # )
        else:

            # likelihood
            global_sigma = pm.HalfNormal("global_sigma", 0.5)
            growth_model = pm.Normal(
                "growth_model",
                mu=mu,
                sigma=global_sigma,
                observed=observed,
                dims="obs"
            )

    return model


