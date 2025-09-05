import pymc as pm
import numpy as np


def CenteredModel(coords, df, actor_idx, partner_idx, turn_take_idx, prev_features, next_features, hitter=False):
    with pm.Model(coords=coords) as model:
        higher = pm.Data("higher", df["higher"].values.astype(float))
        lower = pm.Data("lower", df["lower"].values.astype(float))
        prev_observed = pm.Data("prev_observed", df[prev_features + "_prev"].values)
        observed = pm.Data("observed", df[next_features + "_next"].values)
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
        global_intercept = pm.Normal("global_intercept", 0, 1)
        global_influenceActorPartner = pm.Normal("global_influenceActorPartner", 0, 1)

        global_higher = pm.Normal("global_higher", 0, 1)

        global_higher_influenceActorPartner = pm.Normal("global_higher_influenceActorPartner", 0, 1, dims=(
            "turn_take_idx"))  # influence of a to b [0] and b to a [1]

        global_lower = pm.Normal("global_lower", 0, 1)

        global_lower_influenceActorPartner = pm.Normal("global_lower_influenceActorPartner", 0, 1,
                                                       dims=("turn_take_idx"))

        global_skill_slope = pm.Normal('global_skill_slope', 0, 0.01)

        # level 2
        # fixed effect params
        subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=0.01, dims=("subject_idx"))
        subjects_intercept_W = pm.Normal('subjects_intercept_W', mu=0, sigma=0.01,
                                         dims=("subject_idx"))

        control_skilled_variable = pm.Deterministic("control_skilled_variable", controlled_skill * global_skill_slope)

        mu = pm.Deterministic("mu",
                              (global_intercept )

                              # higher * prev observation
                              + (global_higher * higher)
                              + ((global_higher_influenceActorPartner[turn_take_idx]+ subjects_intercept[actor_idx]) * (higher * prev_observed))

                              # lower * prev observation
                              + (global_lower * lower)
                              + ((global_lower_influenceActorPartner[turn_take_idx]+ subjects_intercept[actor_idx]) * (lower * prev_observed))

                              # prev observation
                              + ((global_influenceActorPartner ) * prev_observed)

                              + control_skilled_variable
                              )

        # likelihood
        global_sigma = pm.HalfStudentT("global_sigma", 1, 3)

        growth_model = pm.Normal(
            "growth_model",
            mu=mu,
            sigma=global_sigma,
            observed=observed,
            dims="obs",

        )

        # mean difference
        global_higher_influenceActorPartner_diff = pm.Deterministic("global_higher_influenceActorPartner_diff",
                                                                    global_higher_influenceActorPartner[1] -
                                                                    global_higher_influenceActorPartner[0])
        global_lower_influenceActorPartner_diff = pm.Deterministic("global_lower_influenceActorPartner_diff",
                                                                   global_lower_influenceActorPartner[1] -
                                                                   global_lower_influenceActorPartner[0])

    return model


def SingleCenteredModel(coords, df, actor_idx, prev_features, next_features, hitter=False):
    with pm.Model(coords=coords) as model:
        higher = pm.Data("higher", df["higher"].values.astype(float))
        lower = pm.Data("lower", df["lower"].values.astype(float))
        prev_observed = pm.Data("prev_observed", df[prev_features + "_prev"].values)
        observed = pm.Data("observed", df[next_features + "_next"].values)
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
        global_intercept = pm.Normal("global_intercept", 0, 1)
        global_influenceActorPartner = pm.Normal("global_influenceActorPartner", 0, 1)

        global_higher = pm.Normal("global_higher", 0, 1)

        global_higher_influenceActorPartner = pm.Normal("global_higher_influenceActorPartner", 0, 1)  # influence of a to b [0] and b to a [1]

        global_lower = pm.Normal("global_lower", 0, 1)

        global_lower_influenceActorPartner = pm.Normal("global_lower_influenceActorPartner", 0, 1)

        global_skill_slope = pm.Normal('global_skill_slope', 0, 0.01)

        # level 2
        # fixed effect params
        subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=0.01, dims=("subject_idx"))
        subjects_intercept_W = pm.Normal('subjects_intercept_W', mu=0, sigma=0.01,
                                         dims=("subject_idx"))

        control_skilled_variable = pm.Deterministic("control_skilled_variable", controlled_skill * global_skill_slope)

        mu = pm.Deterministic("mu",
                              (global_intercept )

                              # higher * prev observation
                              + (global_higher * higher)
                              + ((global_higher_influenceActorPartner+ subjects_intercept[actor_idx]) * (higher * prev_observed))

                              # lower * prev observation
                              + (global_lower * lower)
                              + ((global_lower_influenceActorPartner+ subjects_intercept[actor_idx]) * (lower * prev_observed))

                              # prev observation
                              + ((global_influenceActorPartner ) * prev_observed)

                              + control_skilled_variable
                              )

        # likelihood
        global_sigma = pm.HalfStudentT("global_sigma", 1, 3)

        growth_model = pm.Normal(
            "growth_model",
            mu=mu,
            sigma=global_sigma,
            observed=observed,
            dims="obs",

        )


    return model
