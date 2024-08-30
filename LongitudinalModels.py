import pymc as pm
import pytensor.tensor as pt
import numpy as np

eps = np.finfo(float).eps


def CenteredModelSim(coords, df, BINOMINAL, session_id_idx, analyzed_features, n):
    with pm.Model(coords=coords) as model:
        th_segments = pm.Data("th_segments", df["th_segments"].values, mutable=True)

        # sim group
        control_sim = pm.Data("control_sim", df["control_sim"].values.astype(float), mutable=True)
        over_sim = pm.Data("over_sim", df["over_sim"].values.astype(float), mutable=True)
        under_sim = pm.Data("under_sim", df["under_sim"].values.astype(float), mutable=True)

        # dis group
        control_dis = pm.Data("control_dis", df["control_dis"].values.astype(float), mutable=True)
        over_dis = pm.Data("over_dis", df["over_dis"].values.astype(float), mutable=True)
        under_dis = pm.Data("under_dis", df["under_dis"].values.astype(float), mutable=True)

        # level 1

        global_intercept = pm.Normal("global_intercept", 0, 0.75)
        global_th_segment = pm.Normal("global_th_segment", 0, 0.75)

        global_control_sim = pm.Normal("global_control_sim", 0, 0.5)
        global_under_sim = pm.Normal("global_under_sim", 0, 0.5)
        global_over_sim = pm.Normal("global_over_sim", 0, 0.5)

        global_control_dis = pm.Normal("global_control_dis", 0, 0.5)
        global_under_dis = pm.Normal("global_under_dis", 0, 0.5)
        global_over_dis = pm.Normal("global_over_dis", 0, 0.5)

        global_control_sim_seg = pm.Normal("global_control_sim_seg", 0, 0.75)
        global_under_sim_seg = pm.Normal("global_under_sim_seg", 0, 0.75)
        global_over_sim_seg = pm.Normal("global_over_sim_seg", 0, 0.75)

        global_control_dis_seg = pm.Normal("global_control_dis_seg", 0, 0.75)
        global_under_dis_seg = pm.Normal("global_under_dis_seg", 0, 0.75)
        global_over_dis_seg = pm.Normal("global_over_dis_seg", 0, 0.75)

        # level 2

        # group_intercept_sigma = pm.HalfStudentT("group_intercept_sigma", 1, 3)
        # group_th_segments_sigma = pm.HalfStudentT("group_th_segments_sigma", 1, 3)

        group_intercept = pm.Normal("group_intercept", 0, 1, dims="ids")
        group_th_segments = pm.Normal("group_th_segments", 0, 0.15, dims="ids")

        # likelihood
        growth_model = pm.Deterministic(
            "growth_model",

            (global_intercept + group_intercept[session_id_idx])

            + global_control_sim * control_sim
            + global_under_sim * under_sim
            + global_over_sim * over_sim

            + global_control_dis * control_dis
            + global_under_dis * under_dis
            + global_over_dis * over_dis

            + global_control_sim_seg * (control_sim * th_segments)
            + global_under_sim_seg * (under_sim * th_segments)
            + global_over_sim_seg * (over_sim * th_segments)

            + global_control_dis_seg * (control_dis * th_segments)
            + global_under_dis_seg * (under_dis * th_segments)
            + global_over_dis_seg * (over_dis * th_segments)

            + (global_th_segment + group_th_segments[session_id_idx]) * th_segments

            , dims="obs"
        )

        if BINOMINAL:

            outcome = pm.Binomial("y", n=n, p=pm.math.invlogit(growth_model), observed=df[analyzed_features].values,
                                  dims="obs")

        else:
            # global_sigma = pm.HalfStudentT("global_sigma", 1, 3)
            outcome = pm.Normal("y", mu=pm.math.invlogit(growth_model), sigma=0.5,
                                observed=df[analyzed_features].values, dims="obs")

        return model


def CenteredModel(coords, df, BINOMINAL, subject_idx, analyzed_features, n):
    mu = df[analyzed_features].mean()
    with pm.Model(coords=coords) as model:
        th_segments = pm.Data("th_segments", df["th_segments"].values, mutable=True,
                              dims=("obs"))

        groups = pm.Data("groups", df[["control", "inefficient", "efficient"]].values.astype(float), mutable=True,
                         dims=("obs", "group"))

        # skilled
        controlled_skill = pm.Data("controlled_skill",
                                   df["subject_skill"].values.astype(float),
                                   mutable=True, dims=("obs"))

        observed = pm.Data("observed", df[analyzed_features].values, mutable=True, dims="obs")

        # Define priors
        # level 1
        global_intercept = pm.Normal("global_intercept", 0, 0.5)
        global_th_segment = pm.Normal("global_th_segment", 0, 0.5)


        global_group = pm.Normal("global_group", mu, 1, dims="group")
        global_group_seg = pm.Normal("global_group_seg", 0, 0.5, dims="group")



        # level 2
        # fixed effect params
        global_skill_slope = pm.Normal('global_skill_slope', mu=0, sigma=0.1)
        subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=0.1, dims=("subject_idx"))
        subjects_intercept_seg = pm.Normal('subjects_intercept_seg', mu=0, sigma=0.1, dims=("subject_idx"))

        control_skilled_variable = pm.Deterministic("control_skilled_variable",
                                                    pm.math.dot(controlled_skill, global_skill_slope))

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
                              + (control_skilled_variable))

        if BINOMINAL:


            growth_model = pm.Binomial(
                "growth_model",
                n=n,
                p=pm.math.invlogit(mu),
                observed=observed,
                dims="obs"
            )
            # alpha = pm.Exponential("alpha", 0.1)
            # growth_model = pm.NegativeBinomial("growth_model", mu=pm.math.invlogit(mu), alpha=alpha, observed=observed, dims="obs")
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

#
# def CenteredPolyModel(coords, df, BINOMINAL, subject_idx, analyzed_features, n):
#     mu = df[analyzed_features].mean()
#     sigma = df[analyzed_features].std() * 2
#     with pm.Model(coords=coords) as model:
#         th_segments = pm.Data("th_segments", df[["th_segments", "th_segments", "th_segments"]].values, mutable=True,
#                               dims=("obs", "group"))
#
#         th_segments2  = pm.Data("th_segments", df[["th_segments", "th_segments", "th_segments"]].values ** 2, mutable=True,
#                               dims=("obs", "group"))
#
#         groups = pm.Data("groups", df[["control", "inefficient", "efficient"]].values.astype(float), mutable=True,
#                          dims=("obs", "group"))
#
#         # skilled
#         controlled_skill = pm.Data("controlled_skill",
#                                    df[["subject_skill", "subject_skill", "subject_skill"]].values.astype(float),
#                                    mutable=True, dims=("obs", "group"))
#
#         observed = pm.Data("observed", df[analyzed_features].values, mutable=True, dims="obs")
#
#         # Define priors
#         # level 1
#         global_intercept = pm.Normal("global_intercept", 0, 0.1, dims="group")
#         global_th_segment = pm.Normal("global_th_segment", 0, 0.1, dims="group")
#         global_th_segment2 = pm.Normal("global_th_segment2", 0, 0.1, dims="group")
#
#         global_group = pm.Normal("global_group", mu, 0.1, dims="group")
#         global_group_seg = pm.Normal("global_group_seg", 0, 0.1, dims="group")
#         global_group_seg2 = pm.Normal("global_group_seg2", 0, 0.1, dims="group")
#
#         global_skill_slope = pm.Normal('global_skill_slope', mu=0, sigma=0.1)
#
#         # level 2
#         # fixed effect params
#         subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=0.1, dims=("subject_idx", "group"))
#         subjects_intercept_seg = pm.Normal('subjects_intercept_seg', mu=0, sigma=0.1, dims=("subject_idx", "group"))
#         subjects_intercept_seg2 = pm.Normal('subjects_intercept_seg2', mu=0, sigma=0.1, dims=("subject_idx", "group"))
#
#         control_skilled_variable = pm.Deterministic("control_skilled_variable",
#                                                     pm.math.dot(controlled_skill, global_skill_slope))
#
#         if BINOMINAL:
#             mu = pm.Deterministic("mu", (global_intercept + subjects_intercept[subject_idx])
#                                   + global_group * groups
#                                   + (global_group_seg * th_segments * groups)
#                                   + (global_group_seg2 * th_segments2 * groups) # poly
#                                   + ((global_th_segment + subjects_intercept_seg[subject_idx]) * th_segments)
#                                   + ((global_th_segment2 + subjects_intercept_seg2[subject_idx]) * th_segments2) # poly
#                                   + (control_skilled_variable))
#
#             components = [
#                 pm.Binomial.dist(n=n, p=pm.math.invlogit(mu[:, 0])),
#                 pm.Binomial.dist(n=n, p=pm.math.invlogit(mu[:, 1])),
#                 pm.Binomial.dist(n=n, p=pm.math.invlogit(mu[:, 2])),
#             ]
#
#             weights = pm.Dirichlet("w", np.ones(3), dims="group")
#             growth_model = pm.Mixture(
#                 "growth_model",
#                 comp_dists=components,
#                 w=weights,
#                 observed=observed,
#                 dims="obs"
#             )
#         else:
#             mu = pm.Deterministic("mu", (global_intercept + subjects_intercept[subject_idx])
#                                   + global_group * groups
#                                   + (global_group_seg * th_segments * groups)
#                                   + ((global_th_segment + subjects_intercept_seg[subject_idx]) * th_segments)
#                                   + (control_skilled_variable))
#
#             # likelihood
#             global_sigma = pm.HalfNormal("global_sigma", 0.1)
#             weights = pm.Dirichlet("w", np.ones(3), dims="group")
#             growth_model = pm.NormalMixture(
#                 "growth_model",
#                 mu=mu,
#                 w=weights,
#                 sigma=global_sigma,
#                 observed=observed,
#                 dims="obs"
#             )
#
#     return model


def NonCenteredModel(coords, df, BINOMINAL, session_id_idx, analyzed_features, n):
    with pm.Model(coords=coords) as model:
        th_segments = pm.Data("th_segments", df["th_segments"].values, mutable=True)

        control = pm.Data("control", df["control"].values.astype(float), mutable=True)
        over = pm.Data("over", df["over"].values.astype(float), mutable=True)
        under = pm.Data("under", df["under"].values.astype(float), mutable=True)

        # level 1
        global_th_segment_mu = pm.Normal("global_th_segment_mu", 0, 1)
        global_intercept_mu = pm.Normal("global_intercept_mu", 0, 1)

        global_intercept_tilde = pm.Normal("global_intercept_tilde", 0, 1)
        global_th_segment_tilde = pm.Normal("global_th_segment_tilde", 0, 1)

        global_intercept_sigma = pm.HalfNormal("global_intercept_sigma", 1)
        global_th_segment_sigma = pm.HalfNormal("global_th_segment_sigma", 1)

        global_intercept = pm.Deterministic("global_intercept",
                                            global_intercept_mu + global_intercept_tilde * global_intercept_sigma)
        global_th_segment = pm.Deterministic("global_th_segment",
                                             global_th_segment_mu + global_th_segment_tilde * global_th_segment_sigma)

        # beta for groups
        global_control = pm.Normal("global_control", 0, 1)
        global_under = pm.Normal("global_under", 0, 1)
        global_over = pm.Normal("global_over", 0, 1)

        # beta for segments

        global_control_seg_mu = pm.Normal("global_control_seg_mu", 0, 1)
        global_under_seg_mu = pm.Normal("global_under_seg_mu", 0, 1)
        global_over_seg_mu = pm.Normal("global_over_seg_mu", 0, 1)

        global_control_seg_tilde = pm.Normal("global_control_seg_tilde", 0, 1)
        global_under_seg_tilde = pm.Normal("global_under_seg_tilde", 0, 1)
        global_over_seg_tilde = pm.Normal("global_over_seg_tilde", 0, 1)

        global_control_seg_sigma = pm.HalfNormal("global_control_seg_sigma", 1)
        global_under_seg_sigma = pm.HalfNormal("global_under_seg_sigma", 1)
        global_over_seg_sigma = pm.HalfNormal("global_over_seg_sigma", 1)

        global_control_seg = pm.Deterministic("global_control_seg",
                                              global_control_seg_mu + global_control_seg_tilde * global_control_seg_sigma)
        global_under_seg = pm.Deterministic("global_under_seg",
                                            global_under_seg_mu + global_under_seg_tilde * global_under_seg_sigma)
        global_over_seg = pm.Deterministic("global_over_seg",
                                           global_over_seg_mu + global_over_seg_tilde * global_over_seg_sigma)

        # level 2
        group_intercept_mu = pm.Normal("group_intercept_mu", 0, 1, dims="ids")
        group_th_segments_mu = pm.Normal("group_th_segments_mu", 0, 1, dims="ids")

        group_intercept_tilde = pm.Normal("group_intercept_tilde", 0, 1)
        group_th_segments_tilde = pm.Normal("group_th_segments_tilde", 0, 1)

        group_intercept_sigma = pm.HalfNormal("group_intercept_sigma", 1)
        group_th_segments_sigma = pm.HalfNormal("group_th_segments_sigma", 1)

        group_intercept = pm.Deterministic("group_intercept",
                                           group_intercept_mu + group_intercept_tilde * group_intercept_sigma)
        group_th_segments = pm.Deterministic("group_th_segments",
                                             group_th_segments_mu + group_th_segments_tilde * group_th_segments_sigma)

        # likelihood
        growth_model = pm.Deterministic(
            "growth_model",

            (global_intercept + group_intercept[session_id_idx])
            + global_control * control
            + global_under * under
            + global_over * over
            + global_control_seg * (control * th_segments)
            + global_under_seg * (under * th_segments)
            + global_over_seg * (over * th_segments)
            + (global_th_segment + group_th_segments[session_id_idx]) * th_segments

            , dims="obs"
        )
        if BINOMINAL:

            outcome = pm.Binomial("y", n=n, p=pm.math.invlogit(growth_model), observed=df[analyzed_features].values,
                                  dims="obs")

        else:
            global_sigma = pm.HalfStudentT("global_sigma", 1, 3)
            outcome = pm.Normal("y", growth_model, global_sigma, observed=df[analyzed_features].values, dims="obs")

    return model
