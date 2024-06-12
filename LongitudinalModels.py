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
            global_sigma = pm.HalfStudentT("global_sigma", 1, 3)
            outcome = pm.Normal.dist(growth_model, global_sigma)
            censored_normal = pm.Censored("y", outcome, lower=0, upper=1,  observed=df[analyzed_features].values, dims="obs")
            # outcome = pm.TruncatedNormal("y", growth_model, global_sigma, observed=df[analyzed_features].values, dims="obs", lower=0.0, upper=1.0)

        return model


def CenteredModel(coords, df, BINOMINAL, session_id_idx, analyzed_features, n):
    with pm.Model(coords=coords) as model:
        th_segments = pm.Data("th_segments", df["th_segments"].values, mutable=True)

        control = pm.Data("control", df["control"].values.astype(float), mutable=True)
        over = pm.Data("over", df["over"].values.astype(float), mutable=True)
        under = pm.Data("under", df["under"].values.astype(float), mutable=True)

        # level 1

        global_intercept = pm.Normal("global_intercept", 0, 0.75)
        global_th_segment = pm.Normal("global_th_segment", 0, 0.75)

        global_control = pm.Normal("global_control", 0, 0.5)
        global_under = pm.Normal("global_under", 0, 0.5)
        global_over = pm.Normal("global_over", 0, 0.5)

        global_control_seg = pm.Normal("global_control_seg", 0, 0.75)
        global_under_seg = pm.Normal("global_under_seg", 0, 0.75)
        global_over_seg = pm.Normal("global_over_seg", 0, 0.75)

        # level 2

        # group_intercept_sigma = pm.HalfNormal("group_intercept_sigma", 5)
        # group_th_segments_sigma = pm.HalfNormal("group_th_segments_sigma", 5)

        # group_intercept_sigma = pm.HalfStudentT("group_intercept_sigma", 1, 3)
        # group_th_segments_sigma = pm.HalfStudentT("group_th_segments_sigma", 1, 3)

        group_intercept = pm.Normal("group_intercept", 0, 1, dims="ids")
        group_th_segments = pm.Normal("group_th_segments", 0, 0.15, dims="ids")

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
