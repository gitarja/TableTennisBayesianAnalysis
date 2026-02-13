import pymc as pm
import pytensor
import numpy as np
import pytensor.tensor as pt

from pymc.pytensorf import collect_default_updates

def BayesianHirModel(coords, x, y_f,  priors_mean, priors_std, session_v_idx, hitter_v_idx, group_v_idx, event_v_idx,  mode="combined"):
    with pm.Model(coords=coords) as model:
        X = pm.Data("x", x, dims=("obs", "axis"))
        YF = pm.Data("y_f", y_f, dims="obs")

        # Trial-wise priors (shared dispersion across groups for simplicity)

        prior_noise = pm.Normal("prior_noise", 0, 1)
        subject_intercept = pm.Normal("subject_intercept", priors_mean, priors_std, dims=("subject_idx"))
        session_intercept = pm.Normal("session_intercept", 0.0, 1, dims=("session_idx"))

        # prior
        priors_model = pm.Deterministic("priors_model",  prior_noise + (subject_intercept[hitter_v_idx]) + session_intercept[session_v_idx] , dims="obs")

        # latent sensory
        sense_noise = pm.Normal("sense_noise", 0, 1)
        sigma_sensory = pm.Normal("sigma_sensory", 0, 1, dims=("axis"))
        sensory_evidence = pm.Deterministic("sensory_evidence", sense_noise +  pm.math.dot(X, sigma_sensory) + session_intercept[session_v_idx] , dims="obs")


        # Weight on sensory evidence (0..1), determined by relative precisions
        # It reflects predictive coding / free energy minimization.
        # When priors are more precise (tau_prior high), the posterior sticks closer to b.
        # When sensory evidence is more precise (tau_sens high), the posterior shifts toward X.
        # Either way, the update moves to reduce the prediction error between belief and data.

        # Weight between 0 and 1

        # group- & obs-wise precisions (log-scale ensures positivity)
        log_tau_prior = pm.Normal("log_tau_prior", 0, 0.5, dims=("group_idx", "event_idx"))
        log_tau_sens = pm.Normal("log_tau_sens", 0, 0.5, dims=("group_idx", "event_idx"))
        tau_prior = pm.Deterministic("tau_prior", pm.math.exp(log_tau_prior))
        tau_sens = pm.Deterministic("tau_sens", pm.math.exp(log_tau_sens))

        # derived sensory weight = Kalman gain
        w = pm.Deterministic("w", tau_sens / (tau_prior + tau_sens), dims=("group_idx", "event_idx"))

        # pick the right group's weight per observation
        w_obs = pm.Deterministic("w_obs", w[group_v_idx, event_v_idx]) # ensure observation_v_idx aligns with "obs"

        if mode == "proactive":
            m = pm.Deterministic("m_post", 1 * priors_model + 0 * sensory_evidence, dims="obs")  # proactive
        elif mode == "reactive":
            m = pm.Deterministic("m_post", 0 * priors_model + 1 * sensory_evidence, dims="obs")  # reactive
        else:
            m = pm.Deterministic("m_post", (1.0 - w_obs) * priors_model + w_obs * sensory_evidence, dims="obs") # proactive + reactive


        # Observation noise (shared across subjects for simplicity, could be hierarchical too)
        sigma_f = pm.HalfCauchy("sigma_f", 0.5)

        # Likelihoods tying data to inference process

        a = pm.Gamma("a", alpha=2, beta=0.5)  # controls tails, higher = lighter tail
        b = pm.Gamma("b", alpha=2, beta=0.5)  # controls skewness

        y_f_like = pm.SkewStudentT("y_f_like",
                                    a=a,
                                    b=b,
                                    mu=m,
                                    sigma=sigma_f, observed=YF)


        tau_prior_diff = pm.Deterministic("tau_prior_diff", tau_prior[0] - tau_prior[1])
        tau_sens_diff = pm.Deterministic("tau_sens_diff", tau_sens[0] - tau_sens[1])

        w_eff = pm.Deterministic("tau_prior_sense_diff_eff", tau_sens[0] / (tau_prior[0] + tau_sens[0]))
        w_ineff = pm.Deterministic("tau_prior_sense_diff_ineff", tau_sens[1] / (tau_prior[1] + tau_sens[1]))

    return model


def BayesianHirSegmentModel(coords, x, y_f,  priors_mean, priors_std, session_v_idx, actor_v_idx, partner_v_idx, group_v_idx, event_v_idx):
    with pm.Model(coords=coords) as model:
        X = pm.Data("x", x, dims=("obs", "axis"))
        YF = pm.Data("y_f", y_f, dims="obs")

        # Trial-wise priors (shared dispersion across groups for simplicity)

        # hitter sigma

        w_subject_dist = pm.Normal("w_subject_dist", 0, 0.5, dims=("group_idx", "event_idx"))
        w_subject = pm.Deterministic("w_subject", pm.math.sigmoid(w_subject_dist), dims=("group_idx", "event_idx"))

        # receiver sigma

        prior_noise = pm.Normal("prior_noise", 0, 1)

        subject_intercept = pm.Normal("subject_intercept", priors_mean, priors_std, dims=("subject_idx"))
        session_intercept = pm.Normal("session_intercept", 0.0, 1, dims=("session_idx"))

        # prior
        w_subject_obs =w_subject[group_v_idx, event_v_idx]
        self_partner = pm.Deterministic("self_partner", ((1 - w_subject_obs) * subject_intercept[partner_v_idx] + w_subject_obs * subject_intercept[actor_v_idx]))
        priors_model = pm.Deterministic("priors_model",  prior_noise + self_partner + session_intercept[session_v_idx] , dims="obs")

        # latent sensory
        sense_noise = pm.Normal("sense_noise", 0, 1)
        sigma_sensory = pm.Normal("sigma_sensory", 0, 1, dims=("axis"))
        sensory_evidence = pm.Deterministic("sensory_evidence", sense_noise +  pm.math.dot(X, sigma_sensory) + session_intercept[session_v_idx] , dims="obs")


        # Weight on sensory evidence (0..1), determined by relative precisions
        # It reflects predictive coding / free energy minimization.
        # When priors are more precise (tau_prior high), the posterior sticks closer to b.
        # When sensory evidence is more precise (tau_sens high), the posterior shifts toward X.
        # Either way, the update moves to reduce the prediction error between belief and data.

        # Weight between 0 and 1

        # group- & obs-wise precisions (log-scale ensures positivity)
        log_tau_prior = pm.Normal("log_tau_prior", 0, 0.5, dims=("group_idx", "event_idx"))
        log_tau_sens = pm.Normal("log_tau_sens", 0, 0.5, dims=("group_idx", "event_idx"))
        tau_prior = pm.Deterministic("tau_prior", pm.math.exp(log_tau_prior))
        tau_sens = pm.Deterministic("tau_sens", pm.math.exp(log_tau_sens))

        # derived sensory weight = Kalman gain
        w = pm.Deterministic("w", tau_sens / (tau_prior + tau_sens), dims=("group_idx", "event_idx"))

        # pick the right group's weight per observation
        w_obs = w[group_v_idx, event_v_idx]  # ensure observation_v_idx aligns with "obs"

        m = pm.Deterministic("m_post", (1.0 - w_obs) * priors_model + w_obs * sensory_evidence, dims="obs") # proactive + reactive


        # Observation noise (shared across subjects for simplicity, could be hierarchical too)
        sigma_f = pm.HalfCauchy("sigma_f", 0.5)

        # Likelihoods tying data to inference process

        a = pm.Gamma("a", alpha=2, beta=0.5)  # controls tails, higher = lighter tail
        b = pm.Gamma("b", alpha=2, beta=0.5)  # controls skewness

        y_f_like = pm.SkewStudentT("y_f_like",
                                    a=a,
                                    b=b,
                                    mu=m,
                                    sigma=sigma_f, observed=YF)


        tau_prior_diff = pm.Deterministic("tau_prior_diff", tau_prior[0] - tau_prior[1])
        tau_sens_diff = pm.Deterministic("tau_sens_diff", tau_sens[0] - tau_sens[1])

        w_eff = pm.Deterministic("tau_prior_sense_diff_eff", tau_sens[0] / (tau_prior[0] + tau_sens[0]))
        w_ineff = pm.Deterministic("tau_prior_sense_diff_ineff", tau_sens[1] / (tau_prior[1] + tau_sens[1]))

    return model