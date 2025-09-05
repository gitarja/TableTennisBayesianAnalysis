import pymc as pm
import pytensor.tensor as pt
import numpy as np

eps = np.finfo(float).eps


def CenteredModel(coords, df, hitters_idx, receivers_idx, session_idx, analyzed_features, n=0, BINOMINAL=False, hitter=False):
    print(np.nanmean(df.loc[df.efficient == True, analyzed_features].values))
    print(np.nanmean(df.loc[df.efficient == False, analyzed_features].values))

    mu_m = np.nanmean(df[analyzed_features].values)





    with pm.Model(coords=coords) as model:
        th_segments = pm.Data("th_segments", df["th_segments"].values)
        efficient = pm.Data("efficient", df["efficient"].values.astype(float))
        inefficient = pm.Data("inefficient", df["inefficient"].values.astype(float))
        observed = pm.Data("observed", df[analyzed_features].values, dims="obs")


        # Define priors
        # level 1
        if BINOMINAL:
            global_intercept = pm.HalfNormal("global_intercept", 0.1)

            global_th_segment = pm.HalfNormal("global_th_segment", 0.5)
            global_efficient = pm.HalfNormal("global_efficient", 0.5)
            global_efficient_seg = pm.HalfNormal("global_efficient_seg",  0.5)

            global_inefficient = pm.HalfNormal("global_inefficient",   0.5)
            global_inefficient_seg = pm.HalfNormal("global_inefficient_seg",   0.5 )


            # level 2
            # fixed effect params
            subjects_intercept = pm.HalfNormal('subjects_intercept', sigma=0.1, dims=("subject_idx"))
            sessions_intercept = pm.HalfNormal('sessions_intercept', sigma=0.1, dims=("session_idx"))




        else:
            global_intercept = pm.Normal("global_intercept", 0, 0.1)

            global_th_segment = pm.Normal("global_th_segment", 0, 0.5)

            global_efficient = pm.Normal("global_efficient", mu_m,  0.5)
            global_efficient_seg = pm.Normal("global_efficient_seg", 0,  0.5)



            global_inefficient = pm.Normal("global_inefficient", mu_m,  0.5)
            global_inefficient_seg = pm.Normal("global_inefficient_seg",0,  0.5)



            # level 2
            # fixed effect params
            subjects_intercept = pm.Normal('subjects_intercept', mu=0, sigma=0.1, dims=("subject_idx"))
            sessions_intercept = pm.Normal('sessions_intercept', mu=0, sigma=0.1, dims=("session_idx"))




        # control_skilled_variable = pm.Deterministic("control_skilled_variable", controlled_skill * global_skill_slope)
        
        # difference
        global_diff_of_means = pm.Deterministic("global_diff_of_means", global_efficient - global_inefficient)


        mu = pm.Deterministic("mu",
                              (global_intercept + sessions_intercept[session_idx] + (subjects_intercept[hitters_idx] + subjects_intercept[receivers_idx])/2)
                              # efficient * seg
                               + (global_efficient * efficient)
                              + (global_efficient_seg * (efficient * th_segments))

                              # inefficient * seg
                              + (global_inefficient * inefficient)
                              + (global_inefficient_seg * (inefficient * th_segments))


                              + (global_th_segment  * th_segments)


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
            a = pm.Gamma("a", alpha=2, beta=0.5)  # controls tails, higher = lighter tail
            b = pm.Gamma("b", alpha=2, beta=0.5)  # controls skewness
            global_sigma = pm.HalfCauchy("global_sigma", 1.0)
            growth_model = pm.SkewStudentT("growth_model", a=a, b=b, mu=mu,
                                    sigma=global_sigma, observed=observed, dims="obs",)



    return model



