import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)

# Group sizes
n_group1 = 50
n_group2 = 50

# True parameters
true_mu1 = 2.5
true_mu2 = 3.0
true_sigma = 1.0

# Generate latent continuous variables
latent_group1 = np.random.normal(true_mu1, true_sigma, n_group1)
latent_group2 = np.random.normal(true_mu2, true_sigma, n_group2)


# Convert to ordinal outcomes (5-point Likert scale)
def convert_to_ordinal(latent, cutpoints):
    return np.sum(latent[:, None] > cutpoints, axis=1)


# Cutpoints for the ordered logistic model
cutpoints = np.array([-1.5, -0.5, 0.5, 1.5])

# Generate ordinal data
y_group1 = convert_to_ordinal(latent_group1, cutpoints)
y_group2 = convert_to_ordinal(latent_group2, cutpoints)

# Combine data
y = np.concatenate([y_group1, y_group2])
group = np.concatenate([np.zeros(n_group1), np.ones(n_group2)])

# Ordered logistic regression model
with pm.Model() as ordered_logistic_model:
    # Priors
    mu = pm.Normal("mu", mu=0, sigma=2, shape=2)  # group means
    sigma = pm.HalfNormal("sigma", sigma=1)  # common standard deviation

    # Cutpoints for the ordered logistic (must be ordered)
    cutpoints = pm.Normal(
        "cutpoints",
        mu=[-1, 0, 1],
        sigma=1,
        shape=3,
        transform=pm.distributions.transforms.ordered
    )

    # Linear predictor
    eta = mu[group.astype(int)]

    # Ordered logistic likelihood
    y_obs = pm.OrderedLogistic(
        "y_obs",
        eta=eta,
        cutpoints=cutpoints,
        observed=y
    )

    # Effect size (standardized difference between groups)
    delta = pm.Deterministic("delta", (mu[1] - mu[0]) / sigma)

    # Sampling
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9)

# Diagnostics
az.summary(trace, var_names=["mu", "sigma", "delta"])
az.plot_trace(trace, var_names=["mu", "sigma", "delta"])
plt.show()

# Posterior plot of the difference
az.plot_posterior(trace, var_names=["delta"], ref_val=0)
plt.title("Posterior distribution of effect size (delta)")
plt.show()

# Probability that group 2 > group 1
prob = (trace.posterior["mu"][1] > trace.posterior["mu"][0]).mean()
print(f"Probability that group 2 > group 1: {prob:.3f}")