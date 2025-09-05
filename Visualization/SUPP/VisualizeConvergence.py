import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (2, 4)


results_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double\\Bayesian-ttest\\"

df = pd.read_csv(os.path.join(results_path, "convergence_t-test-all.csv"))

features = [
"ess_bulk",
"ess_tail",
"r_hat"
]


fig, axes = plt.subplots(1, 3, figsize=(8.27, 2.94))
i = 0
for f in features:

    ax = axes[i]

    sns.stripplot(
        data=df, x="parameter", y=f, hue="parameter",
        dodge=True, alpha=.2, legend=False,  ax=ax
    )
    sns.pointplot(
        data=df, x="parameter", y=f, hue="parameter",
        dodge=.1, linestyle="none", errorbar=('ci', 95), capsize=.1,
        markersize=3, markeredgewidth=3, ax=ax, linewidth=0.75,  palette="dark"
    )
    ax.set_title(f, fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=8)

    i += 1

plt.show()