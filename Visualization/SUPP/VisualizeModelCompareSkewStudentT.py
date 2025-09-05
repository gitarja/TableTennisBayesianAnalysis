import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (2, 4)


results_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double\\Bayesian-ttest\\"

df = pd.read_csv(os.path.join(results_path, "convergence_loo_narrow_wider.csv"))

features = [
"LOO difference",

]


fig, axes = plt.subplots(1, 1, figsize=(8, 2))
i = 0
for f in features:

    ax = axes

    sns.stripplot(
        data=df, x=f, y="parameter", hue="parameter",
        dodge=True, alpha=.4, legend=False,  ax=ax,
    )
    sns.pointplot(
        data=df, x=f, y="parameter", hue="parameter",
        dodge=.5, linestyle="none", errorbar=('ci', 95), capsize=.1,
        markersize=3, markeredgewidth=3, ax=ax, linewidth=0.75,  palette="dark",
    )

    ax.set_title(f, fontsize=8)
    ax.set_xlim(-20, 1000)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=8)

    i += 1

plt.show()