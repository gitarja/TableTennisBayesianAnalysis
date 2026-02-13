from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_TTEST
import arviz as az
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20
# compare model
with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_regression\\" + "idata_reactive.pkl", 'rb') as handle:
    idata_reactive = pickle.load(handle)
with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_regression\\" + "idata_proactive.pkl", 'rb') as handle:
    idata_proactive = pickle.load(handle)
with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_regression\\" + "idata_combined.pkl", 'rb') as handle:
    idata_proactive_reactive = pickle.load(handle)

model_compare = az.compare({
    'Reactive': idata_reactive,
    'Proactive': idata_proactive,
    'Proactive and reactive': idata_proactive_reactive,
})
az.plot_compare(model_compare,  figsize=(4, 2), plot_ic_diff=False, legend=False)
results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\all_lower_upper\\"
plt.savefig(os.path.join(results_path, "ProactiveReactive", "model_comparison.pdf"))
# plt.show()


# show the weighting
with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_regression\\" + "idata_combined.pkl", 'rb') as handle:
    idata = pickle.load(handle)


idata.posterior = idata.posterior.assign(
    w_avg=idata.posterior["w"].mean(dim="event_idx")   # shape: (chain, draw, group_idx)
)
ax = az.plot_forest(idata, figsize=(4,  3), var_names="w_avg", kind="forestplot", combined=True, hdi_prob=0.95,coords={"group_idx": ["efficient", "inefficient"]}

)



plt.ylabel("Posterior weighting")
results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\all_lower_upper\\"
plt.savefig(os.path.join(results_path, "ProactiveReactive", "posterior_w.pdf"))
# plt.show()