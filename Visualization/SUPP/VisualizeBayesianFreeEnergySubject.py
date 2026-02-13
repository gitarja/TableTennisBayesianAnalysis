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

# show the weighting
with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_regression\\" + "idata_ARFE.pkl", 'rb') as handle:
    idata = pickle.load(handle)



idata.posterior = idata.posterior.assign(
    w_diff=idata.posterior["w"].mean(dim="event_idx")[:, :, 0] - idata.posterior["w"].mean(dim="event_idx")[:, :, 1]   # shape: (chain, draw, group_idx)
)
print(np.median(idata.posterior["w_diff"]))
print(az.hdi(idata, var_names=["w_diff"], hdi_prob=0.95))
ax = az.plot_forest(idata, figsize=(3.5,  3), var_names="w_diff", kind="forestplot", combined=True, hdi_prob=0.95,coords={"group_idx": ["efficient", "inefficient"]}

)

results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\all_lower_upper\\"
plt.savefig(os.path.join(results_path, "ProactiveReactive", "posterior_w_avg_diff.pdf"))


idata.posterior = idata.posterior.assign(
    w_subject_avg=idata.posterior["w_subject"].mean(dim="event_idx")   # shape: (chain, draw, group_idx)
)
ax = az.plot_forest(idata, figsize=(3.5,  3), var_names="w_subject_avg", kind="forestplot", combined=True, hdi_prob=0.95,coords={"group_idx": ["efficient", "inefficient"]}

)

results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\all_lower_upper\\"
plt.savefig(os.path.join(results_path, "ProactiveReactive", "posterior_w_subject.pdf"))
# plt.show()


