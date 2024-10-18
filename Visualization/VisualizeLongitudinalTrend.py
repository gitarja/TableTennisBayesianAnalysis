from Utils.Conf import DOUBLE_RESULTS_PATH, DOUBLE_FEATURES_FILE_PATH
import pickle
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import numpy as np
from scipy.special import expit

# save the model
n = 5

analyzed_features = "hand_mov_sim"

with open(DOUBLE_RESULTS_PATH + "idata_m3_" + analyzed_features + "_" + str(n) + ".pkl", 'rb') as handle:
    idata = pickle.load(handle)

# compared posterior to observed
az.plot_ppc(idata)
plt.show()

posterior = az.extract(idata.posterior)

global_group = posterior["global_group"].mean(dim="sample")
global_group_seg = posterior["global_group_seg"].mean(dim="sample")

global_skill_slope = posterior["global_skill_slope"].mean() * 0.5
subjects_intercept = posterior["subjects_intercept"].mean(dim="subject_idx")
subjects_intercept_seg = posterior["subjects_intercept_seg"].mean(dim="subject_idx")

global_intercept = posterior["global_intercept"].mean() + subjects_intercept
global_th_segment = posterior["global_th_segment"].mean() + subjects_intercept_seg

if "poly" in analyzed_features:
    global_group_seg2 = posterior["global_group_seg2"].mean(dim="sample")
    subjects_intercept_seg2 = posterior["subjects_intercept_seg2"].mean(dim="subject_idx")
    global_th_segment2 = posterior["global_th_segment2"].mean() + subjects_intercept_seg2

n = 250
time_xi = xr.DataArray(np.arange(n) / 100)
time_xi_2 = (time_xi ** 2)
# plot line
fig, ax = plt.subplots(figsize=(20, 8))
if "poly" in analyzed_features:
    y = global_intercept + global_group + global_group_seg * time_xi + global_th_segment * time_xi + global_group_seg2 * time_xi_2 + global_th_segment2 * time_xi_2 + global_skill_slope
else:
    y = global_intercept + global_group + global_group_seg * time_xi + global_th_segment * time_xi + global_skill_slope

az.plot_hdi(
    time_xi,
    (y[:, 0]).values.reshape(4, 2000, n),
    hdi_prob=0.95,
    fill_kwargs={"alpha": 0.25, "linewidth": 0},
    color="#377eb8",
)

az.plot_hdi(
    time_xi,
    (y[:, 1]).values.reshape(4, 2000, n),
    hdi_prob=0.95,
    fill_kwargs={"alpha": 0.25, "linewidth": 0},
    color="#e41a1c",
)

az.plot_hdi(
    time_xi,
    (y[:, 2]).values.reshape(4, 2000, n),
    hdi_prob=0.95,
    fill_kwargs={"alpha": 0.25, "linewidth": 0},
    color="#4daf4a",
)
plt.show()
if "poly" in analyzed_features:
    az.plot_posterior(
        idata,
        var_names=["global_group", "global_group_seg", "global_group_seg2"],
        ref_val=0,
        color="#4eaf49",
        hdi_prob=0.95,
        ref_val_color="#252525",
        grid=(3, 3)

    )
else:
    az.plot_posterior(
        idata,
        var_names=["global_group", "global_group_seg"],
        ref_val=0,
        color="#4eaf49",
        hdi_prob=0.95,
        ref_val_color="#252525",
        grid=(2, 3)

    )
plt.show()

# ax.plot(
#     time_xi,
#     expit(y[:, 0, :]).T,
#     color="#377eb8",
#     linewidth=0.2,
#     alpha=0.2,
# )
#
# ax.plot(
#     time_xi,
#     expit(y[:, 1, :]).T,
#     color="#e41a1c",
#     linewidth=0.2,
#     alpha=0.2,
# )
#
# ax.plot(
#     time_xi,
#     expit(y[:, 2, :]).T,
#     color="#4daf4a",
#     linewidth=0.2,
#     alpha=0.2,
# )
#


