from Utils.Conf import DOUBLE_RESULTS_PATH, DOUBLE_FEATURES_FILE_PATH
import pickle
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import numpy as np
# save the model
n=10
analyzed_features = "forward_swing_sim"
with open(DOUBLE_RESULTS_PATH + "idata_m3_" + analyzed_features + "_" + str(n) + ".pkl", 'rb') as handle:
    idata = pickle.load(handle)

fig, ax = plt.subplots(figsize=(20, 8))
posterior = az.extract(idata.posterior)

group_intercept_raw = posterior["group_intercept_raw"].mean(dim="ids")
group_th_segments_raw = posterior["group_th_segments_raw"].mean(dim="ids")

group_intercept_mu = posterior["group_intercept_mu"].mean()
group_intercept_sigma = posterior["group_intercept_sigma"].mean()

group_th_segments_mu = posterior["group_th_segments_mu"].mean()
group_th_segments_sigma = posterior["group_th_segments_sigma"].mean()


group_intercept = group_intercept_mu + group_intercept_raw * group_intercept_sigma
group_th_segments = group_th_segments_mu + group_th_segments_raw * group_intercept_sigma

global_intercept_raw = posterior["global_intercept_raw"].mean()
global_th_segment_raw = posterior["global_th_segment_raw"].mean()

global_intercept_mu = posterior["global_intercept_mu"].mean()
global_intercept_sigma = posterior["global_intercept_sigma"].mean()

global_th_segment_mu = posterior["global_th_segment_mu"].mean()
global_th_segment_sigma = posterior["global_th_segment_sigma"].mean()

a = (global_intercept_mu + global_intercept_raw * global_intercept_sigma) + group_intercept
b = (global_th_segment_mu + global_th_segment_raw * global_th_segment_sigma) + group_th_segments


global_control = posterior["global_control"].mean()
global_under = posterior["global_under"].mean()
global_over = posterior["global_over"].mean()

global_control_seg = posterior["global_control_seg"].mean()
global_under_seg = posterior["global_under_seg"].mean()
global_over_seg = posterior["global_over_seg"].mean()

time_xi = xr.DataArray(np.arange(0, 8, 0.1))

# control
ax.plot(
    time_xi,
    (a + b * time_xi + global_control * 1 + global_control_seg * (time_xi * 1)).T,
    color="#377eb8",
    linewidth=0.2,
    alpha=0.2,
)
#under
ax.plot(
    time_xi,
    (a + b * time_xi + global_under * 1 + global_under_seg * (time_xi * 1)).T,
    color="#4daf4a",
    linewidth=0.2,
    alpha=0.2,
)
# over
ax.plot(
    time_xi,
    (a + b * time_xi + global_over * 1 + global_over_seg * (time_xi * 1)).T,
    color="#e41a1c",
    linewidth=0.2,
    alpha=0.2,
)

plt.show()