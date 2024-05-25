from Utils.Conf import DOUBLE_RESULTS_PATH, DOUBLE_FEATURES_FILE_PATH
import pickle
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import numpy as np
# save the model
n=10
analyzed_features = "racket_mov_sim"
with open(DOUBLE_RESULTS_PATH + "idata_m3_" + analyzed_features + "_" + str(n) + ".pkl", 'rb') as handle:
    idata = pickle.load(handle)

fig, ax = plt.subplots(figsize=(20, 8))
posterior = az.extract(idata.posterior)

group_intercept = posterior["group_intercept"].mean(dim="ids")
group_th_segments = posterior["group_th_segments"].mean(dim="ids")

a = posterior["global_intercept"].mean() + group_intercept
b = posterior["global_th_segment"].mean() + group_th_segments


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
    color="blue",
    linewidth=0.2,
    alpha=0.2,
)
#under
ax.plot(
    time_xi,
    (a + b * time_xi + global_under * 1 + global_under_seg * (time_xi * 1)).T,
    color="green",
    linewidth=0.2,
    alpha=0.2,
)
# over
ax.plot(
    time_xi,
    (a + b * time_xi + global_over * 1 + global_over_seg * (time_xi * 1)).T,
    color="red",
    linewidth=0.2,
    alpha=0.2,
)

plt.show()