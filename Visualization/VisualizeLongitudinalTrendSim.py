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


# a = az.summary(
#     idata,
#     stat_focus="median",
# )


# az.plot_trace(idata)
# plt.show()


fig, ax = plt.subplots(figsize=(20, 8))
posterior = az.extract(idata.posterior)

group_intercept = posterior["group_intercept"].mean(dim="ids")
group_th_segments = posterior["group_th_segments"].mean(dim="ids")

a = posterior["global_intercept"].mean() + group_intercept
b = posterior["global_th_segment"].mean() + group_th_segments


global_control_sim = posterior["global_control_sim"].mean()
global_under_sim = posterior["global_under_sim"].mean()
global_over_sim = posterior["global_over_sim"].mean()

global_control_dis = posterior["global_control_dis"].mean()
global_under_dis = posterior["global_under_dis"].mean()
global_over_dis = posterior["global_over_dis"].mean()

global_control_sim_seg = posterior["global_control_sim_seg"].mean()
global_under_sim_seg = posterior["global_under_sim_seg"].mean()
global_over_sim_seg = posterior["global_over_sim_seg"].mean()


global_control_dis_seg = posterior["global_control_dis_seg"].mean()
global_under_dis_seg = posterior["global_under_dis_seg"].mean()
global_over_dis_seg = posterior["global_over_dis_seg"].mean()

time_xi = xr.DataArray(np.arange(0, 10, 0.1))

# control-sim
ax.plot(
    time_xi,
    (a + b * time_xi + global_control_sim * 1 + global_control_sim_seg * (time_xi * 1)).T,
    color="#08519c",
    linewidth=0.2,
    alpha=0.2,
)

# control-dis
ax.plot(
    time_xi,
    (a + b * time_xi + global_control_dis * 1 + global_control_dis_seg * (time_xi * 1)).T,
    color="#6baed6",
    linewidth=0.2,
    alpha=0.2,
)
#under-sim
ax.plot(
    time_xi,
    (a + b * time_xi + global_under_sim * 1 + global_under_sim_seg * (time_xi * 1)).T,
    color="#006d2c",
    linewidth=0.2,
    alpha=0.2,
)
#under-dis
ax.plot(
    time_xi,
    (a + b * time_xi + global_under_dis * 1 + global_under_dis_seg * (time_xi * 1)).T,
    color="#74c476",
    linewidth=0.2,
    alpha=0.2,
)
# over-sim
ax.plot(
    time_xi,
    (a + b * time_xi + global_over_sim * 1 + global_over_sim_seg * (time_xi * 1)).T,
    color="#a50f15",
    linewidth=0.2,
    alpha=0.2,
)
# over-dis
ax.plot(
    time_xi,
    (a + b * time_xi + global_over_dis * 1 + global_over_dis_seg * (time_xi * 1)).T,
    color="#fb6a4a",
    linewidth=0.2,
    alpha=0.2,
)

plt.show()