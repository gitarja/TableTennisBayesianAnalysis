import arviz as az
from Utils.Conf import DOUBLE_RESULTS_PATH_ANOVA
import matplotlib.pyplot as plt
import pickle

analyzed_features = "recover_movement_sim"
with open(DOUBLE_RESULTS_PATH_ANOVA + "model\\" + "idata_m3_transition" + analyzed_features + ".pkl", 'rb') as handle:
    idata = pickle.load(handle)

# plot r-hat
# plot rhat
# nc_rhat = az.rhat(idata)
# ax = (nc_rhat.max()
#           .to_array()
#           .to_series()
#           .plot(kind="barh"))
# plt.show()

# az.plot_trace(idata, var_names=("diff_means_efficient_inefficient"))


# az.plot_posterior(
#     idata,
#     var_names=["diff_meansdiff_efficient", "diff_meansdiff_inefficient", "diff_meansdiff_control", "effect_efficient", "effect_inefficient", "effect_control"],
#     ref_val=0,
#     color="#4eaf49",
#     hdi_prob=0.95,
#     ref_val_color="#252525",
#     grid=(2, 3)
#
# )
# results_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double\\Bayesian-anova\\posterior\\transition\\"
# plt.savefig(results_path + analyzed_features + "_detail.eps", format="eps")

# az.plot_posterior(
#     idata,
#     var_names=["diff_means_efficient_inefficient", "effect_efficient_inefficient"],
#     ref_val=0,
#     color="#87ceeb",
#     hdi_prob=0.95,
#     grid=(1, 2)
#
#
# )
plt.show()

# poseterior predictive check
# az.plot_ppc(idata, num_pp_samples=500, kind="cumulative")
#
# plt.show()
