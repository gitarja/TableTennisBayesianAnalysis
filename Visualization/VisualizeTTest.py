import arviz as az
from Utils.Conf import DOUBLE_RESULTS_PATH_TTEST
import matplotlib.pyplot as plt
import pickle



analyzed_features = "stable_percentage"
with open(DOUBLE_RESULTS_PATH_TTEST + "model\\" + "idata_m3_" + analyzed_features + ".pkl", 'rb') as handle:
    idata = pickle.load(handle)

# # inspect convergence
az.plot_trace(idata, var_names=("control_mean", "inefficient_mean", "efficient_mean"))

plt.show()

# poseterior predictive check
az.plot_ppc(idata, num_pp_samples=500, kind="cumulative")

plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "Mean_std_diff\\" + "ppc_" + analyzed_features + ".png")


az.plot_posterior(
    idata,
    var_names=["diff_means_efficient_control", "diff_means_inefficient_control", "diff_means_efficient_inefficient", "diff_stds_efficient_control", "diff_stds_inefficient_control", "diff_stds_efficient_inefficient"],
    ref_val=0,
    color="#87ceeb",
    hdi_prob=0.95,

)


plt.savefig(DOUBLE_RESULTS_PATH_TTEST + "Mean_std_diff\\" + "post_dist_" + analyzed_features + ".png")
plt.close()
