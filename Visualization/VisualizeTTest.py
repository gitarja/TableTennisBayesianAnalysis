import arviz as az
from Utils.Conf import DOUBLE_RESULTS_PATH_TTEST
import matplotlib.pyplot as plt
import pickle

analyzed_features = "hitter_pursuit"
with open(DOUBLE_RESULTS_PATH_TTEST + "model\\" + "idata_m3_" + analyzed_features + ".pkl", 'rb') as handle:
    idata = pickle.load(handle)

# inspect convergence
az.plot_trace(idata, var_names=("control_mean", "over_mean", "under_mean"))

plt.show()

# poseterior predictive check
az.plot_ppc(idata, num_pp_samples=100, kind="cumulative")

plt.show()


az.plot_posterior(
    idata,
    var_names=["diff_means_under_control", "diff_means_over_control", "diff_means_under_over", "diff_stds_under_control", "diff_stds_over_control", "diff_stds_under_over"],
    ref_val=0,
    color="#87ceeb",
    hdi_prob=0.95,

)



plt.show()
