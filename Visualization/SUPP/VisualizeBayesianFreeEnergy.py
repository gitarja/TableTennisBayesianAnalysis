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
# # compare model
# with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_regression\\" + "idata_reactive.pkl", 'rb') as handle:
#     idata_reactive = pickle.load(handle)
# with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_regression\\" + "idata_proactive.pkl", 'rb') as handle:
#     idata_proactive = pickle.load(handle)
# with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_regression\\" + "idata_proactive_reactive.pkl", 'rb') as handle:
#     idata_proactive_reactive = pickle.load(handle)
#
# model_compare = az.compare({
#     'Reactive': idata_reactive,
#     'Proactive': idata_proactive,
#     'Proactive and reactive': idata_proactive_reactive,
# })
# az.plot_compare(model_compare,  figsize=(6, 3.5), plot_ic_diff=False, legend=False)
# results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\all_lower_upper\\"
# plt.savefig(os.path.join(results_path, "ProactiveReactive", "model_comparison.pdf"))
# plt.show()


# show the weighting
with open(DOUBLE_RESULTS_PATH_TTEST + "\\model_regression\\" + "idata_proactive_reactive.pkl", 'rb') as handle:
    idata = pickle.load(handle)


posterior = idata["posterior"]
w = posterior["w"]
efficient_list = []
efficient_upper_list = []
efficient_lower_list = []
inefficient_list = []
inefficient_upper_list = []
inefficient_lower_list = []
for i in range(2): # group-level
    for j in range(3): #  segment level
        w_ij = w[:, :, i, j]
        hdi = az.hdi(w_ij, 0.95)["w"].data
        mean = np.median(w_ij)
        if i == 0:
            efficient_list.append(mean)
            efficient_lower_list.append(mean - hdi[0])
            efficient_upper_list.append(hdi[1] - mean)

        else:
            inefficient_list.append(mean)
            inefficient_lower_list.append(mean - hdi[0])
            inefficient_upper_list.append(hdi[1] - mean)



x =np.array(["early", "middle", "late"])
eff_yerr = np.vstack([efficient_lower_list, efficient_upper_list])
ineff_yerr = np.vstack([inefficient_lower_list, inefficient_upper_list])

plt.rcParams["figure.figsize"] = (6,  3.5)
plt.errorbar(x, efficient_list, yerr=eff_yerr, marker='o', capsize=0,  color='#68a880', linestyle="none",  mfc='w')
plt.errorbar(x, inefficient_list, yerr=ineff_yerr, marker='o', capsize=0,  color='#b5202d', linestyle="none", mfc='w')


plt.ylabel("Posterior weighting")
results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\all_lower_upper\\"
plt.savefig(os.path.join(results_path, "ProactiveReactive", "posterior_w.pdf"))
# plt.show()