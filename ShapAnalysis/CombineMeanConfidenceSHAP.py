import numpy as np
import pandas as pd
import os
from scipy.stats import t
from Utils.Conf import DOUBLE_RESULTS_PATH_SHAP
# label = "all_sf"
label = "all_lower_upper"
shap_results = np.load("Results\\Final2\\Full-model\\" + label + "_shap.npy")
xval_results = pd.read_pickle("Results\\Final2\\Full-model\\" + label + "_xval.pkl")
confidence_results =  np.load("Results\\Final2\\Full-model\\" + label + "_bootstrap_shap.npy")



avg_data = np.average(np.abs(shap_results), axis=0)
data = np.abs(confidence_results)
std_dev = np.std(data, ddof=1, axis=0)  # ddof=1 for sample standard deviation
n = len(data)

# Standard error
standard_error = std_dev / np.sqrt(n)

# t critical value (two-tailed, 95% CI)
# t_critical = t.ppf(1 - 0.025, df=n-1)
t_critical = 1.96
# Confidence interval
margin_of_error = t_critical * standard_error

df = pd.DataFrame({"features":xval_results.columns.tolist(), "mean_abs": avg_data.tolist(), "CI": margin_of_error.tolist()})
df.to_csv(os.path.join(DOUBLE_RESULTS_PATH_SHAP, label+"_SHAP_sim.csv"))