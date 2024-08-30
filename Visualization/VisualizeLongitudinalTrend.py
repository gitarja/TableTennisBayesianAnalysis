from Utils.Conf import DOUBLE_RESULTS_PATH, DOUBLE_FEATURES_FILE_PATH
import pickle
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import numpy as np
from scipy.special import expit
# save the model
n=10
analyzed_features = "stable_rate"
with open(DOUBLE_RESULTS_PATH + "idata_m3_" + analyzed_features + "_" + str(n) + ".pkl", 'rb') as handle:
    idata = pickle.load(handle)


az.plot_ppc(idata, figsize=(20, 7))
plt.show()