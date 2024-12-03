import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from SHAPPlots import plotShapComparisonBar

results_path = "Results\\2024-11-15\\"

# summarize SHAP and order them
def normalizeShap(arr, concatenate_arr):
    scaled_arr = arr / np.max(np.abs(concatenate_arr))
    return scaled_arr

def summarizeSHAP(label="eff_us"):
    shap_results = np.load( results_path+ label + "_shap.npy")
    xval_results = pd.read_pickle(results_path + label + "_xval.pkl")

    data_columns = xval_results.columns

    shap_avg_abs = np.average(np.abs(shap_results), axis=0)

    sort_idx = np.argsort(shap_avg_abs)

    sorted_columns = data_columns[sort_idx]
    a = [np.argwhere(sorted_columns == data_columns[i]) for i in range(len(data_columns))]
    return shap_results, data_columns.tolist(), np.concatenate(a).flatten()


labels = ["avg_sf", "eff_sf", "ineff_sf"]
columns_list = []
oder_idx_list = []
label_list = []
avg_shap, columns, avg_order_idx = summarizeSHAP(labels[0])
eff_shap, _, eff_order_idx = summarizeSHAP(labels[1])
ineff_shap, _, ineff_order_idx = summarizeSHAP(labels[2])


concatenate_shap = np.concatenate([avg_shap, eff_shap, ineff_shap])

avg_normalize_shape = normalizeShap(avg_shap, concatenate_shap)
eff_normalize_shape = normalizeShap(eff_shap, concatenate_shap)
ineff_normalize_shape = normalizeShap(ineff_shap, concatenate_shap)

np.save(results_path + labels[0] + "_norm_shap.npy", avg_normalize_shape)
np.save(results_path + labels[1] + "_norm_shap.npy", eff_normalize_shape)
np.save(results_path + labels[2] + "_norm_shap.npy", ineff_normalize_shape)