import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from SHAPPlots import plotShapComparisonBar


# summarize SHAP and order them

def summarizeSHAP(label="eff_us"):
    shap_results = np.load("Results\\2024-11-15\\" + label + "_norm_shap.npy")
    xval_results = pd.read_pickle("Results\\2024-11-15\\" + label + "_xval.pkl")

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




plotShapComparisonBar(columns, ineff_shap, avg_shap, eff_shap)

