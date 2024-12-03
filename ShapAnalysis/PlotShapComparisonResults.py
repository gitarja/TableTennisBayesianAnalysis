import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from SHAPPlots import plotShapComparison


# summarize SHAP and order them

def summarizeSHAP(label="eff_us"):
    shap_results = np.load("Results\\2024-10-18\\" + label + "_shap.npy")
    xval_results = pd.read_pickle("Results\\2024-10-18\\" + label + "_xval.pkl")

    data_columns = xval_results.columns

    shap_avg_abs = np.average(np.abs(shap_results), axis=0)

    sort_idx = np.argsort(shap_avg_abs)

    sorted_columns = data_columns[sort_idx]
    a = [np.argwhere(sorted_columns == data_columns[i]) for i in range(len(data_columns))]
    return shap_results, data_columns.tolist(), np.concatenate(a).flatten()


labels = ["avg_usuf", "eff_usuf", "ineff_usuf"]
columns_list = []
oder_idx_list = []
label_list = []
avg_shap, columns, avg_order_idx = summarizeSHAP(labels[0])
eff_shap, _, eff_order_idx = summarizeSHAP(labels[1])
ineff_shap, _, ineff_order_idx = summarizeSHAP(labels[2])

# compute
avg_eff_ktest = []
avg_ineff_ktest = []
for i in range(len(columns)):
    a_shap = avg_shap[:, i]
    b_shap = eff_shap[:, i]
    c_shap = ineff_shap[:, i]

    avg_eff_ktest.append(stats.ks_2samp(a_shap, b_shap)[0])
    avg_ineff_ktest.append(stats.ks_2samp(a_shap, c_shap)[0])

df = pd.DataFrame({"F_NAME": columns, "AVG": avg_order_idx,
                   "AVG_EFF": eff_order_idx - avg_order_idx,
                   "AVG_INEFF": ineff_order_idx - avg_order_idx,
                   "EFFICIENT": eff_order_idx, "INEFFICIENT": ineff_order_idx,
                   "AVG_EFF_KTEST": avg_eff_ktest,
                   "AVG_INEFF_KTEST": avg_ineff_ktest,
                   })

# colors

GREEN_SOFT = "#66c2a5"
ORANGE_SOFT = "#fc8d62"
BLUE = "#80b1d3"
ORANGE = "#fb8072"
# set color
df["COLOR_AVG_EFF"] = [BLUE for i in range(len(columns))]
df.loc[df["AVG_EFF"].values < 0, "COLOR_AVG_EFF"] = ORANGE

df["COLOR_AVG_INEFF"] = [BLUE for i in range(len(columns))]
df.loc[df["AVG_INEFF"].values < 0, "COLOR_AVG_INEFF"] = ORANGE

top_10 = df[df["AVG"] > -1]
features_name = top_10["F_NAME"]
shap_order_avg = top_10["AVG"].values
shap_order_eff = top_10["EFFICIENT"].values
shap_order_diff = top_10["AVG_EFF"].values
color = top_10["COLOR_AVG_EFF"]
size = top_10["AVG_EFF_KTEST"]

plotShapComparison(features_name, shap_order_avg,shap_order_eff, shap_order_diff, color, size, label2="EFFICIENT")


shap_order_ineff = top_10["INEFFICIENT"].values
shap_order_diff = top_10["AVG_INEFF"].values
color = top_10["COLOR_AVG_INEFF"]
size = top_10["AVG_INEFF_KTEST"]
plotShapComparison(features_name, shap_order_avg,shap_order_ineff, shap_order_diff, color, size, label2="INEFFICIENT")
