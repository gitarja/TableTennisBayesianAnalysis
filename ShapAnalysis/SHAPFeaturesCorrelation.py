import os.path
from matplotlib.lines import Line2D
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams.update({'xtick.labelsize': 25, 'ytick.labelsize': 25})
label = "all_sf"
shap_results = np.load("Results\\Final2\\Full-model\\" + label + "_shap.npy")
xval_results = pd.read_pickle("Results\\Final2\\Full-model\\" + label + "_xval.pkl")
# yval_results = np.load("Results\\Final\\Action+Perception\\" + label + "_yval.npy")
results_path = os.path.join(
    "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final\\", label)



explanation = shap.Explanation(values=shap_results, data=xval_results.values, feature_names=xval_results.columns.values.tolist())

fig, ax = plt.subplots(figsize=(12, 8))
y_abs_max = np.max(np.abs(explanation[:, "hitter_p1_al_prec"].values))

shap.plots.scatter(explanation[:, "hitter_p1_al_prec"], color=explanation[:, "receiver_p1_al_prec"],  dot_size=12, ax=ax, show=False)
ax.set_ylim(-1 * y_abs_max, y_abs_max)
ax.set_frame_on(False)
xmin, xmax = ax.get_xaxis().get_view_interval()
ymin, ymax = ax.get_yaxis().get_view_interval()
ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
ax.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.savefig(results_path + "\\hitter_receiver_p1_al_prec.pdf", format='pdf', transparent=True)
plt.close()
# shap.plots.scatter(explanation[:, "hand_movement_sim"], alpha=0.2)

#
# plt.show()