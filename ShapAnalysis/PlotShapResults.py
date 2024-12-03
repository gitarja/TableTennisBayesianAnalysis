import os.path

from SHAPPlots import plotSHAP, plotShapSummary, plotShapInteraction, plotShapAbsoulte
import numpy as np
import pandas as pd

label = "all_sf"
shap_results = np.load("Results\\Final2\\Full-model\\" + label + "_shap.npy")
xval_results = pd.read_pickle("Results\\Final2\\Full-model\\" + label + "_xval.pkl")
# yval_results = np.load("Results\\Final\\Action+Perception\\" + label + "_yval.npy")
results_path = os.path.join(
    "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final\\", label)
# show_column = ["hand_movement_sim", "receiver_start_fs", "receiver_im_racket_effect", "receiver_im_racket_dir",
#                "receiver_fixation_racket_latency"]
show_column = ["receiver_p1_al_prec"]
# plotShapInteraction(shap_values=shap_results, x=xval_results, columns=xval_results.columns.values.tolist(),
#                     results_path=results_path, ref="hitter_p1_al_prec", show_column=show_column)
# plotShapAbsoulte(shap_values=shap_results, x=xval_results, y=yval_results, results_path=results_path, columns=xval_results.columns.values.tolist())
# plotShapSummary(shap_values=shap_results, x=xval_results, results_path=results_path)
plotSHAP(shap_values=shap_results, x=xval_results, columns=xval_results.columns.values.tolist(), results_path=results_path, alpha=0.15)
