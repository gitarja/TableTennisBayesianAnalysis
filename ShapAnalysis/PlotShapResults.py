import os.path

from SHAPPlots import plotSHAP, plotShapSummary, plotShapInteraction
import numpy as np
import pandas as pd

label = "similarity_lower_upper"
shap_results = np.load("Results\\" + label + "_shap.npy")
xval_results = pd.read_pickle("Results\\" + label + "_xval.pkl")
results_path = os.path.join(
    "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Summary-results-V1-2024-10-16\\", label)
# show_column = ["hand_movement_sim", "receiver_start_fs", "receiver_im_racket_effect", "receiver_im_racket_dir",
#                "receiver_fixation_racket_latency"]
# show_column = ["hitter_p1_al_prec"]
# plotShapInteraction(shap_values=shap_results, x=xval_results, columns=xval_results.columns.values.tolist(),
#                     results_path=results_path, ref="receiver_p1_al_prec", show_column=show_column)
plotShapSummary(shap_values=shap_results, x=xval_results, results_path=results_path)
plotSHAP(shap_values=shap_results, x=xval_results, columns=xval_results.columns.values.tolist(), results_path=results_path)
