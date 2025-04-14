import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, features_explanation
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_APM_TTEST
import arviz as az
import pickle
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler



file_name = "idata_"
PREV_ANALYZED_FEATURES = [ "ball_updown", "onset_forward_swing"]
NEXT_ANALYZED_FEATURES = [  "ball_updown", "onset_forward_swing"]


for prev_feature, next_feature in zip(PREV_ANALYZED_FEATURES, NEXT_ANALYZED_FEATURES):
    # load the model
    with open(DOUBLE_RESULTS_PATH_APM_TTEST + file_name + prev_feature + "_prev_" + next_feature + "_next_" + str(
            N_SAMPLES) + ".pkl", 'rb') as handle:

        idata = pickle.load(handle)

    az.plot_posterior(
        idata, var_names=["global_higher_influenceActorPartner_diff", "global_lower_influenceActorPartner_diff"], figsize=(15, 10),   ref_val=0, round_to=2
    )
    plt.show()