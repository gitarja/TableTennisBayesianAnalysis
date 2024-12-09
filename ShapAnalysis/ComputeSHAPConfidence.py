import os

os.environ['OMP_NUM_THREADS'] = '8'
# This must happen before pymc is imported, so you might
# need to restart the kernel for it to take effect.
import pandas as pd

import sys

sys.path.append(os.path.dirname(__file__))
import numpy as np
from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
from Utils.Conf import DOUBLE_SUMMARY_FILE_PATH
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from Utils.Conf import DOUBLE_FEATURES_FILE_PATH
import xgboost
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, \
    make_scorer
from imblearn.metrics import geometric_mean_score
from corr_shap import CorrExplainer
import shap
import matplotlib.pyplot as plt
from Utils.GroupClassification import groupClassifcation
from shap.utils._legacy import LogitLink
import arviz as az
from imblearn.under_sampling import CondensedNearestNeighbour

if __name__ == '__main__':
    cnn = CondensedNearestNeighbour(random_state=42)


    def normalizeShap(arr):
        scaled_arr = arr / np.max(np.abs(arr))
        return scaled_arr


    def trainXGB(X, y):
        mcc_scorer = make_scorer(matthews_corrcoef)

        # Step 5: Create a scoring dictionary
        scoring = {
            'MCC': mcc_scorer,
            'Balanced_Accuracy': 'balanced_accuracy'
        }

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=1945, stratify=y)

        # train more for 0 class
        failure_idx = np.argwhere(y_train == 0).flatten()
        success_idx = np.argwhere(y_test == 1).flatten()

        resample_success_idx = np.random.choice(range(len(success_idx)), size=int(len(success_idx) * .45),
                                                replace=True)

        X_success = X_train.iloc[resample_success_idx]
        y_success = y_train[resample_success_idx]

        X_failure = X_train.iloc[failure_idx]
        y_failure = y_train[failure_idx]

        X_train = pd.concat([X_success, X_failure])
        y_train = np.concatenate([y_success, y_failure])

        weights = [1 if y == 1 else 1 for y in y_train]
        d_train = xgboost.DMatrix(X_train, label=y_train, weight=weights, enable_categorical=True)

        d_val = xgboost.DMatrix(X_val, label=y_val, enable_categorical=True)

        params = {
            "device": "cuda:0",
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "subsample": 1.,
            "max_depth": 10,
            "eval_metric": "aucpr",
            "alpha": .25,
            "scale_pos_weight": .15,
            "min_child_weight": 5,
        }
        model = xgboost.train(
            params,
            d_train,
            5000,
            evals=[(d_val, "val")],
            verbose_eval=False,
            early_stopping_rounds=50,
            num_boost_round=100
        )

        return model



np.random.seed(1945)  # For Replicability
# control group

avg_group, ineff_group, eff_group = groupClassifcation()

all_groups = np.concatenate([avg_group, ineff_group, eff_group])
all_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                        file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                        exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                        include_subjects=all_groups)

all_features = all_reader.getStableUnstableFailureFeatures(group_name="all", success_failure=True, mod="full_mode")

all_X = all_features.loc[:, all_features.columns != 'labels']

label = "all_sf"

X = all_features.loc[:, all_features.columns != 'labels']
y = all_features["labels"].values

print(np.average(y == 1))
print(np.average(y == 0))

# split data

n_booststrap = 3

X_test_list = []
n_column = X.shape[1]
shap_values_norm_list = []
bootstrap_results = np.zeros((n_booststrap, n_column))  # times 3 for KFold
index = 0


shap_values_list = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1945)
for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    model_perm = trainXGB(X_train, y_train)
    X_background = shap.kmeans(all_X, k=15)
    explainer = CorrExplainer(model_perm.inplace_predict, X_background.data, sampling="gauss+empirical",
                                   link=LogitLink())
    for j in range(n_booststrap):
        resample_idx = np.random.choice(range(X_test.shape[0]), size=X_test.shape[0], replace=True)
        X_resample = X_test.iloc[resample_idx]
        shap_values = explainer.shap_values(X_resample)
        shap_values_list.append(shap_values)


# normalize SHAP
all_shap_values = normalizeShap(np.concatenate(shap_values_list))

np.save("Results\\"+label+"_bootstrap_shap.npy", all_shap_values)
