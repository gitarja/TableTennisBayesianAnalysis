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
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, KFold
from Utils.Conf import DOUBLE_FEATURES_FILE_PATH
import xgboost
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, \
    make_scorer, precision_recall_curve
from imblearn.metrics import geometric_mean_score
from corr_shap import CorrExplainer
import shap
import matplotlib.pyplot as plt
from Utils.GroupClassification import groupClassifcation, groupLabeling
from shap.utils._legacy import LogitLink
import arviz as az
from imblearn.under_sampling import CondensedNearestNeighbour

from imblearn.under_sampling import TomekLinks

if __name__ == '__main__':
    cnn = CondensedNearestNeighbour(random_state=42)


    def normalizeShap(arr):
        scaled_arr = arr / np.max(np.abs(arr))
        return scaled_arr


    def trainXGB(X, y, search_params=False):

        mcc_scorer = make_scorer(matthews_corrcoef)

        # Step 5: Create a scoring dictionary
        scoring = {
            'MCC': mcc_scorer,
            'Balanced_Accuracy': 'balanced_accuracy'
        }

        if search_params:

            params = {
                'max_depth': [3, 5, 7, 10],
                'alpha': [0.01, .05, .1, .25, .3, .5],
                'lambda': [0.01, .05, .1, .25, .3, .5],
                'subsample': [.35, .5, .75, 1.],
                'learning_rate': [0.01, 0.05, 0.1],
                # "scale_pos_weight": [.5, 1.5, 2, 2.5, 4.5],
                # "scale_pos_weight": [.2, .5, 1.5, 2],
                "min_child_weight": [1, 3, 5],
                "scale_pos_weight": [.01, .15, .2, .5],

            }

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)

            model = xgboost.XGBClassifier(objective="binary:logistic", eval_metric="aucpr")
            grid_search = GridSearchCV(model, param_grid=params, scoring=scoring, refit="MCC", n_jobs=4,
                                       cv=skf.split(X, y), verbose=3)

            grid_search.fit(X, y)
            print(grid_search.best_score_)
            print(grid_search.best_params_)
            exit()

        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=1945, stratify=y)

            # train more for 0 class
            failure_idx = np.argwhere(y_train == 0).flatten()
            success_idx = np.argwhere(y_train == 1).flatten()

            resample_success_idx = np.random.choice(range(len(success_idx)), size=int(len(success_idx) * .45),
                                                    replace=True)

            X_success = X_train.iloc[resample_success_idx]
            y_success = y_train[resample_success_idx]

            X_failure = X_train.iloc[failure_idx]
            y_failure = y_train[failure_idx]

            X_train = pd.concat([X_success, X_failure])
            y_train = np.concatenate([y_success, y_failure])

            weights = [1 if y == 1 else 1 for y in y_train]
            d_train = xgboost.DMatrix(X_train, label=y_train, weight=weights)

            d_val = xgboost.DMatrix(X_val, label=y_val)

            params =  {
                "device": "cuda:0",
                "learning_rate": 0.05,
                "objective": "binary:logistic",
                "subsample": 1.,
                "max_depth": 10,
                "eval_metric": "aucpr",
                "alpha": .25,
                "scale_pos_weight": .2,
                "min_child_weight": 5,
            }
            model = xgboost.train(
                params,
                d_train,
                1000,
                evals=[(d_val, "val")],
                verbose_eval=False,
                early_stopping_rounds=50,
                num_boost_round=100
            )

        return model



    np.random.seed(1945)  # For Replicability
    # control group

    inefficient_group, efficient_group = groupLabeling()
    all_groups = np.concatenate([inefficient_group, efficient_group])
    label = "all_sf"
    kf = KFold(n_splits=5, shuffle=True, random_state=1954)
    n_booststrap = 3

    X_test_list = []
    n_column = 37
    shap_values_norm_list = []
    bootstrap_results = np.zeros((n_booststrap, n_column))  # times 3 for KFold
    index = 0
    shap_values_list = []
    for i, (train_index, test_index) in enumerate(kf.split(all_groups)):

        train_subjects = all_groups[train_index]
        test_subjects = all_groups[test_index]

        train_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                                file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                                include_subjects=train_subjects)

        train_features = train_reader.getStableUnstableFailureFeatures(group_name="train_subjects", success_failure=True,
                                                                   mod="skill_personal_perception_action_impact")

        test_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                                  include_subjects=test_subjects)

        test_features = test_reader.getStableUnstableFailureFeatures(group_name="test_subjects",
                                                                       success_failure=True,
                                                                       mod="skill_personal_perception_action_impact")

        X_train = train_features.loc[:, train_features.columns != 'labels']
        X_test = test_features.loc[:, test_features.columns != 'labels']
        y_train = train_features["labels"].values
        y_test = test_features["labels"].values

        all_X = pd.concat([X_train, X_test])

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

    np.save("Results\\" + label + "_bootstrap_shap.npy", all_shap_values)


