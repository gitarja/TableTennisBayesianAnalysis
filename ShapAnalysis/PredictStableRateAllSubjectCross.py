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
            'subsample': [.35, .5, .75, 1.],
            'learning_rate': [0.01, 0.05, 0.1],
            # "scale_pos_weight": [.5, 1.5, 2, 2.5, 4.5],
            # "scale_pos_weight": [.2, .5, 1.5, 2],
            "min_child_weight": [1, 3, 5],
            "scale_pos_weight": [.01, .15, .2, .5],

        }



        # train more for 0 class
        # failure_idx = np.argwhere(y == 0).flatten()
        # success_idx = np.argwhere(y == 1).flatten()
        #
        # resample_success_idx = np.random.choice(range(len(success_idx)), size=int(len(success_idx) * .45),
        #                                         replace=True)
        #
        # X_success = X.iloc[resample_success_idx]
        # y_success = y[resample_success_idx]
        #
        # X_failure = X.iloc[failure_idx]
        # y_failure = y[failure_idx]
        #
        # X = pd.concat([X_success, X_failure])
        # y = np.concatenate([y_success, y_failure])

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

        # train more for 0 class

        weights = [1 if y == 1 else 1 for y in y_train]
        d_train = xgboost.DMatrix(X_train, label=y_train, weight=weights)

        d_val = xgboost.DMatrix(X_val, label=y_val)

        params = {
            "device": "cuda:0",
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "subsample": 1.,
            "max_depth": 10,
            "eval_metric": "aucpr",
            "alpha": .05,
            "scale_pos_weight": .5,
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


def evaluateModel(model, X_test, y_test):
    d_test = xgboost.DMatrix(X_test, label=y_test)

    y_pred = model.predict(d_test)
    # predictions = [1 if value > 0.3 else 0 for value in y_pred]
    predictions = [round(value) for value in y_pred]
    mcc = matthews_corrcoef(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, normalize="true")
    acc = balanced_accuracy_score(y_test, predictions)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)

    precision, recall, thresholds = precision_recall_curve(1 - y_test, 1 - y_pred)

    # Calculate the AUC-PR
    auc_pr = auc(recall, precision)

    f1 = f1_score(y_test, predictions, average='weighted')

    g_mean = geometric_mean_score(y_test, predictions)

    print("AUC", auc_pr)
    print("MCC", mcc)
    print("ACC", acc)
    print(cm)
    print("-------------------------")
    return mcc, cm, acc, auc_pr, f1, g_mean


np.random.seed(1945)  # For Replicability
# control group

inefficient_group, efficient_group = groupLabeling()
all_groups = np.concatenate([inefficient_group, efficient_group])
label = "all_sf"
kf = KFold(n_splits=5, shuffle=True, random_state=1954)
auc_list = []
acc_list = []
mcc_list = []
cm_list = []
shap_values_list = []
X_test_list = []
for i, (train_index, test_index) in enumerate(kf.split(all_groups)):
    train_subjects = all_groups[train_index]
    test_subjects = all_groups[test_index]

    train_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                              exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                              include_subjects=train_subjects)

    train_features = train_reader.getStableUnstableFailureFeatures(group_name="train_subjects", success_failure=True,
                                                                   mod="skill_perception_action")

    test_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                             file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                             exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                             include_subjects=test_subjects)

    test_features = test_reader.getStableUnstableFailureFeatures(group_name="test_subjects",
                                                                 success_failure=True,
                                                                 mod="skill_perception_action")

    X_train = train_features.loc[:, train_features.columns != 'labels']
    X_test = test_features.loc[:, test_features.columns != 'labels']
    y_train = train_features["labels"].values
    y_test = test_features["labels"].values

    # find params
    # all_X = pd.concat([X_train, X_test])
    # all_y = np.concatenate([y_train, y_test])
    # model = trainXGB(X_train, y_train, search_params=True)

    # split data

    # create baseline model and test it
    model = trainXGB(X_train, y_train, search_params=False)

    mcc, cm, acc, auc_score, f1, g_mean = evaluateModel(model, X_test, y_test)

    auc_list.append(auc_score)
    mcc_list.append(mcc)
    acc_list.append(acc)
    cm_list.append(np.expand_dims(cm, 0))

#     all_X = pd.concat([X_train, X_test])
#     X_background = shap.kmeans(all_X, k=15)
#     # model.set_param({"device": "cuda:0"})
#     explainer = CorrExplainer(model.inplace_predict, X_background.data, sampling="gauss+empirical", link=LogitLink())
#     shap_values = explainer.shap_values(X_test)
#
#     # shap.summary_plot(shap_values, X_test, max_display=40)
#     # plt.show()
#
#     shap_values_list.append(shap_values)
#     X_test_list.append(X_test)
#
# all_shap_values = normalizeShap(np.concatenate(shap_values_list))
# all_x_test = pd.concat(X_test_list)
#
# np.save("Results\\"+label+"_shap.npy", all_shap_values)
# all_x_test.to_pickle("Results\\"+label+"_xval.pkl")
# shap.summary_plot(all_shap_values, all_x_test, max_display=40)
# plt.show()

print("%f, %f, %f, %f, %f, %f" % (
    np.average(auc_list), np.std(auc_list), np.average(mcc_list), np.std(mcc_list), np.average(acc_list),
    np.std(acc_list)))

confusion_mat = np.concatenate(cm_list, axis=0)

print(np.average(confusion_mat, axis=0))
print(np.std(confusion_mat, axis=0))

# print(np.average(auc_list))
# print(np.average(mcc_list))
# print(np.average(acc_list))
#
# print(np.std(auc_list))
# print(np.std(mcc_list))
# print(np.std(acc_list))
