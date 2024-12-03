import os

os.environ['OMP_NUM_THREADS'] = '8'
# This must happen before pymc is imported, so you might
# need to restart the kernel for it to take effect.
import pandas as pd

import sys

sys.path.append(os.path.dirname(__file__))
import numpy as np

from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
from Utils.Conf import DOUBLE_SUMMARY_FILE_PATH
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, KFold
from Utils.Conf import DOUBLE_FEATURES_FILE_PATH
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, \
    make_scorer, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score

from Utils.GroupClassification import groupClassifcation, groupLabeling

from sklearn.impute import KNNImputer
from imblearn.under_sampling import CondensedNearestNeighbour


def normalizeShap(arr):
    scaled_arr = arr / np.max(np.abs(arr))
    return scaled_arr


def trainModel(X, y, search_params=False):
    mcc_scorer = make_scorer(matthews_corrcoef)

    # Step 5: Create a scoring dictionary
    scoring = {
        'MCC': mcc_scorer,
        'Balanced_Accuracy': 'balanced_accuracy'
    }

    if search_params:

        params = {
            'C': [.3, .5, .7, 1],
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)

        model = LogisticRegression(random_state=0, solver="liblinear", class_weight="balanced")
        grid_search = GridSearchCV(model, param_grid=params, scoring=scoring, refit="MCC", n_jobs=4,
                                   cv=skf.split(X, y), verbose=3)

        grid_search.fit(X, y)
        print(grid_search.best_score_)
        print(grid_search.best_params_)
        exit()

    else:

        # train more for 0 class
        failure_idx = np.argwhere(y == 0).flatten()
        success_idx = np.argwhere(y == 1).flatten()

        resample_success_idx = np.random.choice(range(len(success_idx)), size=int(len(success_idx) * .9),
                                                replace=True)

        X_success = X[resample_success_idx]
        y_success = y[resample_success_idx]

        X_failure = X[failure_idx]
        y_failure = y[failure_idx]

        X = np.concatenate([X_success, X_failure])
        y = np.concatenate([y_success, y_failure])

        model = LogisticRegression(random_state=0, class_weight="balanced", C=1, solver="liblinear").fit(X, y)

    return model


def evaluateModel(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    predictions = [round(value) for value in y_pred]
    mcc = matthews_corrcoef(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, normalize="true")
    acc = balanced_accuracy_score(y_test, predictions)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)

    precision, recall, _ = precision_recall_curve(1 - y_test, 1 - y_pred)

    # Calculate the AUC-PR
    auc_pr = auc(recall, precision)

    f1 = f1_score(y_test, predictions, average='weighted')

    g_mean = geometric_mean_score(y_test, predictions)

    print("AUC", auc_pr)
    print("MCC", mcc)
    print("ACC", acc)
    print("-------------------------")
    return mcc, cm, acc, auc_pr, f1, g_mean


np.random.seed(1945)  # For Replicability
inefficient_group, efficient_group = groupLabeling()
all_groups = np.concatenate([inefficient_group, efficient_group])
label = "all_af"

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
                                                                   mod="full_mode")

    test_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                             file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                             exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                             include_subjects=test_subjects)

    test_features = test_reader.getStableUnstableFailureFeatures(group_name="test_subjects",
                                                                 success_failure=True,
                                                                 mod="full_mode")


    X_train = train_features.loc[:, train_features.columns != 'labels']
    X_test = test_features.loc[:, test_features.columns != 'labels']
    y_train = train_features["labels"].values
    y_test = test_features["labels"].values

    all_X = pd.concat([X_train, X_test])
    imputer = KNNImputer(n_neighbors=5).fit(all_X)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    # split data

    # create baseline model and test it
    model = trainModel(X_train, y_train, search_params=False)

    mcc, cm, acc, auc_score, f1, g_mean = evaluateModel(model, X_test, y_test)

    auc_list.append(auc_score)
    mcc_list.append(mcc)
    acc_list.append(acc)
    cm_list.append(np.expand_dims(cm, 0))


print("%f, %f, %f, %f, %f, %f" % (
    np.average(auc_list), np.std(auc_list), np.average(mcc_list), np.std(mcc_list), np.average(acc_list),
    np.std(acc_list)))

confusion_mat = np.concatenate(cm_list, axis=0)

print(np.average(confusion_mat, axis=0))
print(np.std(confusion_mat, axis=0))
