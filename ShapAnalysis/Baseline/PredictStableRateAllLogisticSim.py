import os

os.environ['OMP_NUM_THREADS'] = '8'
# This must happen before pymc is imported, so you might
# need to restart the kernel for it to take effect.
import pandas as pd

import sys

sys.path.append(os.path.dirname(__file__))
import numpy as np

from Double.GlobalFeaturesReader import ImpressionFeatures
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, KFold
from Utils.Conf import DOUBLE_FEATURES_FILE_PATH
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, \
    make_scorer, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score

from Utils.GroupClassification import groupClassifcation, groupLabeling


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

        model = LogisticRegression(random_state=0, class_weight="balanced", C=1, solver="liblinear").fit(X, y)

    return model


def evaluateModel(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    predictions = [round(value) for value in y_pred]
    mcc = matthews_corrcoef(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, normalize="true")
    acc = balanced_accuracy_score(y_test, predictions)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    predictions_bin = np.array(predictions == y_test)

    # Calculate the AUC-PR
    auc_pr = auc(recall, precision)

    f1 = f1_score(y_test, predictions, average='weighted')

    g_mean = geometric_mean_score(y_test, predictions)

    print("ACC", acc)
    print("MCC", mcc)
    print("GMEAN", g_mean)
    print(cm)
    print("-------------------------")
    return mcc, cm, acc, auc_pr, f1, g_mean, predictions_bin


np.random.seed(1945)  # For Replicability
lower_group, upper_group = groupLabeling()

# lower group
lower_reader = ImpressionFeatures(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                  include_subjects=lower_group, exclude_failure=False,
                                  exclude_no_pair=False)
# upper group
upper_reader = ImpressionFeatures(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                  include_subjects=upper_group, exclude_failure=False,
                                  exclude_no_pair=False)

label = "all"
lower_features, skill_lower = lower_reader.getImpressionFeatures(group="lower",
                                                                 mod="skill_personal_perception_action_impact",
                                                                 return_group_skill=True)

upper_features, skill_upper = upper_reader.getImpressionFeatures(group="upper",
                                                                 mod="skill_personal_perception_action_impact",
                                                                 return_group_skill=True)

X_lower = lower_features.loc[:, lower_features.columns != 'labels']
y_lower = lower_features["labels"].values

X_upper = upper_features.loc[:, upper_features.columns != 'labels']
y_upper = upper_features["labels"].values

X = pd.concat([X_lower, X_upper])
y = np.concatenate([y_lower, y_upper])
group_skill = np.concatenate([skill_lower, skill_upper])
individual_skill = X["subject_skill"].values

# plt.scatter(individual_skill, group_skill)
# plt.show()
print(np.average(y == 1))
print(np.average(y == 0))

# search params
# model = trainModel(X, y, search_params=True)

auc_list = []
acc_list = []
mcc_list = []
cm_list = []
shap_values_list = []
X_test_list = []
y_test_list = []
correct_classification_idx = np.zeros((len(y)))
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)
for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    model = trainModel(X_train, y_train, search_params=False)

    mcc, cm, acc, auc_score, f1, g_mean, pred_bin = evaluateModel(model, X_test, y_test)

    correct_classification_idx[test_index[pred_bin == True]] = 1

    auc_list.append(auc_score)
    mcc_list.append(mcc)
    acc_list.append(acc)
    cm_list.append(np.expand_dims(cm, 0))



print(np.average(cm_list, axis=0))
print(np.std(cm_list, axis=0))
print("%f, %f, %f, %f, %f, %f" % (
    np.average(auc_list), np.std(auc_list), np.average(mcc_list), np.std(mcc_list), np.average(acc_list),
    np.std(acc_list)))
