import os

os.environ['OMP_NUM_THREADS'] = '8'
from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH
from Double.GlobalFeaturesReader import ImpressionFeatures
import xgboost
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_auc_score, auc, balanced_accuracy_score, \
    make_scorer, precision_recall_curve
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RepeatedStratifiedKFold
import pandas as pd

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.linear_model import LogisticRegression

np.random.seed(1945)  # For Replicability




def trainModel(X, y, search_params=False):
    mcc_scorer = make_scorer(matthews_corrcoef)

    # Step 5: Create a scoring dictionary
    scoring = {
        'MCC': mcc_scorer,
        'Balanced_Accuracy': 'balanced_accuracy'
    }

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=1945, stratify=y)

    if search_params:




        def objective(trial):
            params = {

                "class_weight": "balanced", "solver" : "liblinear",
                "tol": trial.suggest_float("tol", 0.01, 1),
                "C": trial.suggest_float("C", 0.5, 10),
                "max_iter" : 1000

            }

            model = LogisticRegression(**params)
            score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        print("Best params:", study.best_params)
        exit()

    else:

        model = LogisticRegression(random_state=0, tol=0.10332172197769018, C=7.423715585041371, max_iter=10000, class_weight="balanced", solver="liblinear").fit(X, y)

    return model


def evaluateModel(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    predictions = [1 if value >= 0.5 else 0 for value in y_pred]


    return predictions, y_pred




lower_group, upper_group = groupLabeling()

# lower group
lower_reader = ImpressionFeatures(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                  include_subjects=lower_group, exclude_failure=False,
                                  exclude_no_pair=True)
# upper group
upper_reader = ImpressionFeatures(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                  include_subjects=upper_group, exclude_failure=False,
                                  exclude_no_pair=True)

label = "all_lower_upper"
mod = "skill_personal_perception_action_impact_ecg_me"

lower_features, skill_lower = lower_reader.getImpressionFeatures(group="lower",
                                                                 mod=mod,
                                                                 return_group_skill=True)

upper_features, skill_upper = upper_reader.getImpressionFeatures(group="upper",
                                                                 mod=mod,
                                                                 return_group_skill=True)

X_lower = lower_features.loc[:, lower_features.columns != 'labels']
y_lower = lower_features["labels"].values

X_upper = upper_features.loc[:, upper_features.columns != 'labels']
y_upper = upper_features["labels"].values

X = pd.concat([X_lower, X_upper])
y = np.concatenate([y_lower, y_upper])

print(X.columns.__len__())
# group_skill = np.concatenate([skill_lower, skill_upper])
# individual_skill = X["subject_skill"].values

# plt.scatter(individual_skill, group_skill)
# plt.show()
print(np.average(y == 1))
print(np.average(y == 0))

# search params
# model = trainModel(X, y, search_params=True)

shap_values_list = []
X_test_list = []
y_test_list = []
pred_bin_list = []
y_pred_list = []
correct_classification_idx = np.zeros((len(y)))
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train = X.iloc[train_index].values
    X_test = X.iloc[test_index].values
    y_train = y[train_index]
    y_test = y[test_index]

    model = trainModel(X_train, y_train, search_params=False)



    # model evaluation
    pred_bin, y_pred = evaluateModel(model, X_test, y_test)

    pred_bin_list.append(pred_bin)
    X_test_list.append(X_test)
    y_test_list.append(y_test)
    y_pred_list.append(y_pred)


# compute metrics
y_test_list = np.concatenate(y_test_list)
pred_bin_list = np.concatenate(pred_bin_list)
y_pred_list = np.concatenate(y_pred_list)
mcc = matthews_corrcoef(y_test_list, pred_bin_list)
cm = confusion_matrix(y_test_list, pred_bin_list, normalize="true")
acc = balanced_accuracy_score(y_test_list, pred_bin_list)
auc_pr = roc_auc_score(y_test_list, y_pred_list)

print("%f, %f, %f" % (acc, mcc, auc_pr))
