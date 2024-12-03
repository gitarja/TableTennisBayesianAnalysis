import os.path

from SHAPPlots import plotSHAP, plotShapSummary, plotShapInteraction, plotShapAbsoulte
import numpy as np
import pandas as pd
import os
os.environ['OMP_NUM_THREADS'] = '8'
# This must happen before pymc is imported, so you might
# need to restart the kernel for it to take effect.
import sys

sys.path.append(os.path.dirname(__file__))
from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
from Utils.Conf import DOUBLE_SUMMARY_FILE_PATH
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from Utils.Conf import DOUBLE_FEATURES_FILE_PATH
import xgboost
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, make_scorer
from imblearn.metrics import geometric_mean_score
from corr_shap import CorrExplainer
import shap
import matplotlib.pyplot as plt
from Utils.GroupClassification import groupClassifcation
from shap.utils._legacy import LogitLink
import arviz as az
from imblearn.under_sampling import CondensedNearestNeighbour

label = "all_sf"
shap_results = np.load("Results\\" + label + "_shap.npy")
xval_results = pd.read_pickle("Results\\" + label + "_xval.pkl")
x_columns = xval_results.columns

shap_abs_results = np.average(np.abs(shap_results), axis=0)


n = 37

top_n = np.argsort(-shap_abs_results)[:n]

top_n_columns = x_columns[top_n].to_list()


def trainXGB(X, y, search_params=False):
    mcc_scorer = make_scorer(matthews_corrcoef)

    # Step 5: Create a scoring dictionary
    scoring = {
        'MCC': mcc_scorer,
        'Balanced_Accuracy': 'balanced_accuracy'
    }

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=1945, stratify=y)
    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_val = xgboost.DMatrix(X_val, label=y_val)

    if search_params:

        params = {
            'max_depth': [3, 5, 7, 10],
            'alpha': [.05, .1, .25, .3, .5],
            'subsample': [.35, .5, .75, 1.],
            'learning_rate': [0.01, 0.05, 0.1],
            # "scale_pos_weight": [.5, 1.5, 2, 2.5, 4.5],
            # "scale_pos_weight": [.2, .5, 1.5, 2],
            "min_child_weight": [1, 3, 5],
            "scale_pos_weight": [.01, .15, .2, .5]
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

        params = {
            "device": "cuda:0",
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "subsample": 1.,
            "max_depth": 5,
            "eval_metric": "aucpr",
            "alpha": .3,
            "scale_pos_weight": 0.2,
            "min_child_weight": 1,
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


def evaluateModel(model, X_test, y_test):
    d_test = xgboost.DMatrix(X_test, label=y_test)

    y_pred = model.predict(d_test)
    predictions = [round(value) for value in y_pred]
    mcc = matthews_corrcoef(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, normalize="true")
    acc = balanced_accuracy_score(y_test, predictions)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)

    f1 = f1_score(y_test, predictions, average='weighted')

    g_mean = geometric_mean_score(y_test, predictions)

    print("AUC", auc_score)
    print("MCC", mcc)
    print("ACC", acc)
    print("-------------------------")
    return mcc, cm, acc, auc_score, f1, g_mean


np.random.seed(1945)  # For Replicability
# control group

avg_group, ineff_group, eff_group = groupClassifcation()

all_groups = np.concatenate([avg_group, ineff_group, eff_group])
all_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                        file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                        exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                        include_subjects=all_groups)

all_features = all_reader.getStableUnstableFailureFeatures(group_name="all", success_failure=True, exclude_action=False)

# remove contact features
# top_n_columns.remove("receiver_im_racket_ball_angle")
# top_n_columns.remove("receiver_im_ball_wrist")
# top_n_columns.remove("receiver_im_racket_ball_wrist")

top_n_columns = ["receiver_im_racket_ball_angle", "receiver_im_ball_wrist", "receiver_im_racket_ball_wrist", "receiver_skill"]
all_X = all_features.loc[:, top_n_columns]
all_y = all_features["labels"].values

label = "all_sf"

print(np.average(all_y == 1))
print(np.average(all_y == 0))

# find params
model = trainXGB(all_X, all_y, search_params=False)

# split data


auc_list = []
acc_list = []
mcc_list = []
shap_values_list = []
X_test_list = []
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)
for i, (train_index, test_index) in enumerate(kf.split(all_X, all_y)):
    X_train = all_X.iloc[train_index]
    X_test = all_X.iloc[test_index]
    y_train = all_y[train_index]
    y_test = all_y[test_index]


    model = trainXGB(X_train, y_train, search_params=False)

    mcc, cm, acc, auc_score, f1, g_mean = evaluateModel(model, X_test, y_test)

    auc_list.append(auc_score)
    mcc_list.append(mcc)
    acc_list.append(acc)



print("%f, %f, %f, %f, %f, %f" % (
np.average(auc_list), np.std(auc_list), np.average(mcc_list), np.std(mcc_list), np.average(acc_list), np.std(acc_list)))









