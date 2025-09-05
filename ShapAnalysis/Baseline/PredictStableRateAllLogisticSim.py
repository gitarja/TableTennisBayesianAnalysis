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
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

np.random.seed(1945)  # For Replicability


def normalizeShap(arr):
    scaled_arr = arr / np.max(np.abs(arr))
    return scaled_arr


def trainXGB(X, y, search_params=False):
    model = LogisticRegression(random_state=0, tol=0.10332172197769018, C=7.423715585041371, max_iter=10000,
                               class_weight="balanced", solver="liblinear").fit(X, y)


    return model


def evaluateModel(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    predictions = [1 if value >= 0.5 else 0 for value in y_pred]

    return predictions, y_pred

def computeMetrices(y_test, y_pred_bin, y_pred):
    mcc = matthews_corrcoef(y_test, y_pred_bin)
    acc = balanced_accuracy_score(y_test, y_pred_bin)
    auc_pr = roc_auc_score(y_test, y_pred)


    return acc, mcc, auc_pr

def hanleyMcneilSE(auc_value, y_true):
    n1 = np.sum(y_true == 1)  # Number of positive cases
    n2 = np.sum(y_true == 0)  # Number of negative cases

    # Hanley-McNeil formula for standard error
    Q1 = auc_value / (2 - auc_value)
    Q2 = 2 * auc_value**2 / (1 + auc_value)
    se = np.sqrt((auc_value * (1 - auc_value) + (n1 - 1) * (Q1 - auc_value**2) + (n2 - 1) * (Q2 - auc_value**2)) / (n1 * n2))

    return se

if __name__ == '__main__':

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
    # model = trainXGB(X, y, search_params=True)



    shap_values_list = []
    X_test_list = []
    y_test_list_all = []

    auc_list = []
    mcc_list = []
    acc_list = []

    correct_classification_idx = np.zeros((len(y)))
    N_BOOST = 50
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)

    for k in tqdm(range(N_BOOST)):
        y_test_list = []
        pred_bin_list = []
        y_pred_list = []

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            model = trainXGB(X_train, y_train, search_params=False)
            resample_idx = np.random.choice(range(X_test.shape[0]), size=X_test.shape[0], replace=True)
            X_test_resample = X_test.iloc[resample_idx]
            y_test_resample = y_test[resample_idx]

            pred_bin, y_pred = evaluateModel(model, X_test_resample, y_test_resample)

            pred_bin_list.append(pred_bin)
            X_test_list.append(X_test)
            y_test_list.append(y_test_resample)
            y_test_list_all.append(y_test_resample)
            y_pred_list.append(y_pred)



        acc, mcc, auc_pr = computeMetrices(np.concatenate(y_test_list), np.concatenate(pred_bin_list), np.concatenate(y_pred_list))

        acc_list.append(acc)
        mcc_list.append(mcc)
        auc_list.append(auc_pr)



    #compute CI

    n = len(acc_list)
    # Standard error of ACC
    standard_error_acc = np.std(acc_list) / np.sqrt(n)
    standard_error_auc = hanleyMcneilSE(np.average(auc_list), np.concatenate(y_test_list_all))
    standard_error_mcc = np.std(mcc_list) / np.sqrt(n)
    #
    t_critical = 1.96
    # # Confidence interval
    margin_of_error_acc = t_critical * standard_error_acc
    margin_of_error_mcc = t_critical * standard_error_mcc
    margin_of_error_auc = t_critical * standard_error_auc
    #
    print("%f, %f, %f" % (np.average(acc_list), np.average(mcc_list), np.average(auc_list)))
    print("%f, %f, %f" % (margin_of_error_acc, margin_of_error_mcc, margin_of_error_auc))

