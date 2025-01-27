from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH
from Double.GlobalFeaturesReader import ImpressionFeatures
import xgboost
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, \
    make_scorer, precision_recall_curve
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
import pandas as pd
import shap
from corr_shap import CorrExplainer
import matplotlib.pyplot as plt
from shap.utils._legacy import LogitLink

np.random.seed(1945)  # For Replicability


def normalizeShap(arr):
    scaled_arr = arr / np.max(np.abs(arr))
    return scaled_arr


def trainXGB(X, y, search_params=False):
    mcc_scorer = make_scorer(matthews_corrcoef)

    # Step 5: Create a scoring dictionary
    scoring = {
        'MCC': mcc_scorer,
        'Accuracy': 'accuracy'
    }
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.05, random_state=1945, stratify=y)
    d_train = xgboost.DMatrix(X_train, label=y_train, enable_categorical=True)
    d_val = xgboost.DMatrix(X_val, label=y_val, enable_categorical=True)

    if search_params:

        params = {
            'max_depth': [3, 5, 7, 10],
            'alpha': [0.05, .1, .25, .3, .5],
            'subsample': [.25, .35, .5, .75, 1.],
            'learning_rate': [0.01, 0.05, 0.1],
            "min_child_weight": [1, 3, 4, 5],
            #"max_leaves": [3, 7, 5, 10, 15],
            # "scale_pos_weight": [.5,1 ],

        }

        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1945)

        model = xgboost.XGBClassifier(objective="binary:logistic", eval_metric="aucpr")
        grid_search = GridSearchCV(model, param_grid=params, scoring=scoring, refit="MCC", n_jobs=2,
                                   cv=skf.split(X, y), verbose=3)

        grid_search.fit(X, y)
        print(grid_search.best_score_)
        print(grid_search.best_params_)
        exit()

    else:

        params ={
            "device": "cuda:0",
            "learning_rate": 0.01,
            "objective": "binary:logistic",
            "subsample": .75,
            "max_depth": 3,
            "eval_metric": "aucpr",
            "alpha": .25,
            "min_child_weight": 3,


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
    d_test = xgboost.DMatrix(X_test, label=y_test, enable_categorical=True)

    y_pred = model.predict(d_test)
    #predictions = [1 if value >= 0.45 else 0 for value in y_pred]
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
    lower_features, skill_lower = lower_reader.getImpressionFeatures(group="lower", mod="skill_personal",
                                                                     return_group_skill=True)

    upper_features, skill_upper = upper_reader.getImpressionFeatures(group="upper", mod="skill_personal",
                                                                     return_group_skill=True)


    X_lower = lower_features.loc[:, lower_features.columns != 'labels']
    y_lower = lower_features["labels"].values

    X_upper = upper_features.loc[:, upper_features.columns != 'labels']
    y_upper = upper_features["labels"].values

    X = pd.concat([X_lower, X_upper])
    y = np.concatenate([y_lower, y_upper])

    print(X.columns.__len__())
    #group_skill = np.concatenate([skill_lower, skill_upper])
    # individual_skill = X["subject_skill"].values

    # plt.scatter(individual_skill, group_skill)
    # plt.show()
    print(np.average(y == 1))
    print(np.average(y == 0))

    # search params
    #model = trainXGB(X, y, search_params=True)

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

        model = trainXGB(X_train, y_train, search_params=False)

        mcc, cm, acc, auc_score, f1, g_mean, pred_bin = evaluateModel(model, X_test, y_test)

        # correct_classification_idx[test_index[pred_bin == True]] = 1

        auc_list.append(auc_score)
        mcc_list.append(mcc)
        acc_list.append(acc)
        cm_list.append(np.expand_dims(cm, 0))

        # compute SHAP

        #model.set_param({"device": "cuda:0"})
    #     explainer = CorrExplainer(model.inplace_predict, X, sampling="gauss+empirical",
    #                               link=LogitLink())
    #     shap_values = explainer.shap_values(X_test)
    #
    #
    #     # shap.summary_plot(normalizeShap(shap_values), X_test, max_display=40)
    #     # plt.show()
    #     shap_values_list.append(shap_values)
    #     X_test_list.append(X_test)
    #     y_test_list.append(y_test)
    #
    # all_shap_values = normalizeShap(np.concatenate(shap_values_list))
    # all_x_test = pd.concat(X_test_list)
    # all_y_tes = np.concatenate(y_test_list)
    # np.save("Results\\" + label + "_lower_upper_shap.npy", all_shap_values)
    # np.save("Results\\" + label + "_lower_upper_yval.npy", all_y_tes)
    # all_x_test.to_pickle("Results\\" + label + "_lower_upper_xval.pkl")
    #
    # shap.summary_plot(all_shap_values, all_x_test, max_display=50)
    # plt.show()

    # print(np.average(cm_list, axis=0))
    # print(np.std(cm_list, axis=0))
    print("%f, %f, %f, %f, %f, %f" % (
    np.average(auc_list), np.std(auc_list), np.average(mcc_list), np.std(mcc_list), np.average(acc_list),
    np.std(acc_list)))
