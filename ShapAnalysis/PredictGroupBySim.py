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
from corr_shap import CorrExplainer
import matplotlib.pyplot as plt
from shap.utils._legacy import LogitLink
import shap
from scipy import stats

np.random.seed(1945)  # For Replicability


# def normalizeShap(arr):
#     scaled_arr = arr / np.max(np.abs(arr))
#     return scaled_arr


def trainXGB(X, y, x_test=None, y_test=None, search_params=False):
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

    # d_train = xgboost.DMatrix(X, label=y, enable_categorical=True)
    # d_val = xgboost.DMatrix(x_test, label=y_test, enable_categorical=True)

    if search_params:

        # params = {
        #     'max_depth': [3, 5, 7, 10],
        #     'alpha': [0.05, .1, .25, .3, .5],
        #     'subsample': [.25, .35, .5, .75, 1.],
        #     'learning_rate': [0.01, 0.03, 0.05],
        #     "min_child_weight": [1, 3, 4, 5],
        #     "max_delta_step": [1, 3],
        #
        #     # "gamma": [0, 0.2, 0.5, 0.75, 1.],
        #     # "max_leaves": [3, 7, 5, 10, 15],
        #     # "scale_pos_weight": [.5,1 ],
        #
        # }
        #
        # skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1945)
        #
        # model = xgboost.XGBClassifier(objective="binary:logistic", eval_metric="aucpr")
        # grid_search = GridSearchCV(model, param_grid=params, scoring=scoring, refit="MCC", n_jobs=4,
        #                            cv=skf.split(X, y), verbose=3)
        #
        # grid_search.fit(X, y)
        # print(grid_search.best_score_)
        # print(grid_search.best_params_)
        # exit()

        def objective(trial):
            params = {
                "device": "cuda:0",

                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 5),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
                "alpha": trial.suggest_float("alpha", 0, .3),
                "max_delta_step": trial.suggest_int("max_delta_step", 1, 3)
            }

            model = xgboost.XGBClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        print("Best params:", study.best_params)

        exit()

    else:

        params = {
            "device": "cuda:0",
            "objective": "binary:logistic",
            "eval_metric": "logloss",

            "learning_rate": 0.02,
            "subsample": 1.,
            "max_depth": 3,
            "alpha": .05,
            "min_child_weight": 3,
            "max_delta_step": 3,

        }

        model = xgboost.train(
            params,
            d_train,
            100,
            evals=[(d_val, "val")],
            verbose_eval=False,
            early_stopping_rounds=50,

        )

    return model


def evaluateModel(model, X_test, y_test):
    d_test = xgboost.DMatrix(X_test, label=y_test, enable_categorical=True)

    y_pred = model.predict(d_test)
    predictions = [1 if value >= 0.5 else 0 for value in y_pred]
    # predictions = [round(value) for value in y_pred]

    return predictions, y_pred


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
    # mod = "top-2"
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



    # plt.scatter(individual_skill, group_skill)
    # plt.show()
    print(np.average(y == 1))
    print(np.average(y == 0))

    # search params
    # model = trainXGB(X, y, search_params=True)

    shap_values_list = []
    X_test_list = []
    y_test_list = []
    y_test_idx_list = []
    pred_bin_list = []
    y_pred_list = []
    correct_classification_idx = np.zeros((len(y)))
    kf = StratifiedKFold(n_splits=3, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        model = trainXGB(X_train, y_train, X_test, y_test, search_params=False)

        # # compute SHAP

        model.set_param({"device": "cuda:0"})
        explainer = CorrExplainer(model.inplace_predict, X_train, sampling="gauss+empirical",
                                  link=LogitLink())
        shap_values = explainer.shap_values(X_test)

        shap_values_list.append(shap_values)

        # model evaluation
        pred_bin, y_pred = evaluateModel(model, X_test, y_test)

        pred_bin_list.append(pred_bin)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        y_pred_list.append(y_pred)
        y_test_idx_list.append(test_index)

    all_y_pred = np.concatenate(y_pred_list)
    np.save("Results\\" + mod + "_pred.npy", all_y_pred)
    all_shap_values = np.concatenate(shap_values_list)
    all_x_test = pd.concat(X_test_list)
    all_y_tes = np.concatenate(y_test_list)
    np.save("Results\\" + label + "_shap.npy", all_shap_values)
    np.save("Results\\" + label + "_yval.npy", all_y_tes)
    all_x_test.to_pickle("Results\\" + label + "_xval.pkl")

    shap.summary_plot(all_shap_values, all_x_test, max_display=50)
    plt.show()

    # compute metrics
    y_test_list = np.concatenate(y_test_list)
    pred_bin_list = np.concatenate(pred_bin_list)
    y_pred_list = np.concatenate(y_pred_list)
    y_test_idx_list = np.concatenate(y_test_idx_list)
    mcc = matthews_corrcoef(y_test_list, pred_bin_list)
    cm = confusion_matrix(y_test_list, pred_bin_list)
    acc = balanced_accuracy_score(y_test_list, pred_bin_list)
    auc_pr = roc_auc_score(y_test_list, y_pred_list)

    print(cm)
    print("%f, %f, %f" % (acc, mcc, auc_pr))

    # plot prediction
    # individual_skill = X["subject_skill"].values
    # group_skill = np.concatenate([skill_lower, skill_upper])
    #
    # correct_idx = y_test_list == pred_bin_list
    # incorrect_idx = y_test_list != pred_bin_list
    # # show the linear reg
    # plt.rcParams["text.usetex"] = True
    # plt.rcParams["font.family"] = "Arial"
    # plt.rcParams['font.size'] = 20
    # plt.scatter(individual_skill[y == 0], group_skill[y == 0], label="overestimate", color="#B5152C", s=50, edgecolors="#636363", linewidths=0.1)
    # plt.scatter(individual_skill[y == 1], group_skill[y == 1], label="underestimate", color="#68a880", s=50, edgecolors="#636363", linewidths=0.1)
    # plt.scatter(individual_skill[y_test_idx_list[incorrect_idx]], group_skill[y_test_idx_list[incorrect_idx]], label="underestimate", color="#000000", s=50,
    #             edgecolors="#000000", linewidths=0.5)
    # plt.show()
