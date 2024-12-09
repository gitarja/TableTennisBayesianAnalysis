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
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, make_scorer, precision_recall_curve
from imblearn.metrics import geometric_mean_score
from corr_shap import CorrExplainer
import shap
import matplotlib.pyplot as plt
from Utils.GroupClassification import groupClassifcation
from shap.utils._legacy import LogitLink
import arviz as az
from imblearn.under_sampling import CondensedNearestNeighbour

from imblearn.under_sampling import TomekLinks
if __name__ == '__main__':
    cnn = CondensedNearestNeighbour(random_state=42)
    def normalizeShap(arr):
        scaled_arr = arr / np.max(np.abs(arr))
        return scaled_arr
    def trainXGB(X, y,  search_params=False):

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
                "min_child_weight": [1,  3, 5],
                "scale_pos_weight": [.01, .15, .2, .5],

            }

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)

            model = xgboost.XGBClassifier(objective= "binary:logistic", eval_metric="aucpr", enable_categorical=True)
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
            success_idx = np.argwhere(y_test == 1).flatten()

            resample_success_idx = np.random.choice(range(len(success_idx)), size=int(len(success_idx) * .85),
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


    def evaluateModel(model, X_test, y_test):
        d_test = xgboost.DMatrix(X_test, label=y_test)

        y_pred = model.predict(d_test)
        # predictions = [1 if value > 0.4 else 0 for value in y_pred]
        predictions = [round(value) for value in y_pred]
        mcc = matthews_corrcoef(y_test, predictions)
        cm = confusion_matrix(y_test, predictions, normalize="true")
        acc = balanced_accuracy_score(y_test, predictions)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)

        precision, recall, thresholds = precision_recall_curve( 1-y_test,  1-y_pred)

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

    avg_group, ineff_group, eff_group = groupClassifcation()

    all_groups = np.concatenate([avg_group, ineff_group, eff_group])
    all_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                        file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                        exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                        include_subjects=all_groups)

    all_features = all_reader.getStableUnstableFailureFeatures(group_name="all", success_failure=True, mod="skill_personal_perception_action_impact")

    all_X = all_features.loc[:, all_features.columns != 'labels']
    all_y = all_features["labels"].values

    all_X = all_X.apply(pd.to_numeric, errors='coerce')
    label = "all_sf"


    print(np.sum(all_y == 1))
    print(np.sum(all_y == 0))

    # find params
    # model = trainXGB(all_X, all_y, search_params=True)

    # split data


    auc_list = []
    acc_list = []
    mcc_list = []
    cm_list = []
    shap_values_list = []
    X_test_list = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1945)
    for i, (train_index, test_index) in enumerate(kf.split(all_X, all_y)):
        X_train = all_X.iloc[train_index]
        X_test = all_X.iloc[test_index]
        y_train = all_y[train_index]
        y_test = all_y[test_index]


        # create baseline model and test it
        model = trainXGB(X_train, y_train,  search_params=False)

        mcc, cm, acc, auc_score, f1, g_mean = evaluateModel(model, X_test, y_test)

        auc_list.append(auc_score)
        mcc_list.append(mcc)
        acc_list.append(acc)
        cm_list.append(np.expand_dims(cm, 0))

        X_background = shap.kmeans(all_X, k=15)
        # model.set_param({"device": "cuda:0"})
        explainer = CorrExplainer(model.inplace_predict, X_background.data, sampling="gauss+empirical", link=LogitLink())
        shap_values = explainer.shap_values(X_test)

        #shap.summary_plot(shap_values, X_test, max_display=40)
        #plt.show()

        shap_values_list.append(shap_values)
        X_test_list.append(X_test)

    all_shap_values = normalizeShap(np.concatenate(shap_values_list))
    all_x_test = pd.concat(X_test_list)

    np.save("Results\\"+label+"_shap.npy", all_shap_values)
    all_x_test.to_pickle("Results\\"+label+"_xval.pkl")
    shap.summary_plot(all_shap_values, all_x_test, max_display=40)
    plt.show()

    print("%f, %f, %f, %f, %f, %f" % (np.average(auc_list), np.std(auc_list), np.average(mcc_list), np.std(mcc_list), np.average(acc_list), np.std(acc_list)))

    print(np.average(cm_list, axis=0))
    print(np.std(cm_list, axis=0))










