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
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, make_scorer
from imblearn.metrics import geometric_mean_score
from corr_shap import CorrExplainer
import shap
import matplotlib.pyplot as plt
from Utils.GroupClassification import groupClassifcation
from shap.utils._legacy import LogitLink
import arviz as az
from imblearn.under_sampling import CondensedNearestNeighbour

n_permutation = 100
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

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=1945, stratify=y)
        d_train = xgboost.DMatrix(X_train, label=y_train)
        d_val = xgboost.DMatrix(X_val, label=y_val)

        if search_params:

            params = {
                'max_depth': [3, 5, 7, 10, 15],
                'alpha': [.05, .1, .25, .3, .5],
                'subsample': [.35, .5, .75, 1.],
                'learning_rate': [0.01, 0.05, 0.1],
                # "scale_pos_weight": [.5, 1.5, 2, 2.5, 4.5],
                "scale_pos_weight": [.2, .5, 1.5, 2],
                "min_child_weight": [1,  3, 5],
                # "scale_pos_weight": [.01, .15, .2, .5]
            }

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)

            model = xgboost.XGBClassifier(objective= "binary:logistic", eval_metric="aucpr")
            grid_search = GridSearchCV(model, param_grid=params, scoring=scoring, refit="MCC", n_jobs=4,
                                               cv=skf.split(X, y), verbose=3)

            grid_search.fit(X, y)
            print(grid_search.best_score_)
            print(grid_search.best_params_)
            exit()

        else:

            params = {
                "device": "cuda:0",
                "learning_rate": 0.05,
                "objective": "binary:logistic",
                "subsample": .75,
                "max_depth": 3,
                "eval_metric": "aucpr",
                "alpha": .25,
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

    all_features = all_reader.getStableUnstableFailureFeatures(group_name="all", success_failure=True)

    all_X = all_features.loc[:, all_features.columns != 'labels']

    reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                            file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                            exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                            include_subjects=ineff_group)

    label = "ineff_sf"

    features = reader.getStableUnstableFailureFeatures(group_name=label, success_failure=True)

    X = features.loc[:, features.columns != 'labels']
    y = features["labels"].values

    print(np.average(y == 1))
    print(np.average(y == 0))

    # find params
    # model = trainXGB(X, y, search_params=True)

    # split data


    auc_list = []
    acc_list = []
    mcc_list = []
    shap_values_list = []
    X_test_list = []
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # print(np.mean(y_train==0))
        # print(np.mean(y_train == 1))
        # print("*******************************")
        #create baseline model and test it
        model = trainXGB(X_train, y_train, search_params=False)

        mcc, cm, acc, auc_score, f1, g_mean = evaluateModel(model, X_test, y_test)

        auc_list.append(auc_score)
        mcc_list.append(mcc)
        acc_list.append(acc)

        # trained model with subset of training data
        # print("Start permutation")
        # for i_perm in range(n_permutation):
        #     bootstrap_idx = np.random.choice(X_train.shape[0], size=int(X_train.shape[0] * 0.75), replace=True)
        #     X_perm = X_train.iloc[bootstrap_idx]
        #     y_perm = y_train[bootstrap_idx]
        #     model_perm = trainXGB(X_perm, y_perm)
        #     evaluateModel(model_perm, X_test, y_test)
        #
        # print("End permutation")
    #
    #     X_background = shap.kmeans(all_X, k=10)
    #     # model.set_param({"device": "cuda:0"})
    #     explainer = CorrExplainer(model.inplace_predict, X_background.data, sampling="gauss+empirical", link=LogitLink())
    #     shap_values = explainer.shap_values(X_test)
    #
    #     #shap.summary_plot(shap_values, X_test, max_display=40)
    #     #plt.show()
    #
    #     shap_values_list.append(shap_values)
    #     X_test_list.append(X_test)
    #
    # all_shap_values = np.concatenate(shap_values_list)
    # all_x_test = pd.concat(X_test_list)
    #
    # np.save("Results\\"+label+"_shap.npy", all_shap_values)
    # all_x_test.to_pickle("Results\\"+label+"_xval.pkl")
    # shap.summary_plot(all_shap_values, all_x_test, max_display=40)
    # plt.show()

    print("%f, %f, %f, %f, %f, %f" % (np.average(auc_list), np.std(auc_list), np.average(mcc_list), np.std(mcc_list), np.average(acc_list), np.std(acc_list)))
    # print(np.average(auc_list))
    # print(np.average(mcc_list))
    # print(np.average(acc_list))
    #
    # print(np.std(auc_list))
    # print(np.std(mcc_list))
    # print(np.std(acc_list))









