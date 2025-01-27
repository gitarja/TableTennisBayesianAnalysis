import os

# This must happen before pymc is imported, so you might
# need to restart the kernel for it to take effect.
import pandas as pd

os.environ['OMP_NUM_THREADS'] = '7'
import sys

sys.path.append(os.path.dirname(__file__))
import numpy as np
from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.RegressionGlobalFeauresReader import RegressionGlobalDoubleFeaturesReader
from Utils.Conf import DOUBLE_SUMMARY_FILE_PATH
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from Utils.Conf import DOUBLE_FEATURES_FILE_PATH
import xgboost
from sklearn.metrics import mean_squared_error, r2_score
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
    def trainXGB(X, y, search_params=False):

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=1945)
        d_train = xgboost.DMatrix(X_train, label=y_train)
        d_val = xgboost.DMatrix(X_val, label=y_val)

        if search_params:



            params = {
                'max_depth': [5, 7, 10],
                'colsample_bytree': [.5, .7, .9],
                'alpha': [.25, .3, .5, .75],
                'subsample': [.5, .6, .8],
                'learning_rate': [0.01, 0.05],
            }

            skf = KFold(n_splits=5, shuffle=True, random_state=1945)

            model = xgboost.XGBRegressor(objective= "reg:pseudohubererror",  eval_metric="mphe")
            grid_search = GridSearchCV(model, param_grid=params, scoring="r2", n_jobs=4,
                                               cv=skf.split(X, y), verbose=3)

            grid_search.fit(X, y)
            print(grid_search.best_score_)
            print(grid_search.best_params_)
            exit()

        else:

            params = {
                "device": "cuda:0",
                "eta": 0.05,
                "objective": "binary:logistic",
                "subsample": 1.,
                "max_depth": 10,
                "eval_metric": "logloss",
                "max_delta_step": 5,
                "alpha": .5,
                "scale_pos_weight": 1.3

            }
            model = xgboost.train(
                params,
                d_train,
                5000,
                evals=[(d_val, "val")],
                verbose_eval=False,
                early_stopping_rounds=30,
                num_boost_round=100
            )



        return model


    def evaluateModel(model, X_test, y_test):
        d_test = xgboost.DMatrix(X_test, label=y_test)

        y_pred = model.predict(d_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)


        print("MSE", mse)
        print("R2", r2)
        print("-------------------------")
        return mse, r2

    np.random.seed(1945)  # For Replicability
    # control group

    avg_group, ineff_group, eff_group = groupClassifcation()

    avg_reader = RegressionGlobalDoubleFeaturesReader(file_path=DOUBLE_FEATURES_FILE_PATH,
                                            file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                            exclude_failure=False, exclude_no_pair=True, hmm_probs=True,
                                            include_subjects=eff_group)

    label = "ineff"

    avg_features = avg_reader.getStableUnstableFailureFeatures(group_name=label)

    X = avg_features.loc[:, avg_features.columns != 'labels']
    y = avg_features["labels"].values

    # find params
    model = trainXGB(X, y, search_params=True)

    # split data


    mse_list = []
    r2_list = []
    shap_values_list = []
    X_test_list = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1945)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # print(np.mean(y_train==0))
        # print(np.mean(y_train == 1))
        # print("*******************************")
        #create baseline model and test it
        model = trainXGB(X_train, y_train, search_params=False)

        mse, r2 = evaluateModel(model, X_test, y_test)

        mse_list.append(mse)
        r2_list.append(r2)










