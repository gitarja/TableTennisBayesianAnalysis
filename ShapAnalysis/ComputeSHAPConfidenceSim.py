from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH
from Double.GlobalFeaturesReader import ImpressionFeatures
import xgboost
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, balanced_accuracy_score, make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
import pandas as pd
import shap
from corr_shap import CorrExplainer
import matplotlib.pyplot as plt
from shap.utils._legacy import LogitLink

if __name__ == '__main__':


    def normalizeShap(arr):
        scaled_arr = arr / np.max(np.abs(arr))
        return scaled_arr


    def trainXGB(X, y):
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

        params = {
            "device": "cuda:0",
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "subsample": .75,
            "max_depth": 3,
            "eval_metric": "aucpr",
            "alpha": .05,
            "min_child_weight": 3,
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



np.random.seed(1945)  # For Replicability
# control group

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
lower_features = lower_reader.getImpressionFeatures(group="lower", mod="skill_action_perception_impact_personal")

upper_features = upper_reader.getImpressionFeatures(group="upper", mod="skill_action_perception_impact_personal")

X_lower = lower_features.loc[:, lower_features.columns != 'labels']
y_lower = lower_features["labels"].values

X_upper = upper_features.loc[:, upper_features.columns != 'labels']
y_upper = upper_features["labels"].values

X = pd.concat([X_lower, X_upper])
y = np.concatenate([y_lower, y_upper])

print(np.average(y == 1))
print(np.average(y == 0))

# split data

n_booststrap = 20

X_test_list = []
n_column = X.shape[1]
shap_values_norm_list = []
bootstrap_results = np.zeros((n_booststrap, n_column))  # times 3 for KFold
index = 0


shap_values_list = []
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1945)
for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    model_perm = trainXGB(X_train, y_train)
    explainer = CorrExplainer(model_perm.inplace_predict, X, sampling="gauss+empirical",
                                   link=LogitLink())
    for j in range(n_booststrap):
        resample_idx = np.random.choice(range(X_test.shape[0]), size=X_test.shape[0], replace=True)
        X_resample = X_test.iloc[resample_idx]
        shap_values = explainer.shap_values(X_resample)
        shap_values_list.append(shap_values)


# normalize SHAP
all_shap_values = normalizeShap(np.concatenate(shap_values_list))


np.save("Results\\"+label+"_bootstrap_shap.npy", all_shap_values)
