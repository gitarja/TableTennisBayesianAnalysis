import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from Utils.Conf import DOUBLE_SUMMARY_FILE_PATH, SINGLE_SUMMARY_FILE_PATH

np.random.seed(1945)


def skillClassification(skill):
    skill_class = np.zeros_like(skill)
    skill_class[(skill > 0) & (skill <= 0.25)] = 0
    skill_class[(skill > 0.25) & (skill <= 0.5)] = 1
    skill_class[(skill > 0.5) & (skill <= 0.75)] = 2
    skill_class[skill > 0.75] = 3

    return skill_class

class DoubleSubjectCrossValidation:

    def __init__(self, n_fold=5):
        df = pd.read_csv(DOUBLE_SUMMARY_FILE_PATH)

        self.df = df[(df["norm_score"] > 0.55) & (df["Tobii_percentage"] > 65)]

        self.skf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=True)



    def getTrainTestData(self, repeat=1):
        subject_train = []
        subject_test = []
        for i in range(repeat):
            df = self.df
            X = df.values
            y = skillClassification(df["skill"].values)

            for j, (train_index, test_index) in enumerate(self.skf.split(X, y)):
                train_data = df.iloc[train_index].reset_index(drop=True).drop(df.filter(regex="Unnamed"),axis=1)
                test_data = df.iloc[test_index].reset_index(drop=True).drop(df.filter(regex="Unnamed"),axis=1)

                subject_train.append(train_data.loc[:, "file_name"].values)
                subject_test.append(test_data.loc[:, "file_name"].values)

        return subject_train, subject_test


    def getSummary(self):
        return self.df


class SubjectCrossValidation:


    def __init__(self, n_fold=5):
        df = pd.read_csv(SINGLE_SUMMARY_FILE_PATH)

        self.df = df[(df["norm_score"] > 0.5) & (df["Tobii_percentage"] > 65)]

        self.skf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=True)



    def getTrainTestData(self, repeat=1):
        subject_train = []
        subject_test = []
        for i in range(repeat):
            df = self.df
            X = df.values
            y = skillClassification(df["skill"].values)

            for j, (train_index, test_index) in enumerate(self.skf.split(X, y)):
                train_data = df.iloc[train_index].reset_index(drop=True).drop(df.filter(regex="Unnamed"),axis=1)
                test_data = df.iloc[test_index].reset_index(drop=True).drop(df.filter(regex="Unnamed"),axis=1)

                subject_train.append(train_data.loc[:, "Subject1"].values)
                subject_test.append(test_data.loc[:, "Subject1"].values)

        return subject_train, subject_test

    def getSummary(self):
        return self.df







if __name__ == '__main__':


    features_reader = DoubleSubjectCrossValidation()

    subject_train, subject_test = features_reader.getTrainTestData(5)

    for i in range(len(subject_test)):
        print(subject_test[i])