import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Utils.Conf import x_double_features_column, normalize_x_double_episode_columns, y_episode_column, DOUBLE_SUMMARY_FILE_PATH
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.impute import KNNImputer

np.random.seed(1945)


def toCategorical(y, num_classes=2):
    """ 1-hot encodes a tensor """
    y = y.astype(int)
    return np.eye(num_classes, dtype='uint8')[y]


class DoubleFeaturesReader:

    def __init__(self, file_path="", include_subjects=["test"], n_window=3, discretization=False):
        df = pd.read_pickle(file_path)

        self.mean = np.nanmean(
            df.loc[:, normalize_x_double_episode_columns], axis=0)
        self.std = np.nanstd(
            df.loc[:, normalize_x_double_episode_columns], axis=0)
        # select subjects subjects
        df = df.loc[df["session_id"].isin(include_subjects), :]
        df = df.loc[(df["session_id"] != "2022-12-19_A_T06") | (
                    df["session_id"] != "2023-02-15_M_T01")]  # session excluded, equipments fail
        # exclude -1 data
        df = df.loc[df["success"] != -1]

        # exclude episode without a pair
        # df = df.loc[df["pair_idx"] != -1]
        self.n_window = n_window
        self.df = df

    def getDF(self):

        return self.df

    def splitEpisode(self, v):
        min_seq = self.n_window
        X_sequence = []

        x_columns = x_double_features_column + y_episode_column
        if len(v) > min_seq:
            # for i in range(0, (len(v) - min_seq)+1):
            #     features = np.concatenate(v.iloc[i:(i + min_seq)][x_columns].values)
            #
            #     X_sequence.append(features)

            for i in range(0, 1):
                features = np.concatenate(v.iloc[-min_seq:][x_columns].values)

                X_sequence.append(features)
            colnames = []
            for t in range(min_seq):
                colnames.extend([(x, t) for x in x_columns])
            df = pd.DataFrame(np.asarray(X_sequence), columns=colnames)
            return df
        else:
            return None

    def constructEpisodes(self, df, train=False):
        subjects_group = df.groupby(['id_subject'])
        X_all = []
        for s in subjects_group:
            for e in s[1].groupby(['episode_label']):
                X_seq = self.splitEpisode(e[1])
                if X_seq is not None:
                    X_all.append(X_seq)

        X_all = pd.concat(X_all, ignore_index=True)
        return X_all

    def contructMixEpisode(self, df):

        subjects_group = df.groupby(['id_subject'])
        X_all = []
        y_all = []
        for s in subjects_group:
            X_seq, y_seq = self.splitEpisode(s[1], th=50, augment=20, min_seq=2)
            X_all = X_all + X_seq
            y_all = y_all + y_seq

        return X_all, y_all

    def getAllData(self, train=False):
        X1 = self.constructEpisodes(self.df, train)
        return X1

    def normalizeDF(self, df, display=False):
        df = df.copy()
        imputer = KNNImputer(n_neighbors=5)
        if display == False:
            df.loc[:, normalize_x_double_episode_columns] = (df.loc[:,
                                                             normalize_x_double_episode_columns] - self.mean) / self.std
            df.loc[:, x_double_features_column] = imputer.fit_transform(df.loc[:, x_double_features_column])


        return df

    def getIndividualObservationData(self, display=False, features_group="all", label=False):
        '''
        :param display:
        :param features_group:
        all : all combination
        per_ec : perception + execution
        per_im : perception + impact
        per : perception
        :return:
        '''
        df = self.normalizeDF(self.df, display)

        x_column = x_double_features_column

        if label:
            x_column = x_column + y_episode_column

        X = df.iloc[:][x_column]
        y = df.iloc[:][y_episode_column].values.ravel()

        return X, y, x_column


class SequentialFeaturesReader(Dataset):

    def __init__(self, file_path="", include_subjects=["test"], n_window=3, n_stride=5):
        df = pd.read_pickle(file_path)
        self.columns = x_double_features_column
        self.mean = np.nanmean(
            df.loc[:, self.columns], axis=0)
        self.std = np.nanstd(
            df.loc[:, self.columns], axis=0)
        df = self.normalizeDF(df)
        # select subjects subjects
        df = df.loc[df["session_id"].isin(include_subjects), :]
        df = df.loc[(df["session_id"] != "2022-12-19_A_T06") | (
                df["session_id"] != "2023-02-15_M_T01")]  # session excluded, equipments fail
        # exclude episode without a pair
        df = df.loc[df["pair_idx"] != -1]
        # exclude -1 data
        df = df.loc[df["success"] != -1]
        self.n_window = n_window
        self.n_stride = n_stride
        self.df = df

        self.X, self.y = self.getAllData()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'inputs': self.X[idx], 'label': self.y[idx]}

        return sample

    def normalizeDF(self, df, display=False):
        df = df.copy()

        if display == False:
            df.loc[:, self.columns] = (df.loc[:, self.columns] - self.mean) / self.std

            df = df.fillna(0)
        return df

    def getAllData(self):
        X, y = self.constructEpisodes(self.df)
        return X, y

    def constructEpisodes(self, df):
        subjects_group = df.groupby(['session_id'])
        X_all = []
        y_all = []
        for i, s in subjects_group:
            for e in s.groupby(['episode_label']):
                d = self.splitEpisode(e[1])
                if d is not None:
                    X_all.append(d[0])
                    y_all.append(d[1])

        X_all = np.vstack(X_all)
        y_all = np.vstack(y_all)
        return X_all, y_all

    def splitEpisode(self, v):
        min_seq = self.n_window
        X_sequence = []
        y_sequence = []

        if len(v) > min_seq:
            for i in range(5):
                stop = (len(v) - (self.n_stride * i))
                features = v.iloc[stop - min_seq:stop][self.columns].values
                label = v.iloc[stop - min_seq:stop][y_episode_column].values
                if len(features) != min_seq:
                    break
                X_sequence.append(features)
                y_sequence.append(label)

            X_sequence = np.asarray(X_sequence)
            y_sequence = np.asarray(y_sequence)
            return X_sequence, y_sequence
        else:
            return None


if __name__ == '__main__':
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double_episode_features.pkl"

    df = pd.read_csv(DOUBLE_SUMMARY_FILE_PATH)
    df = df[(df["Tobii_percentage"] > 65)]
    features_reader = SequentialFeaturesReader(path, df["file_name"].values, n_window=10, n_stride=10)

    features_reader.getAllData()
