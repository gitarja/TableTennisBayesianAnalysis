import numpy as np
from scipy.special import logit, expit
import pandas as pd
from Utils.Conf import SINGLE_FEATURES_FILE_PATH, NORMALIZE_X_DOUBLE_EPISODE_COLUMNS, HMM_MODEL_PATH
from scipy import stats
from sklearn.impute import KNNImputer
import torch
from scipy.ndimage import label
from sklearn.preprocessing import StandardScaler





class RegressionGlobalDoubleFeaturesReader:

    def __init__(self, file_path="", file_summary_path="", include_subjects=None, exclude_failure=True,
                 exclude_no_pair=False, hmm_probs=False, filter_out=False):
        '''
        :param file_path:
        :param file_summary_path:
        :param include_subjects:
        :param exclude_failure:
        :param exclude_no_pair:
        :param hmm_probs:
        :param filter_out: used filter out if u have not excluded participants with the norm_score <= 0.55 & tobii_per <= 65
        '''

        self.df_summary = pd.read_csv(file_summary_path)

        self.df = pd.read_pickle(file_path)

        if hmm_probs:
            self.df = self.timeSeriesFeatures()

        if filter_out:
            df_summary = self.df_summary[
                (self.df_summary["norm_score"] > 0.55) & (self.df_summary["Tobii_percentage"] > 65)]

            self.df = self.df.loc[(self.df["session_id"].isin(df_summary["file_name"].values)), :]

        if include_subjects is not None:
            # select subjects subjects
            self.df = self.df.loc[self.df["session_id"].isin(include_subjects), :]

        if exclude_failure:
            # 0: failure
            # -1: stop
            self.df = self.df.loc[(self.df["success"] != 0) | (self.df["success"] != -1)]
        else:
            self.df = self.df.loc[self.df["success"] != -1]

        if exclude_no_pair:
            self.df = self.df.loc[self.df["pair_idx"] != -1]




    def getStableUnstableFailureFeatures(self, group_name="test"):
        '''
        a function that gives the features of current and previous feature to predict the next states: stable, unstable, and failure
        :param group_name:
        :return:
        '''

        def computeDeviations(x, ref_idx, real_idx):
                # print(np.nanmean(x.values[ref_idx]))
                return  x.values[real_idx]- np.nanmean(x.values[ref_idx])

        group_df = self.df.groupby(['session_id', 'episode_label'])

        # list of features
        receiver_p1_al_list = []
        receiver_p2_al_list = []
        receiver_p3_fx_list = []
        receiver_p1_cs_list = []
        receiver_p2_cs_list = []
        receiver_start_fs_list = []

        receiver_p1_al_onset_list = []
        receiver_p1_al_prec_list = []
        receiver_p1_al_mag_list = []
        receiver_p2_al_mag_list = []
        receiver_p2_al_onset_list = []
        receiver_p2_al_prec_list = []
        receiver_p3_fx_onset_list = []
        receiver_p3_fx_duration_list = []

        hand_movement_sim_list = []
        receiver_racket_to_root_list = []
        receiver_im_ball_updown_list = []
        receiver_im_racket_effect_list = []
        receiver_im_racket_dir_list = []

        hitter_fx_list = []
        hitter_at_and_after_hit_list = []
        hitter_fx_duration_list = []


        team_spatial_position_list = []


        receiver_list = []
        receiver_skill_list = []



        label_list = []

        for name, group in group_df:
            # n_data = len(group["hitter_pr_p2_al"]) - (n_segment - 1)
            # print(n_data)
            if len(group) > 3:
                group.sort_values(by=['observation_label'])



                success_states = group["success"].values == 1

                stable_probs = group["stable_probs"].values



                receiver_idx = np.arange(len(stable_probs) - 2)+1
                labels = stable_probs[receiver_idx+1]

                if len(receiver_idx) > 0:
                    hitter_idx = receiver_idx - 1

                    subjects = group[["id_subject1", "id_subject2"]].values[0]
                    skill_subjects = group[["skill_subject1", "skill_subject2"]].values[0]
                    # get receiver features from the current event
                    receiver_p1_al = group["receiver_pr_p1_al"].values[receiver_idx]
                    receiver_p2_al = group["receiver_pr_p2_al"].values[receiver_idx]
                    receiver_p3_fx = group["receiver_pr_p3_fx"].values[receiver_idx]
                    receiver_p1_cs = group["receiver_pr_p1_cs"].values[receiver_idx]
                    receiver_p2_cs = group["receiver_pr_p2_cs"].values[receiver_idx]

                    receiver_p1_al_onset = group["receiver_pr_p1_al_onset"].values[receiver_idx]
                    receiver_p1_al_prec = group["receiver_pr_p1_al_prec"].values[receiver_idx]
                    receiver_p1_al_mag = group["receiver_pr_p1_al_mag"].values[receiver_idx]
                    receiver_p2_al_mag = group["receiver_pr_p2_al_mag"].values[receiver_idx]
                    receiver_p2_al_onset = group["receiver_pr_p2_al_onset"].values[receiver_idx]
                    receiver_p2_al_prec = group["receiver_pr_p2_al_prec"].values[receiver_idx]
                    receiver_p3_fx_onset = group["receiver_pr_p3_fx_onset"].values[receiver_idx]
                    receiver_p3_fx_duration = group["receiver_pr_p3_fx_duration"].values[receiver_idx]

                    receiver_start_fs = group["receiver_ec_start_fs"].values[receiver_idx]
                    receiver_racket_to_root = computeDeviations(group["receiver_im_racket_to_root"], success_states, receiver_idx)
                    # receiver_racket_to_root = group["receiver_im_racket_to_root"].values[receiver_idx]

                    receiver_im_racket_dir = group["receiver_im_racket_dir"].values[receiver_idx]
                    receiver_im_ball_updown = group["receiver_im_ball_updown"].values[receiver_idx]
                    receiver_im_racket_effect = group["receiver_im_racket_effect"].values[receiver_idx]

                    hand_movement_sim = computeDeviations(group["hand_movement_sim_dtw"], success_states, receiver_idx)
                    # hand_movement_sim = group["hand_movement_sim_dtw"].values[receiver_idx]

                    receiver = subjects[group["receiver"].values[receiver_idx].astype(int)]
                    receiver_skill = skill_subjects[group["receiver"].values[receiver_idx].astype(int)]

                    # get hitter features from the previous event
                    hitter_fx = group["hitter_pr_p3_fx"].values[hitter_idx]
                    hitter_fx_duration = group["hitter_pr_p3_fx_duration"].values[hitter_idx]
                    hitter_at_and_after_hit = group["hitter_at_and_after_hit"].values[hitter_idx]


                    # team spatial position
                    team_spatial_position = group["team_spatial_position"].values[receiver_idx]

                    # append all features to list
                    receiver_p1_al_list.append(receiver_p1_al)
                    receiver_p2_al_list.append(receiver_p2_al)
                    receiver_p3_fx_list.append(receiver_p3_fx)
                    receiver_p1_cs_list.append(receiver_p1_cs)
                    receiver_p2_cs_list.append(receiver_p2_cs)

                    receiver_p1_al_onset_list.append(receiver_p1_al_onset)
                    receiver_p1_al_prec_list.append(receiver_p1_al_prec)
                    receiver_p1_al_mag_list.append(receiver_p1_al_mag)
                    receiver_p2_al_mag_list.append(receiver_p2_al_mag)
                    receiver_p2_al_onset_list.append(receiver_p2_al_onset)
                    receiver_p2_al_prec_list.append(receiver_p2_al_prec)
                    receiver_p3_fx_onset_list.append(receiver_p3_fx_onset)
                    receiver_p3_fx_duration_list.append(receiver_p3_fx_duration)

                    receiver_start_fs_list.append(receiver_start_fs)
                    receiver_racket_to_root_list.append(receiver_racket_to_root)
                    receiver_im_ball_updown_list.append(receiver_im_ball_updown)
                    receiver_im_racket_dir_list.append(receiver_im_racket_dir)
                    receiver_im_racket_effect_list.append(receiver_im_racket_effect)

                    hand_movement_sim_list.append(hand_movement_sim)



                    receiver_list.append(receiver)
                    receiver_skill_list.append(receiver_skill)
                    hitter_fx_list.append(hitter_fx)
                    hitter_at_and_after_hit_list.append(hitter_at_and_after_hit)
                    hitter_fx_duration_list.append(hitter_fx_duration)

                    team_spatial_position_list.append(team_spatial_position)


                    label_list.append(labels)

        fetures_summary = {
            "receiver_p1_al": np.concatenate(receiver_p1_al_list),
            "receiver_p2_al": np.concatenate(receiver_p2_al_list),
            "receiver_p3_fx": np.concatenate(receiver_p3_fx_list),
            "receiver_p1_cs": np.concatenate(receiver_p1_cs_list),
            "receiver_p2_cs": np.concatenate(receiver_p2_cs_list),


            "receiver_p1_al_onset": np.concatenate(receiver_p1_al_onset_list),
            "receiver_p1_al_prec": np.concatenate(receiver_p1_al_prec_list),
            "receiver_p1_al_mag": np.concatenate(receiver_p1_al_mag_list),

            "receiver_p2_al_mag": np.concatenate(receiver_p2_al_mag_list),
            "receiver_p2_al_onset": np.concatenate(receiver_p2_al_onset_list),
            "receiver_p2_al_prec": np.concatenate(receiver_p2_al_prec_list),

            "receiver_p3_fx_onset": np.concatenate(receiver_p3_fx_onset_list),
            "receiver_p3_fx_duration": np.concatenate(receiver_p3_fx_duration_list),

            "receiver_im_racket_dir": np.concatenate(receiver_im_racket_dir_list),
            "receiver_im_ball_updown": np.concatenate(receiver_im_ball_updown_list),
            "receiver_im_racket_effect": np.concatenate(receiver_im_racket_effect_list),
            "receiver_im_racket_to_root": np.concatenate(receiver_racket_to_root_list),
            "receiver_start_fs": np.concatenate(receiver_start_fs_list),
            "hand_movement_sim": np.concatenate(hand_movement_sim_list),

            "hitter_fx": np.concatenate(hitter_fx_list),
            "hitter_at_and_after_hit": np.concatenate(hitter_at_and_after_hit_list),
            "hitter_fx_duration": np.concatenate(hitter_fx_duration_list),
            "team_spatial_position": np.concatenate(team_spatial_position_list),


            "receiver_skill": np.concatenate(receiver_skill_list),

            "labels": np.concatenate(label_list),
        }

        return pd.DataFrame(fetures_summary)



    def timeSeriesFeatures(self):

        df_summary = self.df_summary
        df = self.df
        # df_summary = df_summary[(df_summary["norm_score"] > 0.55) & (df_summary["Tobii_percentage"] > 65)]

        df = df.loc[(df["session_id"].isin(df_summary["file_name"].values)),
             :]

        df = df.loc[(df["session_id"] != "2022-12-19_A_T06") | (
                df["session_id"] != "2023-02-15_M_T01")]  # session excluded, equipments fail

        # combine the bouncing features
        df["bouncing_point_dist_p1"] = df["s1_bouncing_point_dist_p1"].fillna(0).values + df[
            "s2_bouncing_point_dist_p1"].fillna(0).values
        df["bouncing_point_dist_p2"] = df["s1_bouncing_point_dist_p2"].fillna(0).values + df[
            "s2_bouncing_point_dist_p2"].fillna(0).values

        # normalize features
        mean = np.nanmean(
            df.loc[:, NORMALIZE_X_DOUBLE_EPISODE_COLUMNS], axis=0)
        std = np.nanstd(
            df.loc[:, NORMALIZE_X_DOUBLE_EPISODE_COLUMNS], axis=0)
        df.loc[:, NORMALIZE_X_DOUBLE_EPISODE_COLUMNS] = (df.loc[:,
                                                         NORMALIZE_X_DOUBLE_EPISODE_COLUMNS] - mean) / std

        # input missing values
        imputer = KNNImputer(n_neighbors=5)
        df.loc[:, NORMALIZE_X_DOUBLE_EPISODE_COLUMNS] = imputer.fit_transform(
            df.loc[:, NORMALIZE_X_DOUBLE_EPISODE_COLUMNS])

        selected_groups = df.groupby(["session_id", "episode_label"]).filter(lambda x: len(x) >= 3)

        grouped_episodes = selected_groups.groupby(["session_id", "episode_label"])

        self.df["stable_probs"] = -1.
        self.df["unstable_preds"] = -1.
        # load HMM model
        model = torch.load(HMM_MODEL_PATH)
        for i, g in grouped_episodes:
            # if i[0] == "2022-12-07_M_T03":
            #     print("error here")
            g = g.sort_values(by=['observation_label'])
            # g = g.loc[g["receiver"].values==1]
            signal1 = g["ball_speed_after_hit"].values
            #signal2 = g["ball_dir_after_hit"].values
            signal3 = g["bouncing_point_dist_p1"].values
            signal4 = g["bouncing_point_dist_p2"].values
            #signal5 = g["hitter_position_to_bouncing_point"].values

            unstable_prior_state = np.expand_dims(np.asarray([
                np.quantile(signal1, 0.85),
                np.quantile(signal3, 0.75),
                np.quantile(signal4, 0.75),


            ]), 0)

            if len(signal1) > 0:
                # X = np.vstack([signal1, signal2, signal3, signal4]).T
                X = np.vstack([unstable_prior_state, np.vstack([signal1,  signal3, signal4]).T])
                X = np.expand_dims(X, axis=0)
                probs = model.predict_proba(X).numpy()[0][1:, 0]  # 0: stable 1: unstable
                preds = model.predict(X).numpy()[0][1:]

                conditions = np.argwhere(
                    (self.df["session_id"] == g["session_id"].values[0]) & self.df["observation_label"].isin(
                        g["observation_label"].values)).flatten()
                self.df.loc[conditions, "stable_probs"] = probs
                self.df.loc[conditions, "unstable_preds"] = preds

        # set stable score for each group
        group_df = self.df.groupby(['session_id'])
        for name, group in group_df:
            self.df.loc[self.df.session_id == name[0], "team_stable_rate"] = np.average(
                group["unstable_preds"].values == 0)

        return self.df


if __name__ == '__main__':
    from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH

    # control group
    reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                        file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                        include_subjects=None, exclude_failure=True,
                                        exclude_no_pair=False)

    reader.timeSeriesFeatures()
