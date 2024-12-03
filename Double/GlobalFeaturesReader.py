import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit, expit
import pandas as pd
from Utils.Conf import SINGLE_FEATURES_FILE_PATH, NORMALIZE_X_DOUBLE_EPISODE_COLUMNS, HMM_MODEL_PATH
from scipy import stats
from sklearn.impute import KNNImputer
import torch
from scipy.ndimage import label
from scipy.spatial import distance

from sklearn.preprocessing import StandardScaler


class GlobalFeaturesReader:

    def __init__(self, single_summary, double_summary):

        self.single = single_summary
        self.double = double_summary

    def getSingleDoubleFeatures(self, log_scale=False, col="skill"):
        X = []
        y = []
        group_label = []
        for index, row in self.double.iterrows():
            subject1 = row["Subject1"]
            subject2 = row["Subject2"]

            subject1_skill = self.single.loc[self.single["Subject1"] == subject1][[col]].values
            subject2_skill = self.single.loc[self.single["Subject1"] == subject2][[col]].values

            pair_skill = row[col]

            group_name = row["file_name"]

            if (len(subject1_skill) > 0) & (len(subject2_skill) > 0):
                X.append(np.concatenate([subject1_skill[0], subject2_skill[0]]))
                y.append(pair_skill)
                group_label.append(group_name)

        if log_scale:
            return np.log2(np.vstack(X)), np.log2(np.asarray(y)), np.asarray(group_label)

        else:
            return np.vstack(X), np.asarray(y), np.asarray(group_label)


class GlobalDoubleFeaturesReader:

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

    def getGlobalStableUnstableFeatures(self, group_label="control", prefix=None):

        def computeDeviation(x, x_all):
            return np.nanmean(np.abs(x - np.nanmean(x_all)))

        single_df = pd.read_pickle(SINGLE_FEATURES_FILE_PATH)
        group_df = self.df.groupby(['session_id'])

        # recover
        recover_pr_p1_al = []
        recover_pr_p1_al_onset = []
        recover_pr_p1_al_prec = []
        recover_pr_p1_al_mag = []
        recover_pr_p1_cs = []

        recover_pr_p2_al = []
        recover_pr_p2_al_onset = []
        recover_pr_p2_al_prec = []
        recover_pr_p2_cs = []
        recover_pr_p2_al_mag = []

        recover_pursuit = []
        recover_pursuit_duration = []
        recover_pursuit_stability = []
        recover_pursuit_onset = []

        recover_gaze_entropy = []
        recover_gaze_ball_relDiv = []

        recover_hitter_pursuit = []
        recover_hitter_pursuit_duration = []

        recover_bouncing_point_var_p1 = []

        recover_start_fs_std = []
        recover_start_fs_mean = []
        recover_movement_sim = []
        recover_racket_ball_ratio = []

        group_skill = []
        subject_skill = []
        group_labels = []

        subject = []
        for name, group in group_df:
            # set the labels for unstable
            # unstable_idx = ((group["stable_probs"] <= 0.5)  & (group["stable_probs"] != -1)).astype(float)

            unstable_idx = group["unstable_preds"]
            unstable_diff = np.pad(np.diff(unstable_idx), (0, 1), "edge")
            stable_diff = np.pad(np.diff(unstable_idx), (1, 0), "edge")

            # receiver
            for i in range(2):
                receiver_idx = group["receiver"] == i
                hitter_idx = group["hitter"] == i

                # bouncing point variance
                # stay_unstable_receiver_idx = (receiver_idx) & (unstable_idx == 1)
                # stay_stable_receiver_idx = (receiver_idx) & (unstable_idx == 0)
                #
                # stay_unstable_hitter_idx = (hitter_idx) & (unstable_idx == 1)
                # stay_stable_hitter_idx = (hitter_idx) & (unstable_idx == 0)

                # stay_unstable_receiver_idx = (receiver_idx) & ((unstable_idx == 1) & (stable_diff == 0))
                # stay_stable_receiver_idx = (receiver_idx) & (unstable_diff == -1)
                #
                # stay_unstable_hitter_idx = (hitter_idx) & ((unstable_idx == 1) & (stable_diff == 0))
                # stay_stable_hitter_idx = (hitter_idx) & (unstable_diff == -1)
                #
                # stay_unstable_hitter_idx2 = (hitter_idx) & ((unstable_idx == 1) & (stable_diff == 0))
                # stay_stable_hitter_idx2 = (hitter_idx) & (stable_diff == -1)

                stay_unstable_receiver_idx = (receiver_idx) & (unstable_diff == -1)
                stay_stable_receiver_idx = (receiver_idx) & ((unstable_idx == 0) & (stable_diff == 0))

                stay_unstable_hitter_idx = (hitter_idx) & (unstable_diff == -1)
                stay_stable_hitter_idx = (hitter_idx) & ((unstable_idx == 0) & (stable_diff == 0))

                stay_unstable_hitter_idx2 = (hitter_idx) & (stable_diff == -1)
                stay_stable_hitter_idx2 = (hitter_idx) & ((unstable_idx == 0) & (stable_diff == 0))

                # stable (0) or unstable (1)
                for j in range(2):
                    if j == 0:
                        reciver_stat_idx = stay_stable_receiver_idx
                        hitter_stat_idx = stay_stable_hitter_idx
                        hitter_stat_idx2 = stay_stable_hitter_idx2
                        pref = "_" + prefix[0]
                    else:
                        reciver_stat_idx = stay_unstable_receiver_idx
                        hitter_stat_idx = stay_unstable_hitter_idx
                        hitter_stat_idx2 = stay_unstable_hitter_idx2
                        pref = "_" + prefix[1]

                    # recovered effort

                    # subject
                    if i == 0:
                        subject.append(group["id_subject1"].values[0])
                        subject_skill.append(group["skill_subject1"].values[0])
                        recover_bouncing_point_var_p1.append(
                            group[hitter_stat_idx2]["s1_bouncing_point_dist_p1"].mean())

                    else:
                        subject.append(group["id_subject2"].values[0])
                        subject_skill.append(group["skill_subject2"].values[0])
                        recover_bouncing_point_var_p1.append(
                            group[hitter_stat_idx2]["s2_bouncing_point_dist_p1"].mean())

                    # recover features
                    recover_pr_p1_al.append(group[reciver_stat_idx]["receiver_pr_p1_al"].mean())
                    recover_pr_p1_al_onset.append(group[reciver_stat_idx]["receiver_pr_p1_al_onset"].mean())
                    recover_pr_p1_al_mag.append(group[reciver_stat_idx]["receiver_pr_p1_al_mag"].mean())
                    recover_pr_p1_al_prec.append(group[reciver_stat_idx]["receiver_pr_p1_al_prec"].mean())
                    recover_pr_p1_cs.append(group[reciver_stat_idx]["receiver_pr_p1_cs"].mean())

                    recover_pr_p2_al.append(group[reciver_stat_idx]["receiver_pr_p2_al"].mean())
                    recover_pr_p2_al_onset.append(group[reciver_stat_idx]["receiver_pr_p2_al_onset"].mean())
                    recover_pr_p2_al_prec.append(group[reciver_stat_idx]["receiver_pr_p2_al_prec"].mean())
                    recover_pr_p2_cs.append(group[reciver_stat_idx]["receiver_pr_p2_cs"].mean())
                    recover_pr_p2_al_mag.append(group[reciver_stat_idx]["receiver_pr_p2_al_mag"].mean())

                    recover_pursuit.append(group[reciver_stat_idx]["receiver_pr_p3_fx"].mean())
                    recover_pursuit_duration.append(group[reciver_stat_idx]["receiver_pr_p3_fx_duration"].mean())
                    recover_pursuit_onset.append(group[reciver_stat_idx]["receiver_pr_p3_fx_onset"].mean())
                    recover_pursuit_stability.append(group[reciver_stat_idx]["receiver_pr_p3_stability"].mean())

                    recover_gaze_entropy.append(
                        group[reciver_stat_idx]["receiver_gaze_entropy"].replace([np.inf, -np.inf],
                                                                                 np.nan).dropna().mean())
                    recover_gaze_ball_relDiv.append(
                        group[reciver_stat_idx]["receiver_gaze_ball_relDiv"].replace([np.inf, -np.inf],
                                                                                     np.nan).dropna().mean())

                    recover_hitter_pursuit.append(group[hitter_stat_idx]["hitter_pr_p3_fx"].mean())
                    recover_hitter_pursuit_duration.append(group[hitter_stat_idx]["hitter_pr_p3_fx_duration"].mean())

                    recover_start_fs_std.append(group[reciver_stat_idx]["receiver_ec_start_fs"].std())
                    recover_start_fs_mean.append(group[reciver_stat_idx]["receiver_ec_start_fs"].mean())
                    recover_movement_sim.append(group[reciver_stat_idx]["hand_movement_sim_dtw"].mean())

                    # group_labels
                    group_labels.append(group_label + pref)
                    # fixed effect
                    group_skill.append(self.df_summary[self.df_summary["file_name"] == name[0]]["skill"].values[0])

        # recover_hitter_pursuit_duration = np.asarray(recover_hitter_pursuit_duration)
        # recover_hitter_pursuit_duration[np.isnan(recover_hitter_pursuit_duration)] = 0
        fetures_summary = {
            "recover_pr_p1_al": np.asarray(recover_pr_p1_al),
            "recover_pr_p1_al_onset": np.asarray(recover_pr_p1_al_onset),
            "recover_pr_p1_al_mag": np.asarray(recover_pr_p1_al_mag),
            "recover_pr_p1_cs": np.asarray(recover_pr_p1_cs),
            "recover_pr_p1_al_prec": np.asarray(recover_pr_p1_al_prec),

            "recover_pr_p2_al": np.asarray(recover_pr_p2_al),
            "recover_pr_p2_al_onset": np.asarray(recover_pr_p2_al_onset),
            "recover_pr_p2_al_prec": np.asarray(recover_pr_p2_al_prec),
            "recover_pr_p2_cs": np.asarray(recover_pr_p2_cs),
            "recover_pr_p2_al_mag": np.asarray(recover_pr_p2_al_mag),

            "recover_pursuit": np.asarray(recover_pursuit),
            "recover_pursuit_duration": np.asarray(recover_pursuit_duration),
            "recover_pursuit_onset": np.asarray(recover_pursuit_onset),
            "recover_pursuit_stability": np.asarray(recover_pursuit_stability),

            "recover_gaze_entropy": np.asarray(recover_gaze_entropy),
            "recover_gaze_ball_relDiv": np.asarray(recover_gaze_ball_relDiv),

            "recover_hitter_pursuit": np.asarray(recover_hitter_pursuit),
            "recover_hitter_pursuit_duration": np.asarray(recover_hitter_pursuit_duration),

            "recover_start_fs_std": np.asarray(recover_start_fs_std),
            "recover_start_fs_mean": np.asarray(recover_start_fs_mean),
            "recover_movement_sim": np.asarray(recover_movement_sim),
            "recover_bouncing_point_var_p1": np.asarray(recover_bouncing_point_var_p1),

            "group_skill": np.asarray(group_skill),
            "subject": np.asarray(subject),
            "subject_skill": np.asarray(subject_skill),
            "group": np.asarray(group_labels),
        }

        return pd.DataFrame(fetures_summary)

    def getGlobalFeatures(self, group_label="control"):
        def computeDeviation(x, x_all):
            return np.nanmean(np.abs(x - np.nanmean(x_all)))

        single_df = pd.read_pickle(SINGLE_FEATURES_FILE_PATH)
        group_df = self.df.groupby(['session_id'])

        # receiver
        receiver_p1_al = []
        receiver_p2_al = []
        receiver_p1_al_prec = []
        receiver_p1_al_onset = []
        receiver_p1_al_mag = []
        receiver_p2_al_prec = []
        receiver_p2_al_mag = []
        receiver_p2_al_onset = []
        receiver_p1_cs = []
        receiver_p2_cs = []
        receiver_pursuit = []
        receiver_pursuit_onset = []
        receiver_pursuit_duration = []

        receiver_start_fs = []

        # hitter
        hitter_p1_al = []
        hitter_p2_al = []
        hitter_p1_al_prec = []
        hitter_p1_al_mag = []
        hitter_p1_al_onset = []
        hitter_p2_al_prec = []
        hitter_p2_al_mag = []
        hitter_p2_al_onset = []
        hitter_p1_cs = []
        hitter_p2_cs = []
        hitter_pursuit = []
        hitter_pursuit_onset = []
        hitter_pursuit_duration = []

        hand_movement_sim = []
        receiver_fixation_racket_latency = []
        receiver_distance_eye_hand = []
        hitter_at_and_after_hit = []
        receiver_im_racket_dir = []
        receiver_im_ball_updown = []

        group_skill = []
        subject_skill = []

        subject = []
        for name, group in group_df:
            # group = group.loc[group["unstable_preds"] != -1]
            # set the labels for unstable
            unstable_idx = ((group["stable_probs"] <= 0.5) & (group["stable_probs"] != -1)).astype(float).values
            unstable_preds = ((group["unstable_preds"].values == 1) & (group["unstable_preds"].values != -1)).astype(
                float)
            failure_idx = np.argwhere(group["success"] == 0).flatten()

            # check U-> and S-F

            UF = (unstable_preds[failure_idx] == 1)
            SF = (unstable_preds[failure_idx] == 0)

            unstable_diff = np.pad(np.diff(unstable_idx), (0, 1), "edge")
            observable_diff = np.pad(np.diff(group["observation_label"].values), (0, 1), "edge")
            unstable_episode = np.asarray((unstable_diff == 0) & (unstable_idx == 1) & (observable_diff == 1))

            state_groups, num_groups = label(unstable_episode)

            # duration of states
            durations_list = []
            for i in np.unique(state_groups)[1:]:
                duration = np.sum(state_groups == i)
                durations_list.append(duration)

            # receiver
            for i in range(2):
                receiver_idx = np.argwhere(group["receiver"] == i).flatten()
                hitter_idx = np.argwhere(group["hitter"] == i).flatten()

                # normalize length
                if len(receiver_idx) < len(hitter_idx):
                    n = len(receiver_idx)

                else:
                    n = len(hitter_idx)
                receiver_idx = receiver_idx[:n]
                hitter_idx = hitter_idx[:n]

                receiver_p1_al.append(group["receiver_pr_p1_al"].values[receiver_idx])
                receiver_p2_al.append(group["receiver_pr_p2_al"].values[receiver_idx])
                receiver_pursuit.append(group["receiver_pr_p3_fx"].values[receiver_idx])
                receiver_pursuit_onset.append(group["receiver_pr_p3_fx_onset"].values[receiver_idx])
                receiver_pursuit_duration.append(group["receiver_pr_p3_fx_duration"].values[receiver_idx])
                receiver_p1_al_prec.append(group["receiver_pr_p1_al_prec"].values[receiver_idx])
                receiver_p1_al_onset.append(group["receiver_pr_p1_al_onset"].values[receiver_idx])
                receiver_p1_al_mag.append(group["receiver_pr_p1_al_mag"].values[receiver_idx])
                receiver_p2_al_prec.append(group["receiver_pr_p2_al_prec"].values[receiver_idx])
                receiver_p2_al_onset.append(group["receiver_pr_p2_al_onset"].values[receiver_idx])
                receiver_p2_al_mag.append(group["receiver_pr_p2_al_mag"].values[receiver_idx])
                receiver_p1_cs.append(group["receiver_pr_p1_cs"].values[receiver_idx])
                receiver_p2_cs.append(group["receiver_pr_p2_cs"].values[receiver_idx])

                # # AL in P1 precision
                #
                # # hitter
                hitter_p1_al.append(group["hitter_pr_p1_al"].values[hitter_idx])
                hitter_p2_al.append(group["hitter_pr_p2_al"].values[hitter_idx])
                hitter_pursuit.append(group["hitter_pr_p3_fx"].values[hitter_idx])
                hitter_pursuit_onset.append(group["hitter_pr_p3_fx_onset"].values[hitter_idx])
                hitter_pursuit_duration.append(group["hitter_pr_p3_fx_duration"].values[hitter_idx])
                hitter_p1_al_prec.append(group["hitter_pr_p1_al_prec"].values[hitter_idx])
                hitter_p1_al_mag.append(group["hitter_pr_p1_al_mag"].values[hitter_idx])
                hitter_p1_al_onset.append(group["hitter_pr_p1_al_onset"].values[hitter_idx])
                hitter_p2_al_prec.append(group["hitter_pr_p2_al_prec"].values[hitter_idx])
                hitter_p2_al_mag.append(group["hitter_pr_p2_al_mag"].values[hitter_idx])
                hitter_p2_al_onset.append(group["hitter_pr_p2_al_onset"].values[hitter_idx])
                hitter_p1_cs.append(group["hitter_pr_p1_cs"].values[hitter_idx])
                hitter_p2_cs.append(group["hitter_pr_p2_cs"].values[hitter_idx])

                # subject
                if i == 0:
                    subject.extend(group["id_subject1"].values[0] for _ in range(n))
                    subject_skill.extend(group["skill_subject1"].values[0] for _ in range(n))

                else:
                    subject.extend(group["id_subject2"].values[0] for _ in range(n))
                    subject_skill.extend(group["skill_subject2"].values[0] for _ in range(n))

                receiver_start_fs.append(group["receiver_ec_start_fs"].values[receiver_idx])
                hand_movement_sim.append(group["hand_movement_sim_dtw"].values[receiver_idx])
                receiver_fixation_racket_latency.append(group["receiver_fixation_racket_latency"].values[receiver_idx])
                receiver_distance_eye_hand.append(group["receiver_distance_eye_hand"].values[receiver_idx])
                hitter_at_and_after_hit.append(group["hitter_at_and_after_hit"].values[hitter_idx])
                receiver_im_racket_dir.append(group["receiver_im_racket_dir"].values[receiver_idx])
                receiver_im_ball_updown.append(group["receiver_im_ball_updown"].values[receiver_idx])

                # fixed effect
                group_skill.extend(
                    self.df_summary[self.df_summary["file_name"] == name[0]]["skill"].values[0] for _ in range(n))

        fetures_summary = {

            # receiver
            "receiver_p1_al": np.concatenate(receiver_p1_al),
            "receiver_p2_al": np.concatenate(receiver_p2_al),
            "receiver_p1_al_prec": np.concatenate(receiver_p1_al_prec),
            "receiver_p1_al_onset": np.concatenate(receiver_p1_al_onset),
            "receiver_p1_al_mag": np.concatenate(receiver_p1_al_mag),
            "receiver_p2_al_prec": np.concatenate(receiver_p2_al_prec),
            "receiver_p2_al_onset": np.concatenate(receiver_p2_al_onset),
            "receiver_p2_al_mag": np.concatenate(receiver_p2_al_mag),
            "receiver_p1_cs": np.concatenate(receiver_p1_cs),
            "receiver_p2_cs": np.concatenate(receiver_p2_cs),
            "receiver_pursuit": np.concatenate(receiver_pursuit),
            "receiver_pursuit_onset": np.concatenate(receiver_pursuit_onset),
            "receiver_pursuit_duration": np.concatenate(receiver_pursuit_duration),

            "receiver_start_fs": np.concatenate(receiver_start_fs),

            # hitter
            "hitter_p1_al": np.concatenate(hitter_p1_al),
            "hitter_p2_al": np.concatenate(hitter_p2_al),
            "hitter_p1_al_prec": np.concatenate(hitter_p1_al_prec),
            "hitter_p1_al_onset": np.concatenate(hitter_p1_al_onset),
            "hitter_p1_al_mag": np.concatenate(hitter_p1_al_mag),
            "hitter_p2_al_prec": np.concatenate(hitter_p2_al_prec),
            "hitter_p2_al_onset": np.concatenate(hitter_p2_al_onset),
            "hitter_p2_al_mag": np.concatenate(hitter_p2_al_mag),
            "hitter_p1_cs": np.concatenate(hitter_p1_cs),
            "hitter_p2_cs": np.concatenate(hitter_p2_cs),
            "hitter_pursuit": np.concatenate(hitter_pursuit),
            "hitter_pursuit_onset": np.concatenate(hitter_pursuit_onset),
            "hitter_pursuit_duration": np.concatenate(hitter_pursuit_duration),

            "hand_movement_sim": np.concatenate(hand_movement_sim),
            "receiver_fixation_racket_latency": np.concatenate(receiver_fixation_racket_latency),
            "receiver_distance_eye_hand": np.concatenate(receiver_distance_eye_hand),
            "hitter_at_and_after_hit": np.concatenate(hitter_at_and_after_hit),
            "receiver_im_racket_dir": np.concatenate(receiver_im_racket_dir),
            "receiver_im_ball_updown": np.concatenate(receiver_im_ball_updown),

            "group_skill": np.asarray(group_skill),
            "subject": np.asarray(subject),
            "subject_skill": np.asarray(subject_skill),
            "group": group_label
        }

        return pd.DataFrame(fetures_summary)

    def getSegmentateFeatures(self, group_label="control", n_segment=5):

        mean_value_p1_al_prec = self.df['receiver_pr_p1_al_prec'].mean()
        mean_value_p2_al_prec = self.df['receiver_pr_p2_al_prec'].mean()
        mean_value_hand_movement_sim_dtw = self.df['hand_movement_sim_dtw'].mean()

        # Replace NaNs in column S2 with the
        # mean of values in the same column
        self.df['receiver_pr_p1_al_prec'].fillna(value=mean_value_p1_al_prec, inplace=True)
        self.df['receiver_pr_p2_al_prec'].fillna(value=mean_value_p2_al_prec, inplace=True)
        self.df['hand_movement_sim_dtw'].fillna(value=mean_value_hand_movement_sim_dtw, inplace=True)

        group_df = self.df.groupby(['session_id'])

        hand_mov_sim = []
        hitter_pf_rate = []
        receiver_p1_al_prec = []
        receiver_p1_al = []
        receiver_cs_p1 = []
        receiver_p2_al_prec = []
        receiver_p2_al = []
        receiver_cs_p2 = []

        stable_state = []
        th_segments = []
        hitter = []
        receiver = []
        receiver_skill = []
        hitter_skill = []

        for name, group in group_df:
            # n_data = len(group["hitter_pr_p2_al"]) - (n_segment - 1)
            # print(n_data)
            group.sort_values(by=['observation_label'])
            hitter_pursuit_seg = group["hitter_pr_p3_fx"]
            hand_mov_sim_seg = group["hand_movement_sim_dtw"]

            stable_state_seg = group["unstable_preds"] == 0

            receiver_pr_p1_al_seg = group["receiver_pr_p1_al"]
            receiver_p1_al_prec_seg = group["receiver_pr_p1_al_prec"]
            receiver_cs_p1_seg = group["receiver_pr_p1_cs"]

            receiver_pr_p2_al_seg = group["receiver_pr_p2_al"]
            receiver_p2_al_prec_seg = group["receiver_pr_p2_al_prec"]
            receiver_cs_p2_seg = group["receiver_pr_p2_cs"]

            for i in range(2):
                receiver_idx = group["receiver"] == i
                hitter_idx = group["hitter"] == i

                n_data_receiver = len(
                    hand_mov_sim_seg[receiver_idx].rolling(window=n_segment, step=n_segment).mean().values[1:])
                n_data_hitter = len(
                    hitter_pursuit_seg[hitter_idx].rolling(window=n_segment, step=n_segment).mean().values[1:])
                hand_mov_sim.append(
                    hand_mov_sim_seg[receiver_idx].rolling(window=n_segment, step=n_segment).mean().values[1:])
                hitter_pf_rate.append(
                    hitter_pursuit_seg[hitter_idx].rolling(window=n_segment, step=n_segment).sum().values[1:])
                stable_state.append(
                    stable_state_seg[hitter_idx].rolling(window=n_segment, step=n_segment).sum().values[1:])

                # phase 1
                receiver_p1_al.append(
                    receiver_pr_p1_al_seg[receiver_idx].rolling(window=n_segment, step=n_segment).sum().values[1:])
                receiver_p1_al_prec.append(
                    receiver_p1_al_prec_seg[receiver_idx].rolling(window=n_segment, step=n_segment).mean().values[1:])
                receiver_cs_p1.append(
                    receiver_cs_p1_seg[receiver_idx].rolling(window=n_segment, step=n_segment).mean().values[1:])
                # phase 2
                receiver_p2_al.append(
                    receiver_pr_p2_al_seg[receiver_idx].rolling(window=n_segment, step=n_segment).sum().values[1:])
                receiver_p2_al_prec.append(
                    receiver_p2_al_prec_seg[receiver_idx].rolling(window=n_segment, step=n_segment).mean().values[1:])
                receiver_cs_p2.append(
                    receiver_cs_p2_seg[receiver_idx].rolling(window=n_segment, step=n_segment).mean().values[1:])

                th_segments.append((group["observation_label"][receiver_idx].rolling(
                    window=n_segment, step=n_segment).mean().values[1:]) / 100)
                # th_segments.append(np.ones(shape=(n_data, )))
                if i == 0:
                    # add receiver
                    receiver.append(group[receiver_idx]["id_subject1"].values[:n_data_receiver])
                    receiver_skill.append(group[receiver_idx]["skill_subject1"].values[:n_data_receiver])

                    # add hitter
                    hitter.append(group[hitter_idx]["id_subject1"].values[:n_data_hitter])
                    hitter_skill.append(group[hitter_idx]["skill_subject1"].values[:n_data_hitter])

                else:
                    receiver.append(group[receiver_idx]["id_subject2"].values[:n_data_receiver])
                    receiver_skill.append(group[receiver_idx]["skill_subject2"].values[:n_data_receiver])

                    hitter.append(group[hitter_idx]["id_subject2"].values[:n_data_hitter])
                    hitter_skill.append(group[hitter_idx]["skill_subject2"].values[:n_data_hitter])

        fetures_summary = {
            "hand_mov_sim": np.concatenate(hand_mov_sim),
            "hitter_pf_rate": np.concatenate(hitter_pf_rate),
            "receiver_al_p1_prec": np.concatenate(receiver_p1_al_prec),
            "receiver_al_p1": np.concatenate(receiver_p1_al),
            "receiver_cs_p1": np.concatenate(receiver_cs_p1),
            "receiver_al_p2_prec": np.concatenate(receiver_p2_al_prec),
            "receiver_al_p2": np.concatenate(receiver_p2_al),
            "receiver_cs_p2": np.concatenate(receiver_cs_p2),

            "stable_rate": np.concatenate(stable_state),

            "th_segments": np.concatenate(th_segments),

            "receiver": np.concatenate(receiver),
            "receiver_skill": np.concatenate(receiver_skill),
            "hitter": np.concatenate(hitter),
            "hitter_skill": np.concatenate(hitter_skill),
            "group": group_label,
        }

        return pd.DataFrame(fetures_summary)

    def getStableUnstableFailureFeatures(self, group_name="test", success_failure=False, mod="skill",
                                         with_control=False, timepoint=False):
        '''
        a function that gives the features of current and previous feature to predict the next states: stable, unstable, and failure
        :param group_name:
        :return:
        '''

        def computeDeviations(x, ref, real_idx):
            # print(np.nanmean(x.values[ref_idx]))
            return x.values[real_idx] - ref

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
        receiver_racket_to_ball_list = []
        receiver_ec_fs_racket_angle_vel_list = []
        receiver_im_ball_updown_list = []
        receiver_im_racket_effect_list = []
        receiver_im_racket_dir_list = []
        receiver_im_racket_ball_angle_list = []
        receiver_im_racket_ball_wrist_list = []
        receiver_im_ball_wrist_list = []

        receiver_fixation_racket_latency_list = []
        receiver_distance_eye_hand_list = []

        hitter_at_and_after_hit_list = []

        hitter_p1_al_list = []
        hitter_p1_al_onset_list = []
        hitter_p1_al_prec_list = []
        hitter_p1_al_mag_list = []
        hitter_p1_cs_list = []

        hitter_p2_al_list = []
        hitter_p2_al_onset_list = []
        hitter_p2_al_prec_list = []
        hitter_p2_al_mag_list = []
        hitter_p2_cs_list = []
        hitter_fx_list = []
        hitter_fx_onset_list = []
        hitter_fx_duration_list = []
        hitter_position_to_bouncing_point_list = []

        team_spatial_position_list = []

        receiver_list = []
        hitter_list = []
        receiver_skill_list = []

        receiver_timepoint_list = []
        hitter_timepoint_list = []

        label_list = []
        session_list = []

        for name, group in group_df:
            # n_data = len(group["hitter_pr_p2_al"]) - (n_segment - 1)
            # print(n_data)
            group_name = name[0]
            receiver_im_racket_to_root_mean = self.df[
                (self.df["session_id"] == group_name) & (
                        self.df["success"] == 1)]["receiver_im_racket_to_root"].median()
            hand_movement_sim_dtw_mean = self.df[(self.df["session_id"] == group_name) & (self.df["success"] == 1)][
                "hand_movement_sim_dtw"].median()

            if len(group) > 3:
                group.sort_values(by=['observation_label'])

                stable_probs = group["stable_probs"].values
                unstable_states = group["unstable_preds"].values
                unstable_diff = np.pad(np.diff(unstable_states), (0, 1), "edge")
                stable_diff = np.pad(np.diff(unstable_states), (1, 0), "edge")

                us_s = np.argwhere(unstable_diff == -1).flatten()  # from unstable to stable
                # us_s = np.argwhere((unstable_diff == 0) & (unstable_states == 0)).flatten()  # from stable to stable
                # us_us = np.argwhere((stable_diff == 1)  & (unstable_states == 1)).flatten()
                us_us = np.argwhere(
                    (unstable_diff == 0) & (unstable_states == 1)).flatten()  # from unstable to unstable
                s_s = np.argwhere(
                    (unstable_diff == 0) & (unstable_states == 0)).flatten()  # from stable to unstable

                us_f = np.argwhere((unstable_states == 1) & (group["success"] == 0) & (stable_probs < 0.1)).flatten()

                if success_failure:
                    us_s = np.argwhere(group["success"].values == 1).flatten()[1:]
                    us_us = np.argwhere(group["success"].values == 0).flatten()

                else:
                    # us_s = us_s[us_s > 0]
                    # us_us = us_us[us_us > 0]

                    s_s = s_s[s_s > 0]
                    us_us = us_us[us_us > 0]

                # concatenate index and create labels
                s_s_label = [1 for i in range(len(s_s))]
                us_s_label = [1 for i in range(len(us_s))]
                us_us_label = [0 for i in range(len(us_us))]
                us_f_label = [0 for i in range(len(us_f))]

                receiver_idx = np.concatenate([us_s, us_us])
                labels = np.concatenate([us_s_label, us_us_label])

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

                    # receiver_start_fs = computeDeviations(group["receiver_ec_start_fs"],
                    #                                             receiver_start_fs_mean,
                    #                                             receiver_idx)

                    # receiver_racket_to_root = computeDeviations(group["receiver_im_racket_to_root"],
                    #                                             receiver_im_racket_to_root_mean,
                    #                                             receiver_idx)
                    receiver_racket_to_ball = group["receiver_ec_fs_ball_rball_dist"].values[receiver_idx]
                    receiver_ec_fs_racket_angle_vel = group["receiver_ec_fs_racket_angle_vel"].values[receiver_idx]
                    receiver_racket_to_root = group["receiver_im_racket_to_root"].values[receiver_idx]

                    receiver_im_racket_dir = group["receiver_im_racket_dir"].values[receiver_idx]
                    receiver_im_ball_updown = group["receiver_im_ball_updown"].values[receiver_idx]
                    receiver_im_racket_effect = group["receiver_im_racket_effect"].values[receiver_idx]

                    receiver_im_racket_ball_angle = group["receiver_im_racket_ball_angle"].values[receiver_idx]
                    receiver_im_racket_ball_wrist = group["receiver_im_racket_ball_wrist"].values[receiver_idx]
                    receiver_im_ball_wrist = group["receiver_im_ball_wrist"].values[receiver_idx]

                    receiver_fixation_racket_latency = group["receiver_fixation_racket_latency"].values[receiver_idx]
                    receiver_distance_eye_hand = group["receiver_distance_eye_hand"].values[receiver_idx]

                    # hand_movement_sim = computeDeviations(group["hand_movement_sim_dtw"], hand_movement_sim_dtw_mean,
                    #                                        receiver_idx)
                    hand_movement_sim = group["hand_movement_sim_dtw"].values[receiver_idx]

                    receiver = subjects[group["receiver"].values[receiver_idx].astype(int)] + group_name
                    hitter = subjects[group["hitter"].values[hitter_idx].astype(int)]+ group_name
                    session = group["session_id"].values[receiver_idx]
                    receiver_skill = skill_subjects[group["receiver"].values[receiver_idx].astype(int)]

                    # get hitter features from the previous event
                    # hitter_rev_p1_al = group["hitter_pr_p1_al"].values[hitter_idx]
                    # hitter_rev_p1_al_onset = group["hitter_pr_p1_al_onset"].values[hitter_idx]
                    # hitter_rev_p1_al_prec = group["hitter_pr_p1_al_prec"].values[hitter_idx]
                    # hitter_rev_p1_al_mag = group["hitter_pr_p1_al_mag"].values[hitter_idx]
                    #
                    # hitter_rev_p2_al = group["hitter_pr_p2_al"].values[hitter_idx]
                    # hitter_rev_p2_al_onset = group["hitter_pr_p2_al_onset"].values[hitter_idx]
                    # hitter_rev_p2_al_prec = group["hitter_pr_p2_al_prec"].values[hitter_idx]
                    # hitter_rev_p2_al_mag = group["hitter_pr_p2_al_mag"].values[hitter_idx]
                    #
                    # hitter_rev_p1_cs = group["hitter_pr_p1_cs"].values[hitter_idx]
                    # hitter_rev_p2_cs = group["hitter_pr_p2_cs"].values[hitter_idx]
                    #
                    # hitter_rev_fx = group["hitter_pr_p3_fx"].values[
                    #     hitter_idx]  # what the hitter does before being receiver
                    # hitter_rev_fx_duration = group["hitter_pr_p3_fx_duration"].values[
                    #     hitter_idx]  # what the hitter does before being receiver

                    # team spatial position
                    team_spatial_position = group["team_spatial_position"].values[receiver_idx]

                    # get hitter features from current episodes
                    hitter_p1_al = group["hitter_pr_p1_al"].values[receiver_idx]
                    hitter_p1_al_onset = group["hitter_pr_p1_al_onset"].values[receiver_idx]
                    hitter_p1_al_prec = group["hitter_pr_p1_al_prec"].values[receiver_idx]
                    hitter_p1_al_mag = group["hitter_pr_p1_al_mag"].values[receiver_idx]
                    hitter_p1_cs = group["hitter_pr_p1_cs"].values[receiver_idx]

                    hitter_p2_al = group["hitter_pr_p2_al"].values[receiver_idx]
                    hitter_p2_al_onset = group["hitter_pr_p2_al_onset"].values[receiver_idx]
                    hitter_p2_al_prec = group["hitter_pr_p2_al_prec"].values[receiver_idx]
                    hitter_p2_al_mag = group["hitter_pr_p2_al_mag"].values[receiver_idx]
                    hitter_p2_cs = group["hitter_pr_p2_cs"].values[receiver_idx]
                    hitter_fx = group["hitter_pr_p3_fx"].values[receiver_idx]
                    hitter_fx_onset = group["hitter_pr_p3_fx_onset"].values[receiver_idx]
                    hitter_fx_duration = group["hitter_pr_p3_fx_duration"].values[receiver_idx]

                    hitter_position_to_bouncing_point = group["hitter_position_to_bouncing_point"].values[
                        receiver_idx]
                    hitter_at_and_after_hit = group["hitter_at_and_after_hit"].values[
                        receiver_idx]  # what the hitter does when the reciever takes an action

                    # time point
                    receiver_timepoint = group["observation_label"].values[receiver_idx]
                    hitter_timepoint = group["observation_label"].values[hitter_idx]

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
                    receiver_racket_to_ball_list.append(receiver_racket_to_ball)
                    receiver_ec_fs_racket_angle_vel_list.append(receiver_ec_fs_racket_angle_vel)

                    receiver_im_ball_updown_list.append(receiver_im_ball_updown)
                    receiver_im_racket_dir_list.append(receiver_im_racket_dir)
                    receiver_im_racket_effect_list.append(receiver_im_racket_effect)
                    receiver_im_racket_ball_angle_list.append(receiver_im_racket_ball_angle)
                    receiver_im_racket_ball_wrist_list.append(receiver_im_racket_ball_wrist)
                    receiver_im_ball_wrist_list.append(receiver_im_ball_wrist)

                    receiver_fixation_racket_latency_list.append(receiver_fixation_racket_latency)
                    receiver_distance_eye_hand_list.append(receiver_distance_eye_hand)

                    hand_movement_sim_list.append(hand_movement_sim)

                    receiver_list.append(receiver)
                    hitter_list.append(hitter)
                    session_list.append(session)
                    receiver_skill_list.append(receiver_skill)

                    receiver_timepoint_list.append(receiver_timepoint)
                    hitter_timepoint_list.append(hitter_timepoint)

                    # hitter from previous episode
                    # hitter_rev_p1_al_list.append(hitter_rev_p1_al)
                    # hitter_rev_p1_al_onset_list.append(hitter_rev_p1_al_onset)
                    # hitter_rev_p1_al_prec_list.append(hitter_rev_p1_al_prec)
                    # hitter_rev_p1_al_mag_list.append(hitter_rev_p1_al_mag)
                    #
                    # hitter_rev_p2_al_list.append(hitter_rev_p2_al)
                    # hitter_rev_p2_al_onset_list.append(hitter_rev_p2_al_onset)
                    # hitter_rev_p2_al_prec_list.append(hitter_rev_p2_al_prec)
                    # hitter_rev_p2_al_mag_list.append(hitter_rev_p2_al_mag)
                    #
                    # hitter_rev_p1_cs_list.append(hitter_rev_p1_cs)
                    # hitter_rev_p2_cs_list.append(hitter_rev_p2_cs)
                    #
                    # hitter_rev_fx_list.append(hitter_rev_fx)
                    # hitter_rev_fx_duration_list.append(hitter_rev_fx_duration)

                    # hitter from current episode
                    hitter_p1_al_list.append(hitter_p1_al)
                    hitter_p1_al_onset_list.append(hitter_p1_al_onset)
                    hitter_p1_al_prec_list.append(hitter_p1_al_prec)
                    hitter_p1_al_mag_list.append(hitter_p1_al_mag)
                    hitter_p1_cs_list.append(hitter_p1_cs)
                    hitter_p2_al_list.append(hitter_p2_al)
                    hitter_p2_al_onset_list.append(hitter_p2_al_onset)
                    hitter_p2_al_prec_list.append(hitter_p2_al_prec)
                    hitter_p2_al_mag_list.append(hitter_p2_al_mag)
                    hitter_p2_cs_list.append(hitter_p2_cs)
                    hitter_fx_list.append(hitter_fx)
                    hitter_fx_onset_list.append(hitter_fx_onset)
                    hitter_fx_duration_list.append(hitter_fx_duration)
                    hitter_position_to_bouncing_point_list.append(hitter_position_to_bouncing_point)

                    hitter_at_and_after_hit_list.append(hitter_at_and_after_hit)

                    team_spatial_position_list.append(team_spatial_position)

                    label_list.append(labels)

        fetures_summary = {
            "labels": np.concatenate(label_list),
        }
        if "skill" in mod:
            fetures_summary.update({
                "receiver_skill": np.concatenate(receiver_skill_list),
            })
        if "perception" in mod:
            fetures_summary.update({
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

                "hitter_p1_al": np.concatenate(hitter_p1_al_list),
                "hitter_p2_al": np.concatenate(hitter_p2_al_list),
                "hitter_fx": np.concatenate(hitter_fx_list),
                "hitter_p1_cs": np.concatenate(hitter_p1_cs_list),
                "hitter_p2_cs": np.concatenate(hitter_p2_cs_list),

                "hitter_p1_al_onset": np.concatenate(hitter_p1_al_onset_list),
                "hitter_p1_al_prec": np.concatenate(hitter_p1_al_prec_list),
                "hitter_p1_al_mag": np.concatenate(hitter_p1_al_mag_list),

                "hitter_p2_al_onset": np.concatenate(hitter_p2_al_onset_list),
                "hitter_p2_al_prec": np.concatenate(hitter_p2_al_prec_list),
                "hitter_p2_al_mag": np.concatenate(hitter_p2_al_mag_list),

                "hitter_fx_onset": np.concatenate(hitter_fx_onset_list),
                "hitter_fx_duration": np.concatenate(hitter_fx_duration_list),

            })
        if "action" in mod:
            fetures_summary.update({

                # action
                "receiver_im_racket_dir": np.concatenate(receiver_im_racket_dir_list),
                "receiver_im_ball_updown": np.concatenate(receiver_im_ball_updown_list),
                "receiver_start_fs": np.concatenate(receiver_start_fs_list),
                "hand_movement_sim": np.concatenate(hand_movement_sim_list),
                "receiver_fixation_racket_latency": np.concatenate(receiver_fixation_racket_latency_list),
                "receiver_distance_eye_hand": np.concatenate(receiver_distance_eye_hand_list),
                "hitter_at_and_after_hit": np.concatenate(hitter_at_and_after_hit_list),

            })
        if mod == "full_mode":
            fetures_summary.update({
                # perception
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

                "hitter_p1_al": np.concatenate(hitter_p1_al_list),
                "hitter_p2_al": np.concatenate(hitter_p2_al_list),
                "hitter_fx": np.concatenate(hitter_fx_list),
                "hitter_p1_cs": np.concatenate(hitter_p1_cs_list),
                "hitter_p2_cs": np.concatenate(hitter_p2_cs_list),

                "hitter_p1_al_onset": np.concatenate(hitter_p1_al_onset_list),
                "hitter_p1_al_prec": np.concatenate(hitter_p1_al_prec_list),
                "hitter_p1_al_mag": np.concatenate(hitter_p1_al_mag_list),

                "hitter_p2_al_onset": np.concatenate(hitter_p2_al_onset_list),
                "hitter_p2_al_prec": np.concatenate(hitter_p2_al_prec_list),
                "hitter_p2_al_mag": np.concatenate(hitter_p2_al_mag_list),

                "hitter_fx_onset": np.concatenate(hitter_fx_onset_list),
                "hitter_fx_duration": np.concatenate(hitter_fx_duration_list),

                # action
                "receiver_im_racket_dir": np.concatenate(receiver_im_racket_dir_list),
                "receiver_im_ball_updown": np.concatenate(receiver_im_ball_updown_list),
                "receiver_start_fs": np.concatenate(receiver_start_fs_list),
                "hand_movement_sim": np.concatenate(hand_movement_sim_list),
                "receiver_fixation_racket_latency": np.concatenate(receiver_fixation_racket_latency_list),
                "receiver_distance_eye_hand": np.concatenate(receiver_distance_eye_hand_list),
                "hitter_at_and_after_hit": np.concatenate(hitter_at_and_after_hit_list),
                # contact
                "receiver_im_racket_ball_angle": np.concatenate(receiver_im_racket_ball_angle_list),
                "receiver_im_racket_ball_wrist": np.concatenate(receiver_im_racket_ball_wrist_list),
                "receiver_im_ball_wrist": np.concatenate(receiver_im_ball_wrist_list),

                "receiver_skill": np.concatenate(receiver_skill_list),
                "labels": np.concatenate(label_list),

            })

        if with_control:
            fetures_summary.update({"receiver": np.concatenate(receiver_list), "session": np.concatenate(session_list)})

        if timepoint:
            fetures_summary.update({"receiver_timepoint": np.concatenate(receiver_timepoint_list),
                                    "hitter_timepoint": np.concatenate(hitter_timepoint_list),
                                    "receiver": np.concatenate(receiver_list),
                                    "hitter": np.concatenate(hitter_list)
                                    })

        return pd.DataFrame(fetures_summary)

    # def getPredictionsFeatures(self, n_segment=10, n_sub_seg=5, n_stride=2):
    #
    #     def rollingDiff(x, method="mean"):
    #         if method =="mean":
    #             z = x.rolling(window=n_sub_seg, step=n_stride).mean().values
    #             return np.average(np.diff(z[~np.isnan(z)]))
    #         elif method == "std":
    #             z = x.rolling(window=n_sub_seg, step=n_stride).std().values
    #             return np.mean(np.diff(z[~np.isnan(z)]))
    #
    #     mean_value_p1_al_prec = self.df.iloc[:n_segment]['receiver_pr_p1_al_prec'].mean()
    #     mean_value_p2_al_prec = self.df.iloc[:n_segment]['receiver_pr_p2_al_prec'].mean()
    #     mean_value_hand_movement_sim_dtw = self.df['hand_movement_sim_dtw'].mean()
    #
    #     # Replace NaNs in column S2 with the
    #     # mean of values in the same column
    #     self.df['receiver_pr_p1_al_prec'].fillna(value=mean_value_p1_al_prec, inplace=True)
    #     self.df['receiver_pr_p2_al_prec'].fillna(value=mean_value_p2_al_prec, inplace=True)
    #     self.df['hand_movement_sim_dtw'].fillna(method="ffill", inplace=True)
    #     self.df['receiver_pr_p3_fx_duration'].fillna(method="ffill", inplace=True)
    #     self.df['hitter_pr_p3_fx_duration'].fillna(method="ffill", inplace=True)
    #     self.df['receiver_pr_p1_al_mag'].fillna(method="ffill", inplace=True)
    #     self.df['receiver_pr_p2_al_mag'].fillna(method="ffill", inplace=True)
    #     self.df['receiver_pr_p1_al_onset'].fillna(method="ffill", inplace=True)
    #     self.df['receiver_pr_p2_al_onset'].fillna(method="ffill", inplace=True)
    #     self.df['receiver_pr_p3_fx_onset'].fillna(method="ffill", inplace=True)
    #
    #     self.df['hitter_pr_p1_al_prec'].fillna(value=mean_value_p1_al_prec, inplace=True)
    #     self.df['hitter_pr_p2_al_prec'].fillna(value=mean_value_p2_al_prec, inplace=True)
    #
    #     single_df = pd.read_pickle(SINGLE_FEATURES_FILE_PATH)
    #     group_df = self.df.groupby(['session_id'])
    #
    #     X = np.zeros((len(group_df), 29))
    #     y = np.zeros((len(group_df)))
    #     i = 0
    #     for name, group in group_df:
    #         # eye movement features
    #         # receiver
    #         # negative big difference correlate with higher number of stable state
    #         receiver_al_p1_rate = rollingDiff(group.iloc[:n_segment]["receiver_pr_p1_al"])
    #         receiver_al_p2_rate = rollingDiff(group.iloc[:n_segment]["receiver_pr_p2_al"])
    #         receiver_pursuit_rate = rollingDiff(group.iloc[:n_segment]["receiver_pr_p3_fx"])
    #
    #         receiver_pursuit_duration = rollingDiff(group.iloc[:n_segment]["receiver_pr_p3_fx_duration"])
    #         receiver_pursuit_onset = rollingDiff(group.iloc[:n_segment]["receiver_pr_p3_fx_onset"])
    #
    #         receiver_al_p1_prec = rollingDiff(group.iloc[:n_segment]["receiver_pr_p1_al_prec"])
    #         receiver_al_p2_prec = rollingDiff(group.iloc[:n_segment]["receiver_pr_p2_al_prec"])
    #         receiver_al_p1_mag = rollingDiff(group.iloc[:n_segment]["receiver_pr_p1_al_mag"])
    #         receiver_al_p2_mag = rollingDiff(group.iloc[:n_segment]["receiver_pr_p2_al_mag"])
    #         receiver_al_p1_onset = rollingDiff(group.iloc[:n_segment]["receiver_pr_p1_al_onset"])
    #         receiver_al_p2_onset = rollingDiff(group.iloc[:n_segment]["receiver_pr_p2_al_onset"])
    #         receiver_p1_cs = rollingDiff(group.iloc[:n_segment]["receiver_pr_p1_cs"])
    #         receiver_p2_cs = rollingDiff(group.iloc[:n_segment]["receiver_pr_p2_cs"])
    #
    #         # hitter
    #         hitter_al_p1_rate = rollingDiff(group.iloc[:n_segment]["hitter_pr_p1_al"])
    #         hitter_al_p2_rate = rollingDiff(group.iloc[:n_segment]["hitter_pr_p2_al"])
    #         hitter_pursuit_rate = rollingDiff(group.iloc[:n_segment]["hitter_pr_p3_fx"])
    #
    #         hitter_al_p1_prec = rollingDiff(group.iloc[:n_segment]["hitter_pr_p1_al_prec"])
    #         hitter_al_p2_prec = rollingDiff(group.iloc[:n_segment]["hitter_pr_p2_al_prec"])
    #
    #         # body movement features
    #         receiver_fs_ball_racket_ratio = rollingDiff(group.iloc[:n_segment]["receiver_ec_fs_ball_racket_ratio"])
    #         receiver_racket_force = np.std(group.iloc[:n_segment]["receiver_im_racket_force"])
    #         receiver_fs_ball_racket_dir = np.std(group.iloc[:n_segment]["receiver_ec_fs_ball_racket_dir"])
    #         receiver_racket_to_root = np.std(group.iloc[:n_segment]["receiver_racket_to_root"])
    #
    #         hand_movement_sim = rollingDiff(group.iloc[:n_segment]["hand_movement_sim_dtw"], method="std")
    #
    #         team_spatial_position = np.std(group.iloc[:n_segment]["team_spatial_position"])
    #
    #
    #         # skill
    #         skill1 = group["skill_subject1"].values[0]
    #         skill2 = group["skill_subject2"].values[0]
    #
    #         # subject sim
    #         subject_1_sample = single_df.loc[single_df["id_subject"] == group["id_subject1"].values[0]]
    #         subject_2_sample = single_df.loc[single_df["id_subject"] == group["id_subject2"].values[0]]
    #
    #         sim_score = stats.ks_2samp(subject_1_sample["ec_start_fs"].values,
    #                                    subject_2_sample["ec_start_fs"].values)
    #         double_force = np.average(group.iloc[:n_segment]["receiver_im_racket_force"].values)
    #         force_change1 = np.average(subject_1_sample["im_racket_force"].values - double_force)
    #         force_change2 = np.average(subject_2_sample["im_racket_force"].values - double_force)
    #         # stable rate
    #         # stable_rate = np.average(group["unstable_preds"] == 0)
    #         stable_rate = group["team_stable_rate"].values[0]
    #
    #         X[i] = [hitter_pursuit_rate, receiver_al_p1_rate, receiver_al_p2_rate, receiver_al_p1_prec,
    #                 receiver_al_p2_prec, receiver_p1_cs, receiver_p2_cs, hand_movement_sim, receiver_pursuit_rate,
    #                 receiver_pursuit_duration,  receiver_al_p1_mag, receiver_al_p2_mag,
    #                 receiver_al_p1_onset,
    #                 receiver_al_p2_onset, receiver_fs_ball_racket_ratio, receiver_racket_force, receiver_pursuit_onset,
    #                 receiver_fs_ball_racket_dir, hitter_al_p1_rate, hitter_al_p2_rate, hitter_al_p1_prec,
    #                 hitter_al_p2_prec, skill1, skill2, sim_score.statistic, force_change1, force_change2, receiver_racket_to_root,
    #                 team_spatial_position
    #                 ]
    #         y[i] = stable_rate
    #         i += 1
    #
    #     # scaler = StandardScaler()
    #     # y = scaler.fit_transform(y.reshape(-1, 1))
    #     # import matplotlib.pyplot as plt
    #     # plt.hist(y)
    #     # plt.show()
    #     return X, y

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

        selected_groups = df.groupby(["session_id", "episode_label"]).filter(lambda x: len(x) >= 5)

        grouped_episodes = selected_groups.groupby(["session_id", "episode_label"])

        self.df["stable_probs"] = -1.
        self.df["unstable_preds"] = -1.
        # load HMM model
        # model = torch.load(HMM_MODEL_PATH)
        # for i, g in grouped_episodes:
        #     # if i[0] == "2022-12-07_M_T03":
        #     #     print("error here")
        #     g = g.sort_values(by=['observation_label'])
        #     # g = g.loc[g["receiver"].values==1]
        #     signal1 = g["ball_speed_after_hit"].values
        #     # signal2 = g["ball_dir_after_hit"].values
        #     signal3 = g["bouncing_point_dist_p1"].values
        #     signal4 = g["bouncing_point_dist_p2"].values
        #     # signal5 = g["hitter_position_to_bouncing_point"].values
        #
        #     unstable_prior_state = np.expand_dims(np.asarray([
        #         np.quantile(signal1, 0.85),
        #         np.quantile(signal3, 0.75),
        #         np.quantile(signal4, 0.75),
        #
        #     ]), 0)
        #
        #     if len(signal1) > 0:
        #         X = np.vstack([unstable_prior_state, np.vstack([signal1, signal3, signal4]).T])
        #         X = np.expand_dims(X, axis=0)
        #         probs = model.predict_proba(X).numpy()[0][1:, 0]  # 0: stable 1: unstable
        #         preds = model.predict(X).numpy()[0][1:]
        #
        #         # X = np.vstack([signal1, signal3, signal4]).T
        #         # X = np.expand_dims(X, axis=0)
        #         # probs = model.predict_proba(X).numpy()[0][:, 0] # 0: stable 1: unstable
        #         # preds = model.predict(X).numpy()[0]
        #
        #         conditions = np.argwhere(
        #             (self.df["session_id"] == g["session_id"].values[0]) & self.df["observation_label"].isin(
        #                 g["observation_label"].values)).flatten()
        #         self.df.loc[conditions, "stable_probs"] = probs
        #         self.df.loc[conditions, "unstable_preds"] = preds

        # set stable score for each group
        # group_df = self.df.groupby(['session_id'])
        # for name, group in group_df:
        #     self.df.loc[self.df.session_id == name[0], "team_stable_rate"] = np.average(
        #         group["unstable_preds"].values == 0)

        return self.df


class ImpressionFeatures:

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

        self.single_df = pd.read_pickle(SINGLE_FEATURES_FILE_PATH)

        if filter_out:
            df_summary = self.df_summary[
                (self.df_summary["norm_score"] > 0.55) & (self.df_summary["Tobii_percentage"] > 65)]

            self.df = self.df.loc[(self.df["session_id"].isin(df_summary["file_name"].values)), :]

        if include_subjects is not None:
            # select subjects subjects
            self.df = self.df.loc[self.df["session_id"].isin(include_subjects), :]
            self.df_summary = self.df_summary.loc[self.df_summary["file_name"].isin(include_subjects), :]

            # import arviz as az
            #
            # az.plot_dist(self.df_summary["n_success"].values, rug=True, kind="kde")
            # plt.show()

        if exclude_failure:
            # 0: failure
            # -1: stop
            self.df = self.df.loc[self.df["success"] == 1]
            self.single_df = self.single_df.loc[self.single_df["success"] == 1]
        else:
            self.df = self.df.loc[self.df["success"] != -1]

        if exclude_no_pair:
            self.df = self.df.loc[self.df["pair_idx"] != -1]

        # self.single_df = self.single_df.loc[(self.single_df["success"] != 0) | (self.single_df["success"] != -1)]

    def getImpressionFeatures(self, n_index=10, group="control", mod="skill"):
        if group == "lower":
            y = 0
        else:
            y = 1

        def computeStyleSim(features_name, s1, s2, bins=7, is_int=False):
            # a = stats.ks_2samp(self.single_df[self.single_df["id_subject"] == s1][features_name].values,
            #                self.single_df[self.single_df["id_subject"] == s2][features_name].values, )
            #
            # return a.statistic
            x1 = self.single_df[self.single_df["id_subject"] == s1][features_name].values
            x2 = self.single_df[self.single_df["id_subject"] == s2][features_name].values

            x1 = x1[~np.isnan(x1)]
            x2 = x2[~np.isnan(x2)]
            n_min = len(x2) if len(x1) > len(x2) else len(x1)
            x1 = x1[:n_min]
            x2 = x2[:n_min]
            x = np.concatenate([x1, x2])
            if is_int:
                max_x = np.max(x) + 1
                data1 = np.histogram(x1, bins=np.arange(max_x), density=True)[0]
                data2 = np.histogram(x2, bins=np.arange(max_x), density=True)[0]
            else:
                data1 = np.histogram(x1, bins=bins, range=(np.min(x), np.max(x)), density=True)[0]
                data2 = np.histogram(x2, bins=bins, range=(np.min(x), np.max(x)), density=True)[0]

            bc_coeff = np.sum(np.sqrt(data1 * data2))
            bhattacharyya_distance = -np.log(bc_coeff)

            return bhattacharyya_distance

        def computeMeanFeatures(features_name, s1, s2):
            single_df = self.single_df
            df1 = single_df[single_df["id_subject"] == s1][features_name].values
            df2 = single_df[single_df["id_subject"] == s2][features_name].values
            # n_min = len(df2) if len(df1) > len(df2) else len(df1)
            return 0.5 * (np.nanmean(df1) + np.nanmean(df2))
            # return np.nanmean(np.concatenate([df1, df2]))

        p1_al_on_sim_list = []
        p1_al_prec_sim_list = []
        p1_al_gM_sim_list = []
        p1_cs_sim_list = []

        # p2
        p2_al_on_sim_list = []
        p2_al_prec_sim_list = []
        p2_al_gM_sim_list = []
        p2_cs_sim_list = []

        # p3
        p3_fx_on_sim_list = []
        p3_fx_du_sim_list = []

        # action
        ec_start_fs_sim_list = []
        fixation_racket_latency_sim_list = []
        distance_eye_hand_sim_list = []
        im_ball_updown_sim_list = []

        # impact
        im_racket_ball_angle_sim_list = []
        im_racket_ball_wrist_sim_list = []
        im_ball_wrist_sim_list = []

        # mean
        p1_al_on_mean_list = []
        p1_al_prec_mean_list = []
        p1_al_gM_mean_list = []
        p1_cs_mean_list = []

        # p2
        p2_al_on_mean_list = []
        p2_al_prec_mean_list = []
        p2_al_gM_mean_list = []
        p2_cs_mean_list = []

        # p3
        p3_fx_on_mean_list = []
        p3_fx_du_mean_list = []

        # action
        ec_start_fs_mean_list = []
        im_ball_updown_mean_list = []
        fixation_racket_latency_mean_list = []
        distance_eye_hand_mean_list = []

        # impact
        im_racket_ball_angle_mean_list = []
        im_racket_ball_wrist_mean_list = []
        im_ball_wrist_mean_list = []

        # skills
        subject_skill_list = []
        for _, g in self.df_summary.iterrows():
            s1 = g["Subject1"]
            s2 = g["Subject2"]

            # similarity
            # p1
            p1_al_on_sim = computeStyleSim("pr_p1_al_on", s1, s2)
            p1_al_prec_sim = computeStyleSim("pr_p1_al_prec", s1, s2)
            p1_al_gM_sim = computeStyleSim("pr_p1_al_gM", s1, s2)
            p1_cs_sim = computeStyleSim("pr_p1_sf", s1, s2, is_int=True)

            # p2
            p2_al_on_sim = computeStyleSim("pr_p2_al_on", s1, s2)
            p2_al_prec_sim = computeStyleSim("pr_p2_al_prec", s1, s2)
            p2_al_gM_sim = computeStyleSim("pr_p2_al_gM", s1, s2)
            p2_cs_sim = computeStyleSim("pr_p2_sf", s1, s2, is_int=True)

            # p3
            p3_fx_on_sim = computeStyleSim("pr_p3_fx_on", s1, s2)
            p3_fx_du_sim = computeStyleSim("pr_p3_fx_du", s1, s2)

            # action
            ec_start_fs_sim = computeStyleSim("ec_start_fs", s1, s2, bins=3)
            fixation_racket_latency_sim = computeStyleSim("fixation_racket_latency", s1, s2, bins=5)
            distance_eye_hand_sim = computeStyleSim("distance_eye_hand", s1, s2, bins=5)
            im_ball_updown_sim = computeStyleSim("im_ball_updown", s1, s2, bins=5)

            # impact
            im_racket_ball_angle_sim = computeStyleSim("im_racket_ball_angle", s1, s2, bins=5)
            im_racket_ball_wrist_sim = computeStyleSim("im_racket_ball_wrist", s1, s2, bins=5)
            im_ball_wrist_sim = computeStyleSim("im_ball_wrist", s1, s2, bins=5)

            # means
            p1_al_on_mean = computeMeanFeatures("pr_p1_al_on", s1, s2)
            p1_al_prec_mean = computeMeanFeatures("pr_p1_al_prec", s1, s2)
            p1_al_gM_mean = computeMeanFeatures("pr_p1_al_gM", s1, s2)
            p1_cs_mean = computeMeanFeatures("pr_p1_sf", s1, s2)

            p2_al_on_mean = computeMeanFeatures("pr_p2_al_on", s1, s2)
            p2_al_prec_mean = computeMeanFeatures("pr_p2_al_prec", s1, s2)
            p2_al_gM_mean = computeMeanFeatures("pr_p2_al_gM", s1, s2)
            p2_cs_mean = computeMeanFeatures("pr_p2_sf", s1, s2)

            p3_fx_on_mean = computeMeanFeatures("pr_p3_fx_on", s1, s2)
            p3_fx_du_mean = computeMeanFeatures("pr_p3_fx_du", s1, s2)

            # action
            ec_start_fs_mean = computeMeanFeatures("ec_start_fs", s1, s2)
            fixation_racket_latency_mean = computeMeanFeatures("fixation_racket_latency", s1, s2)
            distance_eye_hand_mean = computeMeanFeatures("distance_eye_hand", s1, s2)
            im_ball_updown_mean = computeMeanFeatures("im_ball_updown", s1, s2)

            # impact
            im_racket_ball_angle_mean = computeMeanFeatures("im_racket_ball_angle", s1, s2)
            im_racket_ball_wrist_mean = computeMeanFeatures("im_racket_ball_wrist", s1, s2)
            im_ball_wrist_mean = computeMeanFeatures("im_ball_wrist", s1, s2)

            # skills
            subject_1_skill = self.single_df[self.single_df["id_subject"] == s1]["skill_subject"].values[0]
            subject_2_skill = self.single_df[self.single_df["id_subject"] == s2]["skill_subject"].values[0]

            # append features to list
            p1_al_on_sim_list.append(p1_al_on_sim)
            p1_al_prec_sim_list.append(p1_al_prec_sim)
            p1_al_gM_sim_list.append(p1_al_gM_sim)
            p1_cs_sim_list.append(p1_cs_sim)

            # p2
            p2_al_on_sim_list.append(p2_al_on_sim)
            p2_al_prec_sim_list.append(p2_al_prec_sim)
            p2_al_gM_sim_list.append(p2_al_gM_sim)
            p2_cs_sim_list.append(p2_cs_sim)

            # p3
            p3_fx_on_sim_list.append(p3_fx_on_sim)
            p3_fx_du_sim_list.append(p3_fx_du_sim)

            # action
            ec_start_fs_sim_list.append(ec_start_fs_sim)
            fixation_racket_latency_sim_list.append(fixation_racket_latency_sim)
            distance_eye_hand_sim_list.append(distance_eye_hand_sim)
            im_ball_updown_sim_list.append(im_ball_updown_sim)

            # impact
            im_racket_ball_angle_sim_list.append(im_racket_ball_angle_sim)
            im_racket_ball_wrist_sim_list.append(im_racket_ball_wrist_sim)
            im_ball_wrist_sim_list.append(im_ball_wrist_sim)

            # skill
            subject_skill_list.append(0.5 * (subject_1_skill + subject_2_skill))

            p1_al_on_mean_list.append(p1_al_on_mean)
            p1_al_prec_mean_list.append(p1_al_prec_mean)
            p1_al_gM_mean_list.append(p1_al_gM_mean)
            p1_cs_mean_list.append(p1_cs_mean)

            # p2
            p2_al_on_mean_list.append(p2_al_on_mean)
            p2_al_prec_mean_list.append(p2_al_prec_mean)
            p2_al_gM_mean_list.append(p2_al_gM_mean)
            p2_cs_mean_list.append(p2_cs_mean)

            # p3
            p3_fx_on_mean_list.append(p3_fx_on_mean)
            p3_fx_du_mean_list.append(p3_fx_du_mean)

            # action
            ec_start_fs_mean_list.append(ec_start_fs_mean)
            fixation_racket_latency_mean_list.append(fixation_racket_latency_mean)
            distance_eye_hand_mean_list.append(distance_eye_hand_mean)
            im_ball_updown_mean_list.append(im_ball_updown_mean)

            # impact
            im_racket_ball_angle_mean_list.append(im_racket_ball_angle_mean)
            im_racket_ball_wrist_mean_list.append(im_racket_ball_wrist_mean)
            im_ball_wrist_mean_list.append(im_ball_wrist_mean)

        if mod == "skill":
            fetures_summary = {
                "subject_skill": subject_skill_list,

                "labels": y,
            }

        elif mod == "skill_perception":
            fetures_summary = {
                "p1_al_onset_sim": p1_al_on_sim_list,
                "p1_al_prec_sim": p1_al_prec_sim_list,
                "p1_al_mag_sim": p1_al_gM_sim_list,

                "p2_al_onset_sim": p2_al_on_sim_list,
                "p2_al_prec_sim": p2_al_prec_sim_list,
                "p2_al_mag_sim": p2_al_gM_sim_list,

                "p3_fx_onset_sim": p3_fx_on_sim_list,
                "p3_fx_du_sim": p3_fx_du_sim_list,

                "p1_cs_sim": p1_cs_sim_list,
                "p2_cs_sim": p2_cs_sim_list,

                # mean
                "p1_al_onset_mean": p1_al_on_mean_list,
                "p1_al_prec_mean": p1_al_prec_mean_list,
                "p1_al_mag_mean": p1_al_gM_mean_list,
                "p1_cs_mean": p1_cs_mean_list,

                # p2
                "p2_al_onset_mean": p2_al_on_mean_list,
                "p2_al_prec_mean": p2_al_prec_mean_list,
                "p2_al_mag_mean": p2_al_gM_mean_list,
                "p2_cs_mean": p2_cs_mean_list,

                # p3
                "p3_fx_onset_mean": p3_fx_on_mean_list,
                "p3_fx_du_mean": p3_fx_du_mean_list,

                "subject_skill": subject_skill_list,

                "labels": y,
            }
        elif mod == "skill_perception_action":
            fetures_summary = {
                "p1_al_onset_sim": p1_al_on_sim_list,
                "p1_al_prec_sim": p1_al_prec_sim_list,
                "p1_al_mag_sim": p1_al_gM_sim_list,

                "p2_al_onset_sim": p2_al_on_sim_list,
                "p2_al_prec_sim": p2_al_prec_sim_list,
                "p2_al_mag_sim": p2_al_gM_sim_list,

                "p3_fx_onset_sim": p3_fx_on_sim_list,
                "p3_fx_du_sim": p3_fx_du_sim_list,

                "p1_cs_sim": p1_cs_sim_list,
                "p2_cs_sim": p2_cs_sim_list,

                # mean
                "p1_al_onset_mean": p1_al_on_mean_list,
                "p1_al_prec_mean": p1_al_prec_mean_list,
                "p1_al_mag_mean": p1_al_gM_mean_list,
                "p1_cs_mean": p1_cs_mean_list,

                # p2
                "p2_al_onset_mean": p2_al_on_mean_list,
                "p2_al_prec_mean": p2_al_prec_mean_list,
                "p2_al_mag_mean": p2_al_gM_mean_list,
                "p2_cs_mean": p2_cs_mean_list,

                # p3
                "p3_fx_onset_mean": p3_fx_on_mean_list,
                "p3_fx_du_mean": p3_fx_du_mean_list,

                # action
                "ec_start_fs_sim": ec_start_fs_sim_list,
                "fixation_racket_latency_sim": fixation_racket_latency_sim_list,
                "distance_eye_hand_sim": distance_eye_hand_sim_list,
                "im_ball_updown_sim": im_ball_updown_sim_list,

                "ec_start_fs_mean": ec_start_fs_mean_list,
                "fixation_racket_latency_mean": fixation_racket_latency_mean_list,
                "distance_eye_hand_mean": distance_eye_hand_mean_list,
                "im_ball_updown_mean": im_ball_updown_mean_list,

                "subject_skill": subject_skill_list,

                "labels": y,
            }
        else:
            fetures_summary = {
                "p1_al_onset_sim": p1_al_on_sim_list,
                "p1_al_prec_sim": p1_al_prec_sim_list,
                "p1_al_mag_sim": p1_al_gM_sim_list,

                "p2_al_onset_sim": p2_al_on_sim_list,
                "p2_al_prec_sim": p2_al_prec_sim_list,
                "p2_al_mag_sim": p2_al_gM_sim_list,

                "p3_fx_onset_sim": p3_fx_on_sim_list,
                "p3_fx_du_sim": p3_fx_du_sim_list,

                "p1_cs_sim": p1_cs_sim_list,
                "p2_cs_sim": p2_cs_sim_list,

                # mean
                "p1_al_onset_mean": p1_al_on_mean_list,
                "p1_al_prec_mean": p1_al_prec_mean_list,
                "p1_al_mag_mean": p1_al_gM_mean_list,
                "p1_cs_mean": p1_cs_mean_list,

                # p2
                "p2_al_onset_mean": p2_al_on_mean_list,
                "p2_al_prec_mean": p2_al_prec_mean_list,
                "p2_al_mag_mean": p2_al_gM_mean_list,
                "p2_cs_mean": p2_cs_mean_list,

                # p3
                "p3_fx_onset_mean": p3_fx_on_mean_list,
                "p3_fx_du_mean": p3_fx_du_mean_list,

                # action
                "ec_start_fs_sim": ec_start_fs_sim_list,
                "fixation_racket_latency_sim": fixation_racket_latency_sim_list,
                "distance_eye_hand_sim": distance_eye_hand_sim_list,
                "im_ball_updown_sim": im_ball_updown_sim_list,

                "ec_start_fs_mean": ec_start_fs_mean_list,
                "fixation_racket_latency_mean": fixation_racket_latency_mean_list,
                "distance_eye_hand_mean": distance_eye_hand_mean_list,
                "im_ball_updown_mean": im_ball_updown_mean_list,

                # impact
                "im_racket_ball_angle_sim": im_racket_ball_angle_sim_list,
                "im_racket_ball_wrist_sim": im_racket_ball_wrist_sim_list,
                "im_ball_wrist_sim": im_ball_wrist_sim_list,

                "im_racket_ball_angle_mean": im_racket_ball_angle_mean_list,
                "im_racket_ball_wrist_mean": im_racket_ball_wrist_mean_list,
                "im_ball_wrist_mean": im_ball_wrist_mean_list,

                "subject_skill": subject_skill_list,

                "labels": y,
            }

        return pd.DataFrame(fetures_summary)

    def getSnippetFeatures(self, n_index=10, group="lower"):
        if group == "lower":
            y = 0
        else:
            y = 1

        def computeMeanFeatures(df, features_name):
            time_series = df[features_name].values
            # from scipy.stats import linregress
            #
            # x = np.arange(len(time_series))
            # slope, intercept, r_value, p_value, std_err = linregress(x, time_series)
            return np.nanmean(time_series), np.nanstd(time_series)

        # p1
        receiver_pr_p1_al_onset_list = []
        receiver_pr_p1_al_prec_list = []
        receiver_pr_p1_al_mag_list = []
        receiver_pr_p1_cs_list = []

        hitter_pr_p1_al_onset_list = []
        hitter_pr_p1_al_prec_list = []
        hitter_pr_p1_al_mag_list = []
        hitter_pr_p1_cs_list = []

        # p2
        receiver_pr_p2_al_onset_list = []
        receiver_pr_p2_al_prec_list = []
        receiver_pr_p2_al_mag_list = []
        receiver_pr_p2_cs_list = []

        hitter_pr_p2_al_onset_list = []
        hitter_pr_p2_al_prec_list = []
        hitter_pr_p2_al_mag_list = []
        hitter_pr_p2_cs_list = []

        # p3
        receiver_pr_p3_fx_onset_list = []
        receiver_pr_p3_fx_duration_list = []

        hitter_pr_p3_fx_onset_list = []
        hitter_pr_p3_fx_duration_list = []

        # impact
        receiver_ec_start_fs_list = []
        receiver_im_racket_dir_list = []
        receiver_im_racket_effect_list = []
        receiver_im_ball_updown_list = []
        hand_movement_sim_dtw_list = []
        receiver_fixation_racket_latency_list = []
        receiver_im_racket_ball_angle_list = []
        receiver_im_racket_ball_wrist_list = []
        receiver_im_ball_wrist_list = []
        receiver_distance_eye_hand_list = []

        # std
        # p1
        receiver_pr_p1_al_onset_std_list = []
        receiver_pr_p1_al_prec_std_list = []
        receiver_pr_p1_al_mag_std_list = []
        receiver_pr_p1_cs_std_list = []

        hitter_pr_p1_al_onset_std_list = []
        hitter_pr_p1_al_prec_std_list = []
        hitter_pr_p1_al_mag_std_list = []
        hitter_pr_p1_cs_std_list = []

        # p2
        receiver_pr_p2_al_onset_std_list = []
        receiver_pr_p2_al_prec_std_list = []
        receiver_pr_p2_al_mag_std_list = []
        receiver_pr_p2_cs_std_list = []

        hitter_pr_p2_al_onset_std_list = []
        hitter_pr_p2_al_prec_std_list = []
        hitter_pr_p2_al_mag_std_list = []
        hitter_pr_p2_cs_std_list = []

        # p3
        receiver_pr_p3_fx_onset_std_list = []
        receiver_pr_p3_fx_duration_std_list = []

        hitter_pr_p3_fx_onset_std_list = []
        hitter_pr_p3_fx_duration_std_list = []

        # impact
        receiver_ec_start_fs_std_list = []
        receiver_im_racket_dir_std_list = []
        receiver_im_racket_effect_std_list = []
        receiver_im_ball_updown_std_list = []
        hand_movement_sim_dtw_std_list = []
        receiver_fixation_racket_latency_std_list = []
        receiver_im_racket_ball_angle_std_list = []
        receiver_im_racket_ball_wrist_std_list = []
        receiver_im_ball_wrist_std_list = []
        receiver_distance_eye_hand_std_list = []

        # plot le

        # skills
        subject_skill_list = []
        for _, g in self.df_summary.iterrows():
            s1 = g["Subject1"]
            s2 = g["Subject2"]

            # snipset of double summary
            # double_df = self.df[self.df["session_id"] == g["file_name"]][:n_index] #first
            double_df = self.df[self.df["session_id"] == g["file_name"]][-n_index:]  # last
            # n_data = len(self.df[self.df["session_id"] == g["file_name"]])
            # mid_data = n_data // 2
            # n_index_half = (n_index //2)
            # double_df = self.df[self.df["session_id"] == g["file_name"]][mid_data:mid_data+n_index] # mid
            # p1
            receiver_pr_p1_al_onset = computeMeanFeatures(double_df, "receiver_pr_p1_al_onset")
            receiver_pr_p1_al_prec = computeMeanFeatures(double_df, "receiver_pr_p1_al_prec")
            receiver_pr_p1_al_mag = computeMeanFeatures(double_df, "receiver_pr_p1_al_mag")
            receiver_pr_p1_cs = computeMeanFeatures(double_df, "receiver_pr_p1_cs")

            hitter_pr_p1_al_onset = computeMeanFeatures(double_df, "hitter_pr_p1_al_onset")
            hitter_pr_p1_al_prec = computeMeanFeatures(double_df, "hitter_pr_p1_al_prec")
            hitter_pr_p1_al_mag = computeMeanFeatures(double_df, "hitter_pr_p1_al_mag")
            hitter_pr_p1_cs = computeMeanFeatures(double_df, "hitter_pr_p1_cs")

            # p2
            receiver_pr_p2_al_onset = computeMeanFeatures(double_df, "receiver_pr_p2_al_onset")
            receiver_pr_p2_al_prec = computeMeanFeatures(double_df, "receiver_pr_p2_al_prec")
            receiver_pr_p2_al_mag = computeMeanFeatures(double_df, "receiver_pr_p2_al_mag")
            receiver_pr_p2_cs = computeMeanFeatures(double_df, "receiver_pr_p2_cs")

            hitter_pr_p2_al_onset = computeMeanFeatures(double_df, "hitter_pr_p2_al_onset")
            hitter_pr_p2_al_prec = computeMeanFeatures(double_df, "hitter_pr_p2_al_prec")
            hitter_pr_p2_al_mag = computeMeanFeatures(double_df, "hitter_pr_p2_al_mag")
            hitter_pr_p2_cs = computeMeanFeatures(double_df, "hitter_pr_p2_cs")

            # p3
            receiver_pr_p3_fx_onset = computeMeanFeatures(double_df, "receiver_pr_p3_fx_onset")
            receiver_pr_p3_fx_duration = computeMeanFeatures(double_df, "receiver_pr_p3_fx_duration")

            hitter_pr_p3_fx_onset = computeMeanFeatures(double_df, "hitter_pr_p3_fx_onset")
            hitter_pr_p3_fx_duration = computeMeanFeatures(double_df, "hitter_pr_p3_fx_duration")

            # impact
            receiver_ec_start_fs = computeMeanFeatures(double_df, "receiver_ec_start_fs")
            receiver_im_racket_dir = computeMeanFeatures(double_df, "receiver_im_racket_dir")
            receiver_im_racket_effect = computeMeanFeatures(double_df, "receiver_im_racket_effect")
            receiver_im_ball_updown = computeMeanFeatures(double_df, "receiver_im_ball_updown")

            receiver_im_racket_ball_angle = computeMeanFeatures(double_df, "receiver_im_racket_ball_angle")
            receiver_im_racket_ball_wrist = computeMeanFeatures(double_df, "receiver_im_racket_ball_wrist")
            receiver_im_ball_wrist = computeMeanFeatures(double_df, "receiver_im_ball_wrist")

            hand_movement_sim_dtw = computeMeanFeatures(double_df, "hand_movement_sim_dtw")
            receiver_fixation_racket_latency = computeMeanFeatures(double_df, "receiver_fixation_racket_latency")
            receiver_distance_eye_hand = computeMeanFeatures(double_df, "receiver_distance_eye_hand")

            # skills
            subject_1_skill = self.single_df[self.single_df["id_subject"] == s1]["skill_subject"].values[0]
            subject_2_skill = self.single_df[self.single_df["id_subject"] == s2]["skill_subject"].values[0]

            # mean
            # p1
            receiver_pr_p1_al_onset_list.append(receiver_pr_p1_al_onset[0])
            receiver_pr_p1_al_prec_list.append(receiver_pr_p1_al_prec[0])
            receiver_pr_p1_al_mag_list.append(receiver_pr_p1_al_mag[0])
            receiver_pr_p1_cs_list.append(receiver_pr_p1_cs[0])

            hitter_pr_p1_al_onset_list.append(hitter_pr_p1_al_onset[0])
            hitter_pr_p1_al_prec_list.append(hitter_pr_p1_al_prec[0])
            hitter_pr_p1_al_mag_list.append(hitter_pr_p1_al_mag[0])
            hitter_pr_p1_cs_list.append(hitter_pr_p1_cs[0])

            # p2
            receiver_pr_p2_al_onset_list.append(receiver_pr_p2_al_onset[0])
            receiver_pr_p2_al_prec_list.append(receiver_pr_p2_al_prec[0])
            receiver_pr_p2_al_mag_list.append(receiver_pr_p2_al_mag[0])
            receiver_pr_p2_cs_list.append(receiver_pr_p2_cs[0])

            hitter_pr_p2_al_onset_list.append(hitter_pr_p2_al_onset[0])
            hitter_pr_p2_al_prec_list.append(hitter_pr_p2_al_prec[0])
            hitter_pr_p2_al_mag_list.append(hitter_pr_p2_al_mag[0])
            hitter_pr_p2_cs_list.append(hitter_pr_p2_cs[0])

            # p3
            receiver_pr_p3_fx_onset_list.append(receiver_pr_p3_fx_onset[0])
            receiver_pr_p3_fx_duration_list.append(receiver_pr_p3_fx_duration[0])

            hitter_pr_p3_fx_onset_list.append(hitter_pr_p3_fx_onset[0])
            hitter_pr_p3_fx_duration_list.append(hitter_pr_p3_fx_duration[0])

            # impact
            receiver_ec_start_fs_list.append(receiver_ec_start_fs[0])
            receiver_im_racket_dir_list.append(receiver_im_racket_dir[0])
            receiver_im_racket_effect_list.append(receiver_im_racket_effect[0])
            receiver_im_ball_updown_list.append(receiver_im_ball_updown[0])
            hand_movement_sim_dtw_list.append(hand_movement_sim_dtw[0])
            receiver_fixation_racket_latency_list.append(receiver_fixation_racket_latency[0])
            receiver_im_racket_ball_angle_list.append(receiver_im_racket_ball_angle[0])
            receiver_im_racket_ball_wrist_list.append(receiver_im_racket_ball_wrist[0])
            receiver_im_ball_wrist_list.append(receiver_im_ball_wrist[0])
            receiver_distance_eye_hand_list.append(receiver_distance_eye_hand[0])

            # std
            # p1
            receiver_pr_p1_al_onset_std_list.append(receiver_pr_p1_al_onset[1])
            receiver_pr_p1_al_prec_std_list.append(receiver_pr_p1_al_prec[1])
            receiver_pr_p1_al_mag_std_list.append(receiver_pr_p1_al_mag[1])
            receiver_pr_p1_cs_std_list.append(receiver_pr_p1_cs[1])

            hitter_pr_p1_al_onset_std_list.append(hitter_pr_p1_al_onset[1])
            hitter_pr_p1_al_prec_std_list.append(hitter_pr_p1_al_prec[1])
            hitter_pr_p1_al_mag_std_list.append(hitter_pr_p1_al_mag[1])
            hitter_pr_p1_cs_std_list.append(hitter_pr_p1_cs[1])

            # p2
            receiver_pr_p2_al_onset_std_list.append(receiver_pr_p2_al_onset[1])
            receiver_pr_p2_al_prec_std_list.append(receiver_pr_p2_al_prec[1])
            receiver_pr_p2_al_mag_std_list.append(receiver_pr_p2_al_mag[1])
            receiver_pr_p2_cs_std_list.append(receiver_pr_p2_cs[1])

            hitter_pr_p2_al_onset_std_list.append(hitter_pr_p2_al_onset[1])
            hitter_pr_p2_al_prec_std_list.append(hitter_pr_p2_al_prec[1])
            hitter_pr_p2_al_mag_std_list.append(hitter_pr_p2_al_mag[1])
            hitter_pr_p2_cs_std_list.append(hitter_pr_p2_cs[1])

            # p3
            receiver_pr_p3_fx_onset_std_list.append(receiver_pr_p3_fx_onset[1])
            receiver_pr_p3_fx_duration_std_list.append(receiver_pr_p3_fx_duration[1])

            hitter_pr_p3_fx_onset_std_list.append(hitter_pr_p3_fx_onset[1])
            hitter_pr_p3_fx_duration_std_list.append(hitter_pr_p3_fx_duration[1])

            # impact
            receiver_ec_start_fs_std_list.append(receiver_ec_start_fs[1])
            receiver_im_racket_dir_std_list.append(receiver_im_racket_dir[1])
            receiver_im_racket_effect_std_list.append(receiver_im_racket_effect[1])
            receiver_im_ball_updown_std_list.append(receiver_im_ball_updown[1])
            hand_movement_sim_dtw_std_list.append(hand_movement_sim_dtw[1])
            receiver_fixation_racket_latency_std_list.append(receiver_fixation_racket_latency[1])
            receiver_im_racket_ball_angle_std_list.append(receiver_im_racket_ball_angle[1])
            receiver_im_racket_ball_wrist_std_list.append(receiver_im_racket_ball_wrist[1])
            receiver_im_ball_wrist_std_list.append(receiver_im_ball_wrist[1])
            receiver_distance_eye_hand_std_list.append(receiver_distance_eye_hand[1])

            # skill
            subject_skill_list.append(0.5 * (subject_1_skill + subject_2_skill))

        fetures_summary = {

            # snippet features (mean)

            "hitter_pr_p1_al_onset": hitter_pr_p1_al_onset_list,
            "hitter_pr_p1_al_prec": hitter_pr_p1_al_prec_list,
            "hitter_pr_p1_al_mag": hitter_pr_p1_al_mag_list,
            "hitter_pr_p1_cs": hitter_pr_p1_cs_list,
            "hitter_pr_p2_al_onset": hitter_pr_p2_al_onset_list,
            "hitter_pr_p2_al_prec": hitter_pr_p2_al_prec_list,
            "hitter_pr_p2_al_mag": hitter_pr_p2_al_mag_list,

            "receiver_pr_p1_al_onset": receiver_pr_p1_al_onset_list,
            "receiver_pr_p1_al_prec": receiver_pr_p1_al_prec_list,
            "receiver_pr_p1_al_mag": receiver_pr_p1_al_mag_list,
            "receiver_pr_p1_cs": receiver_pr_p1_cs_list,
            "receiver_pr_p2_al_onset": receiver_pr_p2_al_onset_list,
            "receiver_pr_p2_al_prec": receiver_pr_p2_al_prec_list,
            "receiver_pr_p2_al_mag": receiver_pr_p2_al_mag_list,

            "receiver_pr_p3_fx_onset": receiver_pr_p3_fx_onset_list,
            "receiver_pr_p3_fx_duration": receiver_pr_p3_fx_duration_list,
            "hitter_pr_p3_fx_onset": hitter_pr_p3_fx_onset_list,
            "hitter_pr_p3_fx_duration": hitter_pr_p3_fx_duration_list,

            "receiver_ec_start_fs": receiver_ec_start_fs_list,
            "receiver_im_racket_dir": receiver_im_racket_dir_list,
            "receiver_im_racket_effect": receiver_im_racket_effect_list,
            "receiver_im_ball_updown": receiver_im_ball_updown_list,
            "hand_movement_sim_dtw": hand_movement_sim_dtw_list,
            "receiver_fixation_racket_latency": receiver_fixation_racket_latency_list,
            "receiver_distance_eye_hand": receiver_distance_eye_hand_list,

            # snippet features (std)
            #
            # "hitter_pr_p1_al_onset_std": hitter_pr_p1_al_onset_std_list,
            # "hitter_pr_p1_al_prec_std": hitter_pr_p1_al_prec_std_list,
            # "hitter_pr_p1_al_mag_std": hitter_pr_p1_al_mag_std_list,
            # "hitter_pr_p1_cs_std": hitter_pr_p1_cs_std_list,
            # "hitter_pr_p2_al_onset_std": hitter_pr_p2_al_onset_std_list,
            # "hitter_pr_p2_al_prec_std": hitter_pr_p2_al_prec_std_list,
            # "hitter_pr_p2_al_mag_std": hitter_pr_p2_al_mag_std_list,
            #
            # "receiver_pr_p1_al_onset_std": receiver_pr_p1_al_onset_std_list,
            # "receiver_pr_p1_al_prec_std": receiver_pr_p1_al_prec_std_list,
            # "receiver_pr_p1_al_mag_std": receiver_pr_p1_al_mag_std_list,
            # "receiver_pr_p1_cs_std": receiver_pr_p1_cs_std_list,
            # "receiver_pr_p2_al_onset_std": receiver_pr_p2_al_onset_std_list,
            # "receiver_pr_p2_al_prec_std": receiver_pr_p2_al_prec_std_list,
            # "receiver_pr_p2_al_mag_std": receiver_pr_p2_al_mag_std_list,
            #
            # "receiver_pr_p3_fx_onset_std": receiver_pr_p3_fx_onset_std_list,
            # "receiver_pr_p3_fx_duration_std": receiver_pr_p3_fx_duration_std_list,
            # "hitter_pr_p3_fx_onset_std": hitter_pr_p3_fx_onset_std_list,
            # "hitter_pr_p3_fx_duration_std": hitter_pr_p3_fx_duration_std_list,
            #
            # "receiver_ec_start_fs_std": receiver_ec_start_fs_std_list,
            # "receiver_im_racket_dir_std": receiver_im_racket_dir_std_list,
            # "receiver_im_racket_effect_std": receiver_im_racket_effect_std_list,
            # "receiver_im_ball_updown_std": receiver_im_ball_updown_std_list,
            # "hand_movement_sim_dtw_std": hand_movement_sim_dtw_std_list,
            # "receiver_fixation_racket_latency_std": receiver_fixation_racket_latency_std_list,
            # "receiver_distance_eye_hand_std": receiver_distance_eye_hand_std_list,

            "subject_skill": subject_skill_list,

            "labels": y,
        }

        return pd.DataFrame(fetures_summary)

    def getSnippetFeaturesPCA(self, n_index=10, group="lower"):
        if group == "lower":
            y = 0
        else:
            y = 1

        included_feautures = [
            "receiver_pr_p1_al_onset",
            "receiver_pr_p1_al_prec",
            "receiver_pr_p1_al_mag",
            "receiver_pr_p1_cs",
            "hitter_pr_p1_al_onset",
            "hitter_pr_p1_al_prec",
            "hitter_pr_p1_al_mag",
            "hitter_pr_p1_cs",
            "receiver_pr_p2_al_onset",
            "receiver_pr_p2_al_prec",
            "receiver_pr_p2_al_mag",
            "receiver_pr_p2_cs",
            "hitter_pr_p2_al_onset",
            "hitter_pr_p2_al_prec",
            "hitter_pr_p2_al_mag",
            "hitter_pr_p2_cs",
            "receiver_pr_p3_fx_onset",
            "receiver_pr_p3_fx_duration",
            "hitter_pr_p3_fx_onset",
            "hitter_pr_p3_fx_duration",
            "receiver_ec_start_fs",
            "receiver_im_racket_dir",
            "receiver_im_racket_effect",
            "receiver_im_ball_updown",
            "receiver_im_racket_ball_angle",
            "receiver_im_racket_ball_wrist",
            "receiver_im_ball_wrist",
            "hand_movement_sim_dtw",
            "receiver_fixation_racket_latency",
            "receiver_distance_eye_hand"
        ]

        features_list = []
        # skills
        subject_skill_list = []
        for _, g in self.df_summary.iterrows():
            s1 = g["Subject1"]
            s2 = g["Subject2"]

            df = self.df[self.df["session_id"] == g["file_name"]]
            # snipset of double summary
            # double_df =df[:n_index] #first
            double_df = df[-n_index:]  # last
            # middle
            # n_data = len(df)
            # mid_data = n_data // 2
            # n_index_half = (n_index //2)
            # double_df = df[mid_data:mid_data+n_index] # mid

            if len(double_df) < n_index:
                print(len(double_df))
            features = double_df.loc[:, included_feautures].values
            features_list.append(features.flatten())
            # skills
            subject_1_skill = self.single_df[self.single_df["id_subject"] == s1]["skill_subject"].values[0]
            subject_2_skill = self.single_df[self.single_df["id_subject"] == s2]["skill_subject"].values[0]

            subject_skill_list.append(0.5 * (subject_1_skill + subject_2_skill))

        fetures_summary = {
            "features": features_list,
            "subject_skill": subject_skill_list,

            "labels": y,
        }

        return pd.DataFrame(fetures_summary)


if __name__ == '__main__':
    from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH

    # control group
    reader = ImpressionFeatures(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                include_subjects=None, exclude_failure=True,
                                exclude_no_pair=False)

    df = reader.getImpressionFeatures()

    lower_group_data = reader.getImpressionFeatures()
