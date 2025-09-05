import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Utils.Conf import SINGLE_FEATURES_FILE_PATH, NORMALIZE_X_DOUBLE_EPISODE_COLUMNS, HMM_MODEL_PATH, \
    SINGLE_SUMMARY_FILE_PATH, DOUBLE_SUMMARY_FILE_PATH, ECG_FEATURES_FILE_PATH, DOUBLE_ME_FEATURES_FILE_PATH, \
    SINGLE_ME_FEATURES_FILE_PATH
from sklearn.impute import KNNImputer
from scipy.ndimage import label
import hrvanalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance


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
        self.single_summary_df = pd.read_csv(SINGLE_SUMMARY_FILE_PATH)
        self.double_summary_df = pd.read_csv(DOUBLE_SUMMARY_FILE_PATH)
        self.df_summary = pd.read_csv(file_summary_path)
        self.ecg_df = pd.read_pickle(ECG_FEATURES_FILE_PATH)
        self.df = pd.read_pickle(file_path)
        self.single_df = pd.read_pickle(SINGLE_FEATURES_FILE_PATH)
        self.me_df = pd.read_pickle(DOUBLE_ME_FEATURES_FILE_PATH)

        # if hmm_probs:
        #     self.df = self.timeSeriesFeatures()

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

            "receiver_im_racket_dir": np.concatenate(receiver_im_racket_dir),
            "receiver_im_ball_updown": np.concatenate(receiver_im_ball_updown),

            "hitter_at_and_after_hit": np.concatenate(hitter_at_and_after_hit),

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

    def getCoupledFeatures(self, group_name="test", success_failure=False, mod="skill",
                           with_control=False, timepoint=False):

        group_df = self.df.groupby(['session_id', 'episode_label'])

        onset_forward_swing_prev_list = []
        onset_forward_swing_next_list = []

        al_prec_p1_prev_list = []
        al_prec_p1_next_list = []

        al_prec_HP_p1_prev_list = []
        al_prec_HP_p1_next_list = []

        ball_updown_prev_list = []
        ball_updown_next_list = []

        receiver_list = []
        receiver_idx_list = []
        hitter_idx_list = []
        hitter_list = []
        receiver_skill_list = []
        hitter_skill_list = []
        for name, group in group_df:
            # n_data = len(group["hitter_pr_p2_al"]) - (n_segment - 1)
            # print(n_data)

            group_name = name[0]
            skill_subjects = group[["skill_subject1", "skill_subject2"]].values[0]
            subjects = group[["id_subject1", "id_subject2"]].values[0]
            if len(group) > 5:
                group.sort_values(by=['observation_label'])

                prev_fs = group["receiver_ec_start_fs"].values[:-1]
                next_fs = group["receiver_ec_start_fs"].values[1:]

                prev_al_prec_p1 = group["receiver_pr_p1_al_prec"].values[:-1]
                next_al_prec_p1 = group["receiver_pr_p1_al_prec"].values[1:]

                prev_al_HP_prec_p1 = group["hitter_pr_p1_al_prec"].values[1:]
                next_al_HP_prec_p1 = group["receiver_pr_p1_al_prec"].values[1:]

                prev_ball_updown = group["receiver_im_ball_updown"].values[:-1]
                next_ball_updown = group["receiver_im_ball_updown"].values[1:]

                receiver = subjects[group["receiver"].values.astype(int)[1:]]
                hitter = subjects[group["hitter"].values.astype(int)[1:]]

                receiver_skill = skill_subjects[group["receiver"].values.astype(int)[1:]]
                hitter_skill = skill_subjects[group["hitter"].values.astype(int)[1:]]

                receiver_idx = group["receiver"].values.astype(int)[1:]
                hitter_idx = group["hitter"].values.astype(int)[1:]

                # onset of forward swing
                onset_forward_swing_prev_list.append(prev_fs)
                onset_forward_swing_next_list.append(next_fs)

                # precision of al in p1
                al_prec_p1_prev_list.append(prev_al_prec_p1)
                al_prec_p1_next_list.append(next_al_prec_p1)

                # precision of al HP in p1
                al_prec_HP_p1_prev_list.append(prev_al_HP_prec_p1)
                al_prec_HP_p1_next_list.append(next_al_HP_prec_p1)

                # ball up down
                ball_updown_prev_list.append(prev_ball_updown)
                ball_updown_next_list.append(next_ball_updown)

                receiver_idx_list.append(receiver_idx)
                hitter_idx_list.append(hitter_idx)
                receiver_list.append(receiver)
                hitter_list.append(hitter)
                receiver_skill_list.append(receiver_skill)
                hitter_skill_list.append(hitter_skill)

        fetures_summary = {
            # features
            "onset_forward_swing_prev": np.concatenate(onset_forward_swing_prev_list),
            "onset_forward_swing_next": np.concatenate(onset_forward_swing_next_list),

            "al_prec_p1_prev": np.concatenate(al_prec_p1_prev_list),
            "al_prec_p1_next": np.concatenate(al_prec_p1_next_list),

            "al_prec_HP_p1_prev": np.concatenate(al_prec_HP_p1_prev_list),
            "al_prec_HP_p1_next": np.concatenate(al_prec_HP_p1_next_list),

            "ball_updown_prev": np.concatenate(ball_updown_prev_list),
            "ball_updown_next": np.concatenate(ball_updown_next_list),

            # control
            "receiver_idx": np.concatenate(receiver_idx_list),
            "hitter_idx": np.concatenate(hitter_idx_list),
            "receiver": np.concatenate(receiver_list),
            "hitter": np.concatenate(hitter_list),
            "receiver_skill": np.concatenate(receiver_skill_list),
            "hitter_skill": np.concatenate(hitter_skill_list),
        }
        df = pd.DataFrame(fetures_summary)
        return df

    def getSelfReportFeatures(self):

        def convertFacilitating(answer):
            if answer == "Much less":
                return 1
            elif answer == "Less":
                return 2
            elif answer == "Slightly less":
                return 3
            elif answer == "Equally":
                return 4
            elif answer == "Slightly more":
                return 5
            elif answer == "More":
                return 6
            elif answer == "Much more":
                return 7
            else:
                return -1

        def convertGender(g):
            if g == "Man":
                return 0
            else:
                return 1

        subjects = np.unique(np.concatenate(self.df[["id_subject1", "id_subject2"]].values))
        double_subject_id = []
        double_subject_gender = []
        double_score = []
        double_self_report_score = []
        double_team_score = []
        double_facilitating_skill = []
        double_partner_skill = []
        double_order_play = []
        double_file_name = []
        double_skill_comp = []

        subject_indv_id = []
        subject_indv_skill = []
        subject_indv_myskill = []
        subject_indv_gender = []
        subject_indv_age = []
        subject_indv_height = []
        subject_indv_weight = []
        subject_indv_education = []

        def skillOrder(s1, s2):
            subject1_skill = self.single_summary_df[self.single_summary_df["Subject1"] == s1]["skill"].values
            subject2_skill = self.single_summary_df[self.single_summary_df["Subject1"] == s2]["skill"].values

            if subject1_skill < subject2_skill:
                return 0
            else:
                return 1

        def extractDoubleInfo(subject_id):
            subject_double_list = self.double_summary_df[
                (self.double_summary_df["Subject1"] == subject_id) | (self.double_summary_df["Subject2"] == subject_id)]

            subject_Iteam_score = self.single_summary_df[self.single_summary_df["Subject1"] == subject_id][
                ["IT1-skill", "IT2-skill"]].values
            subject_team_score = self.single_summary_df[self.single_summary_df["Subject1"] == subject_id][
                ["Team1-skill", "Team2-skill"]].values
            subject_facilitating_score = self.single_summary_df[self.single_summary_df["Subject1"] == subject_id][
                ["Facilitator1", "Facilitator1"]].values
            subject_partner_score = self.single_summary_df[self.single_summary_df["Subject1"] == subject_id][
                ["Partner1-skill", "Partner1-skill"]].values
            subjects = np.concatenate([subject_double_list["Subject1"].values, subject_double_list["Subject2"].values])
            subject_partners = subjects[subjects != subject_id]
            return subject_double_list, subject_Iteam_score, subject_team_score, subject_facilitating_score, subject_partner_score, subject_partners

        for s in subjects:
            # add subject 1
            subject_double_list, subject_Iteam_score, subject_team_score, subject_facilitating_score, subject_partner_score, subject_partners = extractDoubleInfo(
                s)
            subject = self.single_summary_df[self.single_summary_df["Subject1"] == s]
            # print(len(subject_double_list))
            # if len(subject_double_list) == 1:
            #     print("1")
            # print("----------------------")
            # print(subject_double_list["file_name"].values)
            # print("---------------------------")
            double_subject_id.extend([s] * len(subject_double_list))
            double_order_play.extend(np.arange(len(subject_double_list)))
            double_subject_gender.extend([subject["Gender"].values[0]] * len(subject_double_list))
            double_file_name.extend(subject_double_list["file_name"].values)
            double_score.extend(subject_double_list["skill"].values)
            double_self_report_score.extend(subject_Iteam_score[0][:len(subject_double_list)])
            double_team_score.extend(subject_team_score[0][:len(subject_double_list)])
            double_facilitating_skill.extend(subject_facilitating_score[0][:len(subject_double_list)])
            double_partner_skill.extend(subject_partner_score[0][:len(subject_double_list)])
            double_skill_comp.extend([skillOrder(s, p) for p in subject_partners])

            # add individual

            subject_indv_id.append(s)
            subject_indv_skill.append(subject["skill"].values[0])
            subject_indv_myskill.append(
                self.single_summary_df[self.single_summary_df["Subject1"] == s]["I-skill"].values[0])
            subject_indv_gender.append(subject["Gender"].values[0])
            subject_indv_age.append(subject["Age"].values[0])
            subject_indv_height.append(subject["Height"].values[0])
            subject_indv_weight.append(subject["Weight"].values[0])
            subject_indv_education.append(subject["Education"].values[0])

        # print(subjects)

        joint_action_df = pd.DataFrame({
            "double_subject_id": np.asarray(double_subject_id),
            "double_subject_gender": np.asarray(double_subject_gender),
            "double_score": np.asarray(double_score),
            "double_self_report_score": np.asarray(double_self_report_score),
            "double_team_score": np.asarray(double_team_score),
            "double_facilitating_skill": np.asarray(double_facilitating_skill),
            "double_partner_skill": np.asarray(double_partner_skill),
            "double_file_name": np.asarray(double_file_name),
            "double_order_play": np.asarray(double_order_play),
            "double_skill_comp": np.asarray(double_skill_comp)
        })

        joint_action_df['double_facilitating_skill'] = joint_action_df['double_facilitating_skill'].apply(
            convertFacilitating)
        joint_action_df['double_subject_gender'] = joint_action_df['double_subject_gender'].apply(
            convertGender)
        individual_action_df = pd.DataFrame({
            "subject_indv_id": subject_indv_id,
            "subject_indv_skill": subject_indv_skill,
            "subject_indv_myskill": subject_indv_myskill,
            "subject_indv_gender": subject_indv_gender,
            "subject_indv_age": subject_indv_age,
            "subject_indv_height": subject_indv_height,
            "subject_indv_weight": subject_indv_weight,
            "subject_indv_education": subject_indv_education,
        })
        individual_action_df['subject_indv_gender'] = individual_action_df['subject_indv_gender'].apply(
            convertGender)

        return joint_action_df, individual_action_df

    def getSingleDoubleFeatures(self):
        double_ecg = self.ecg_df[self.ecg_df["double_single"] == "D"]
        single_ecg = self.ecg_df[self.ecg_df["double_single"] == "S"]

        def similarity(x1, x2, bins=7):
            x1 = x1[np.isnan(x1) != True]
            x2 = x2[np.isnan(x2) != True]

            return np.mean(x2) - np.mean(x1)

        def ecgSimilarity(x1, x2, bins=7):
            # frequency domain
            x1_freq = hrvanalysis.get_frequency_domain_features(x1, method="lomb")
            x2_freq = hrvanalysis.get_frequency_domain_features(x2, method="lomb")

            # spatial
            x1_spatial = hrvanalysis.get_time_domain_features(x1)
            x2_spatial = hrvanalysis.get_time_domain_features(x2)

            hf_sim = x1_freq["hf"] - x2_freq["hf"]
            lfhf_sim = x2_freq["lf_hf_ratio"] - x1_freq["lf_hf_ratio"]
            rmsdd_sim = x1_spatial["rmssd"] - x2_spatial["rmssd"]

            return hf_sim, lfhf_sim, rmsdd_sim

        group_df = self.df.groupby(['session_id'])

        # perception hitter
        std_hitter_p1_al_prec_list = []
        std_hitter_p1_al_onset_list = []
        std_hitter_p1_al_mag_list = []
        std_hitter_p1_cs_list = []

        std_hitter_p2_al_prec_list = []
        std_hitter_p2_al_onset_list = []
        std_hitter_p2_al_mag_list = []
        std_hitter_p2_cs_list = []

        std_hitter_p3_fx_onset_list = []
        std_hitter_p3_fx_du_list = []

        # perception receiver
        std_receiver_p1_al_prec_list = []
        std_receiver_p1_al_onset_list = []
        std_receiver_p1_al_mag_list = []
        std_receiver_p1_cs_list = []

        std_receiver_p2_al_prec_list = []
        std_receiver_p2_al_onset_list = []
        std_receiver_p2_al_mag_list = []
        std_receiver_p2_cs_list = []

        std_receiver_p3_fx_onset_list = []
        std_receiver_p3_fx_du_list = []

        # action
        std_start_fs_list = []
        std_fixation_racket_latency_list = []
        std_distance_eye_hand_list = []

        # impact
        std_im_ball_wrist_list = []
        std_im_racket_ball_wrist_list = []
        std_im_racket_ball_angle_list = []
        std_im_ball_updown_list = []

        # ecg
        hf_sim_list = []
        lfhf_sim_list = []
        rmsdd_sim_list = []

        # others
        std_distance_list = []
        std_spatial_use_list = []

        id_subject_list = []
        id_partner_list = []
        age_subject_list = []
        gender_subject_list = []
        for name, group in group_df:
            skill_subjects = group[["skill_subject1", "skill_subject2"]].values[0]
            subjects = group[["id_subject1", "id_subject2"]].values[0]

            i = 0
            for s, p in zip(subjects, np.flip(subjects)):
                # info
                age_s = self.single_summary_df[self.single_summary_df["Subject1"] == s][
                    "Age"].values

                gender_s = self.single_summary_df[self.single_summary_df["Subject1"] == s][
                    "Gender"].values

                # single features

                # perception
                single_p1_al_prec = self.single_df[self.single_df["id_subject"] == s]["pr_p1_al_prec"].values
                single_p1_al_onset = self.single_df[self.single_df["id_subject"] == s]["pr_p1_al_on"].values
                single_p1_al_mag = self.single_df[self.single_df["id_subject"] == s]["pr_p1_al_gM"].values
                single_p1_cs = self.single_df[self.single_df["id_subject"] == s]["pr_p1_sf"].values

                single_p2_al_prec = self.single_df[self.single_df["id_subject"] == s]["pr_p2_al_prec"].values
                single_p2_al_onset = self.single_df[self.single_df["id_subject"] == s]["pr_p2_al_on"].values
                single_p2_al_mag = self.single_df[self.single_df["id_subject"] == s]["pr_p2_al_gM"].values
                single_p2_cs = self.single_df[self.single_df["id_subject"] == s]["pr_p2_sf"].values

                single_p3_fx_on = self.single_df[self.single_df["id_subject"] == s]["pr_p3_fx_on"].values
                single_p3_fx_du = self.single_df[self.single_df["id_subject"] == s]["pr_p3_fx_du"].values

                # action
                single_start_fs = self.single_df[self.single_df["id_subject"] == s]["ec_start_fs"].values
                single_fixation_racket_latency = self.single_df[self.single_df["id_subject"] == s][
                    "fixation_racket_latency"].values
                single_distance_eye_hand = self.single_df[self.single_df["id_subject"] == s]["distance_eye_hand"].values

                # impact
                single_im_racket_ball_wrist = self.single_df[self.single_df["id_subject"] == s][
                    "im_racket_ball_wrist"].values
                single_im_ball_wrist = self.single_df[self.single_df["id_subject"] == s][
                    "im_ball_wrist"].values
                single_im_racket_ball_angle = self.single_df[self.single_df["id_subject"] == s][
                    "im_racket_ball_angle"].values
                single_im_ball_updown = self.single_df[self.single_df["id_subject"] == s]["im_ball_updown"].values

                # ecg
                single_rr = single_ecg[single_ecg["subject1"] == s]["rr1"].values

                # other
                single_bounce_p1 = self.single_df[self.single_df["id_subject"] == s][
                    ["bouncing_point_p1_x", "bouncing_point_p1_z"]].values
                single_at_after_hit = self.single_df[self.single_df["id_subject"] == subjects[0]][
                    "dis_at_after_hit"].values

                # double features
                # perception
                # hitter
                hitter_p1_al_prec = group["hitter_pr_p1_al_prec"].values[group["hitter"] == i]
                hitter_p1_al_onset = group["hitter_pr_p1_al_onset"].values[group["hitter"] == i]
                hitter_p1_al_mag = group["hitter_pr_p1_al_mag"].values[group["hitter"] == i]
                hitter_p1_cs = group["hitter_pr_p1_cs"].values[group["hitter"] == i]

                hitter_p2_al_prec = group["hitter_pr_p2_al_prec"].values[group["hitter"] == i]
                hitter_p2_al_onset = group["hitter_pr_p2_al_onset"].values[group["hitter"] == i]
                hitter_p2_al_mag = group["hitter_pr_p2_al_mag"].values[group["hitter"] == i]
                hitter_p2_cs = group["hitter_pr_p2_cs"].values[group["hitter"] == i]

                hitter_p3_fx_onset = group["hitter_pr_p3_fx_onset"].values[group["hitter"] == i]
                hitter_p3_fx_duration = group["hitter_pr_p3_fx_duration"].values[group["hitter"] == i]

                # receiver
                receiver_p1_al_prec = group["receiver_pr_p1_al_prec"].values[group["receiver"] == i]
                receiver_p1_al_onset = group["receiver_pr_p1_al_onset"].values[group["receiver"] == i]
                receiver_p1_al_mag = group["receiver_pr_p1_al_mag"].values[group["receiver"] == i]
                receiver_p1_cs = group["receiver_pr_p1_cs"].values[group["receiver"] == i]

                receiver_p2_al_prec = group["receiver_pr_p2_al_prec"].values[group["receiver"] == i]
                receiver_p2_al_onset = group["receiver_pr_p2_al_onset"].values[group["receiver"] == i]
                receiver_p2_al_mag = group["receiver_pr_p2_al_mag"].values[group["receiver"] == i]
                receiver_p2_cs = group["receiver_pr_p2_cs"].values[group["receiver"] == i]

                receiver_p3_fx_onset = group["receiver_pr_p3_fx_onset"].values[group["hitter"] == i]
                receiver_p3_fx_duration = group["receiver_pr_p3_fx_duration"].values[group["hitter"] == i]

                # action
                start_fs = group["receiver_ec_start_fs"].values[group["receiver"] == i]
                fixation_racket_latency = group["receiver_fixation_racket_latency"].values[group["receiver"] == i]
                distance_eye_hand = group["receiver_distance_eye_hand"].values[group["receiver"] == i]

                # impact
                im_ball_wrist = group["receiver_im_ball_wrist"].values[group["receiver"] == i]
                im_racket_ball_wrist = group["receiver_im_racket_ball_wrist"].values[group["receiver"] == i]
                im_racket_ball_angle = group["receiver_im_racket_ball_angle"].values[group["receiver"] == i]
                im_ball_updown = group["receiver_im_ball_updown"].values[group["receiver"] == i]

                # ecg
                trial_date, trial_session, trial_name = group["session_id"].values[0].split("_")

                trial_rr = double_ecg[
                    (double_ecg["date"] == trial_date) & (double_ecg["session"] == trial_session) & (
                            double_ecg["trial_name"] == trial_name)]
                trial_subjects = trial_rr[["subject1", "subject2"]].values
                rr_column = "rr1" if (np.argwhere(trial_subjects.flatten() == s)[0] == 0) else "rr2"
                double_trial_rr = trial_rr[rr_column].values

                # others
                bounce_p1_s = group[["bouncing_point_p1_x", "bouncing_point_p1_z"]].values[group["hitter"] == i]
                at_after_hit_fs = group["hitter_at_and_after_hit"].values[group["receiver"] == i]

                # compute features
                # perception

                # hitter
                std_hitter_p1_al_prec = similarity(single_p1_al_prec, hitter_p1_al_prec)
                std_hitter_p1_al_onset = similarity(single_p1_al_onset, hitter_p1_al_onset)
                std_hitter_p1_al_mag = similarity(single_p1_al_mag, hitter_p1_al_mag)
                std_hitter_p1_cs = similarity(single_p1_cs, hitter_p1_cs)

                std_hitter_p2_al_prec = similarity(single_p2_al_prec, hitter_p2_al_prec)
                std_hitter_p2_al_onset = similarity(single_p2_al_onset, hitter_p2_al_onset)
                std_hitter_p2_al_mag = similarity(single_p2_al_mag, hitter_p2_al_mag)
                std_hitter_p2_cs = similarity(single_p2_cs, hitter_p2_cs)

                std_hitter_p3_fx_onset = similarity(single_p3_fx_on, hitter_p3_fx_onset)
                std_hitter_p3_fx_duration = similarity(single_p3_fx_du, hitter_p3_fx_duration)

                # receiver
                std_receiver_p1_al_prec = similarity(single_p1_al_prec, receiver_p1_al_prec)
                std_receiver_p1_al_onset = similarity(single_p1_al_onset, receiver_p1_al_onset)
                std_receiver_p1_al_mag = similarity(single_p1_al_mag, receiver_p1_al_mag)
                std_receiver_p1_cs = similarity(single_p1_cs, receiver_p1_cs)

                std_receiver_p2_al_prec = similarity(single_p2_al_prec, receiver_p2_al_prec)
                std_receiver_p2_al_onset = similarity(single_p2_al_onset, receiver_p2_al_onset)
                std_receiver_p2_al_mag = similarity(single_p2_al_mag, receiver_p2_al_mag)
                std_receiver_p2_cs = similarity(single_p2_cs, receiver_p2_cs)

                std_receiver_p3_fx_onset = similarity(single_p3_fx_on, receiver_p3_fx_onset)
                std_receiver_p3_fx_duration = similarity(single_p3_fx_du, receiver_p3_fx_duration)

                # action
                std_start_fs = similarity(single_start_fs, start_fs)
                std_fixation_racket_latency = similarity(single_fixation_racket_latency, fixation_racket_latency)
                std_distance_eye_hand = similarity(single_distance_eye_hand, distance_eye_hand)

                # impact
                std_im_ball_wrist = similarity(single_im_ball_wrist, im_ball_wrist)
                std_im_racket_ball_wrist = similarity(single_im_racket_ball_wrist, im_racket_ball_wrist)
                std_im_racket_ball_angle = similarity(single_im_racket_ball_angle, im_racket_ball_angle)
                std_im_ball_updown = similarity(single_im_ball_updown, im_ball_updown)

                # ecg
                hf_sim, lfhf_sim, rmsdd_sim = ecgSimilarity(single_rr[0], double_trial_rr[0])

                # others
                single_distance_s = np.linalg.norm(single_bounce_p1 - np.mean(single_bounce_p1, axis=0), axis=-1)
                distance_s = np.linalg.norm(bounce_p1_s - np.mean(bounce_p1_s, axis=0), axis=-1)
                std_distance_s = similarity(single_distance_s, distance_s)
                std_spatial_use = similarity(single_at_after_hit, at_after_hit_fs)

                # add to list
                id_subject_list.append(s)
                id_partner_list.append(p)

                age_subject_list.append(age_s[0])
                gender_subject_list.append(gender_s[0])
                # hitter perception
                std_hitter_p1_al_prec_list.append(std_hitter_p1_al_prec)
                std_hitter_p1_al_onset_list.append(std_hitter_p1_al_onset)
                std_hitter_p1_al_mag_list.append(std_hitter_p1_al_mag)
                std_hitter_p1_cs_list.append(std_hitter_p1_cs)

                std_hitter_p2_al_prec_list.append(std_hitter_p2_al_prec)
                std_hitter_p2_al_onset_list.append(std_hitter_p2_al_onset)
                std_hitter_p2_al_mag_list.append(std_hitter_p2_al_mag)
                std_hitter_p2_cs_list.append(std_hitter_p2_cs)

                std_hitter_p3_fx_onset_list.append(std_hitter_p3_fx_onset)
                std_hitter_p3_fx_du_list.append(std_hitter_p3_fx_duration)

                # receiver perception
                std_receiver_p1_al_prec_list.append(std_receiver_p1_al_prec)
                std_receiver_p1_al_onset_list.append(std_receiver_p1_al_onset)
                std_receiver_p1_al_mag_list.append(std_receiver_p1_al_mag)
                std_receiver_p1_cs_list.append(std_receiver_p1_cs)

                std_receiver_p2_al_prec_list.append(std_receiver_p2_al_prec)
                std_receiver_p2_al_onset_list.append(std_receiver_p2_al_onset)
                std_receiver_p2_al_mag_list.append(std_receiver_p2_al_mag)
                std_receiver_p2_cs_list.append(std_receiver_p2_cs)

                std_receiver_p3_fx_onset_list.append(std_receiver_p3_fx_onset)
                std_receiver_p3_fx_du_list.append(std_receiver_p3_fx_duration)

                # action
                std_start_fs_list.append(std_start_fs)
                std_fixation_racket_latency_list.append(std_fixation_racket_latency)
                std_distance_eye_hand_list.append(std_distance_eye_hand)

                # impact
                std_im_ball_wrist_list.append(std_im_ball_wrist)
                std_im_racket_ball_wrist_list.append(std_im_racket_ball_wrist)
                std_im_racket_ball_angle_list.append(std_im_racket_ball_angle)
                std_im_ball_updown_list.append(std_im_ball_updown)

                # ecg
                hf_sim_list.append(hf_sim)
                lfhf_sim_list.append(lfhf_sim)
                rmsdd_sim_list.append(rmsdd_sim)

                # others
                std_distance_list.append(std_distance_s)
                std_spatial_use_list.append(std_spatial_use)

                # add subject
                i += 1

        fetures_summary = {
            "id_subject": np.asarray(id_subject_list),
            "id_partner": np.asarray(id_partner_list),

            "age_subject": np.asarray(age_subject_list),
            "gender_subject": np.asarray(gender_subject_list) == "Man",

            # hitter perception
            "v_hitter_p1_al_onset": np.asarray(std_hitter_p1_al_onset_list),
            "v_hitter_p1_al_prec": np.asarray(std_hitter_p1_al_prec_list),
            "v_hitter_p1_al_mag": np.asarray(std_hitter_p1_al_mag_list),
            "v_hitter_p1_cs": np.asarray(std_hitter_p1_cs_list),

            "v_hitter_p2_al_onset": np.asarray(std_hitter_p2_al_prec_list),
            "v_hitter_p2_al_prec": np.asarray(std_hitter_p2_al_onset_list),
            "v_hitter_p2_al_mag": np.asarray(std_hitter_p2_al_mag_list),
            "v_hitter_p2_cs": np.asarray(std_hitter_p2_cs_list),

            "v_hitter_p3_fx_onset": np.asarray(std_hitter_p3_fx_onset_list),
            "v_hitter_p3_fx_duration": np.asarray(std_hitter_p3_fx_du_list),

            # receiver perception

            "v_receiver_p1_al_onset": np.asarray(std_receiver_p1_al_onset_list),
            "v_receiver_p1_al_prec": np.asarray(std_receiver_p1_al_prec_list),
            "v_receiver_p1_al_mag": np.asarray(std_receiver_p1_al_mag_list),
            "v_receiver_p1_cs": np.asarray(std_receiver_p1_cs_list),

            "v_receiver_p2_al_onset": np.asarray(std_receiver_p2_al_prec_list),
            "v_receiver_p2_al_prec": np.asarray(std_receiver_p2_al_onset_list),
            "v_receiver_p2_al_mag": np.asarray(std_receiver_p2_al_mag_list),
            "v_receiver_p2_cs": np.asarray(std_receiver_p2_cs_list),

            "v_receiver_p3_fx_onset": np.asarray(std_receiver_p3_fx_onset_list),
            "v_receiver_p3_fx_duration": np.asarray(std_receiver_p3_fx_du_list),

            # action
            "v_start_fs": np.asarray(std_start_fs_list),
            "v_fixation_racket_latency": np.asarray(std_fixation_racket_latency_list),
            "v_distance_eye_han": np.asarray(std_distance_eye_hand_list),

            # impact
            "v_im_ball_wrist": np.asarray(std_im_ball_wrist_list),
            "v_im_racket_ball_wrist": np.asarray(std_im_racket_ball_wrist_list),
            "v_im_racket_ball_angle": np.asarray(std_im_racket_ball_angle_list),
            "v_im_ball_updown": np.asarray(std_im_ball_updown_list),

            # rr
            "hf_sim": np.asarray(hf_sim_list),
            "lfhf_sim": np.asarray(lfhf_sim_list),
            "rmsdd_sim": np.asarray(rmsdd_sim_list),

            # others
            "v_bounce_point": np.asarray(std_distance_list),
            "v_spatial_use": np.asarray(std_spatial_use_list),

        }

        return pd.DataFrame(fetures_summary)

    def getFEFeatures(self, min_group_n=3):

        def convertSegmentationLabel(v):
            v[v<=10] = 0
            v[(v>10) & (v<=40)] = 1
            v[v > 40] = 2
            return v
        scaler = StandardScaler()
        df = self.df.iloc[self.df["success"].values == 1]
        group_df = df.groupby(['session_id', 'episode_label'])

        # prior
        priors_mean = []
        priors_std = []
        # sensory evidence (trial (t-1))
        sense_p1_visual_angle_error = []
        sense_p2_visual_angle_error = []
        sense_p3_pursuit_duration = []
        sense_swing_onset = []
        sense_distance_eye_hand = []
        sense_racket_ball_angle = []
        sense_racket_ball_wrist = []
        sense_im_ball_updown = []

        # posterior (trial t)
        post_al_p1_visual_angle_error = []

        observation_list = []
        receiver_list = []
        hitter_list = []
        session_list = []
        self.single_df["pr_p1_al_prec"] = scaler.fit_transform(
            self.single_df["pr_p1_al_prec"].values.reshape((-1, 1))).flatten()
        for name, group in group_df:
            if len(group) > min_group_n:
                group.sort_values(by=['observation_label'])

                prev_data = group.iloc[0:-1]
                curr_data = group.iloc[1:]
                if len(prev_data) !=  len(curr_data):
                    print("error")


                receiver_idx = np.concatenate([np.argwhere(prev_data["success"].values == 1).flatten()])
                hitter_idx = np.concatenate([np.argwhere(curr_data["success"].values == 1).flatten()])

                subjects = group[["id_subject1", "id_subject2"]].values[0]

                receiver = subjects[prev_data["receiver"].values[receiver_idx].astype(int)]
                hitter = subjects[curr_data["hitter"].values[hitter_idx].astype(int)]
                session = curr_data["session_id"].values[hitter_idx]




                # prior
                prior_mean1 = np.mean(self.single_df[self.single_df["id_subject"].values == subjects[0]]["pr_p1_al_prec"].values)
                prior_mean2 = np.mean(self.single_df[self.single_df["id_subject"].values == subjects[1]]["pr_p1_al_prec"].values)


                prior_std1 = np.std(self.single_df[self.single_df["id_subject"].values == subjects[0]]["pr_p1_al_prec"].values)
                prior_std2 = np.std(self.single_df[self.single_df["id_subject"].values == subjects[1]]["pr_p1_al_prec"].values)

                priors = np.asarray([prior_mean1, prior_mean2])
                std = np.asarray([prior_std1, prior_std2])

                priors_mean.append(priors[curr_data["hitter"].values[hitter_idx].astype(int)])
                priors_std.append(std[curr_data["hitter"].values[hitter_idx].astype(int)])
                # posterior
                post_al_p1_visual_angle_error.append(curr_data["hitter_pr_p1_al_prec"].values[hitter_idx])

                # sensory evidence
                sense_p1_visual_angle_error.append(prev_data["receiver_pr_p1_al_prec"].values[receiver_idx])
                sense_p2_visual_angle_error.append(prev_data["receiver_pr_p2_al_prec"].values[receiver_idx])
                sense_p3_pursuit_duration.append(prev_data["receiver_pr_p3_fx_duration"].values[receiver_idx])
                sense_swing_onset.append(prev_data["receiver_ec_start_fs"].values[receiver_idx])
                sense_distance_eye_hand.append(prev_data["receiver_distance_eye_hand"].values[receiver_idx])
                sense_racket_ball_angle.append(prev_data["receiver_im_racket_ball_angle"].values[receiver_idx])
                sense_racket_ball_wrist.append(prev_data["receiver_im_racket_ball_wrist"].values[receiver_idx])
                sense_im_ball_updown.append(prev_data["receiver_im_ball_updown"].values[receiver_idx])

                # segments (average length of a rally is 136) -> into (early, mid, late)
                observation_list.append(convertSegmentationLabel(curr_data["observation_label"].values[hitter_idx]))

                # add other
                receiver_list.append(receiver)
                hitter_list.append(hitter)
                session_list.append(session)

        fetures_summary = {
            # priors
            "priors_mean": np.concatenate(priors_mean),
            "priors_std": np.concatenate(priors_std),



            # posterior
            "post_visual_angle_error": np.concatenate(post_al_p1_visual_angle_error),

            # sense
            "sense_p1_visual_angle_error": np.concatenate(sense_p1_visual_angle_error),
            "sense_p2_visual_angle_error": np.concatenate(sense_p2_visual_angle_error),
            "sense_p3_pursuit_duration": np.concatenate(sense_p3_pursuit_duration),
            "sense_swing_onset": np.concatenate(sense_swing_onset),
            "sense_distance_eye_hand": np.concatenate(sense_distance_eye_hand),
            "sense_racket_ball_angle": np.concatenate(sense_racket_ball_angle),
            "sense_racket_ball_wrist": np.concatenate(sense_racket_ball_wrist),
            "sense_ball_updown": np.concatenate(sense_im_ball_updown),

            "observation_seg" : np.concatenate(observation_list),
            "receiver": np.concatenate(receiver_list),
            "hitter": np.concatenate(hitter_list),
            "session": np.concatenate(session_list),

        }
        return pd.DataFrame(fetures_summary)

    def getStableUnstableFailureFeatures(self, group_name="test", success_failure=False, mod="skill",
                                         with_control=False, timepoint=False, min_group_n=3):
        '''
        a function that gives the features of current and previous feature to predict the next states: stable, unstable, and failure
        :param group_name:
        :return:
        '''
        double_ecg = self.ecg_df[self.ecg_df["double_single"] == "D"]
        single_ecg = self.ecg_df[self.ecg_df["double_single"] == "S"]

        def genderSim(s1, s2):
            if ((s1 == "Man") & (s2 == "Man")):
                return 0.0
            elif ((s1 == "Woman") & (s2 == "Woman")):
                return 1.0
            else:
                return 2.0

        def rrSimilarityMean(x1, x2, b1, b2):

            x1 = x1[30:]
            x2 = x2[30:]

            # frequency domain
            x1_freq = hrvanalysis.get_frequency_domain_features(x1, method="lomb")
            x2_freq = hrvanalysis.get_frequency_domain_features(x2, method="lomb")

            # spatial
            x1_spatial = hrvanalysis.get_time_domain_features(x1)
            x2_spatial = hrvanalysis.get_time_domain_features(x2)

            hf_sim = np.abs(x1_freq["hf"] - x2_freq["hf"])
            lfhf_sim = np.abs(x1_freq["lf_hf_ratio"] - x2_freq["lf_hf_ratio"])
            rmsdd_sim = np.abs(x1_spatial["rmssd"] - x2_spatial["rmssd"])

            hf_mean = 0.5 * (x1_freq["hf"] + x2_freq["hf"])
            lfhf_mean = 0.5 * (x1_freq["lf_hf_ratio"] + x2_freq["lf_hf_ratio"])
            rmsdd_mean = 0.5 * (x1_spatial["rmssd"] + x2_spatial["rmssd"])

            return hf_sim, lfhf_sim, rmsdd_sim, hf_mean, lfhf_mean, rmsdd_mean

        def meSimilarityMean(x1, x2):
            me_foot_sim = x1["me_foot"] - x2["me_foot"]
            me_shoulder_arm_sim = x1["me_shoulder_arm"] - x2["me_shoulder_arm"]
            me_whole_sim = x1["me_whole"] - x2["me_whole"]

            me_foot_mean = np.abs(x1["me_foot"] + x2["me_foot"])
            me_shoulder_arm_mean = np.abs(x1["me_shoulder_arm"] + x2["me_shoulder_arm"])
            me_whole_mean = np.abs(x1["me_whole"] + x2["me_whole"])
            return me_foot_sim, me_shoulder_arm_sim, me_whole_sim, me_foot_mean, me_shoulder_arm_mean, me_whole_mean

        # success_df = self.df[self.df["success"] == 1]
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
        hitter_bouncing_to_partner_list = []

        team_spatial_position_list = []

        receiver_list = []
        hitter_list = []

        receiver_age_list = []
        hitter_age_list = []

        receiver_gender_list = []
        hitter_gender_list = []

        receiver_skill_list = []
        hitter_skill_list = []
        individuals_skill_list = []
        individuals_skill_sim_list = []
        individuals_skill_max_list = []
        gender_sim_list = []
        height_sim_list = []
        weight_sim_list = []
        age_sim_list = []
        relationship_list = []

        # ecg
        hf_sim_list = []
        lfhf_sim_list = []
        rmsdd_sim_list = []
        hf_mean_list = []
        lfhf_mean_list = []
        rmsdd_mean_list = []

        # me
        me_foot_sim_list = []
        me_shoulder_arm_sim_list = []
        me_whole_sim_list = []
        me_foot_mean_list = []
        me_shoulder_arm_mean_list = []
        me_whole_mean_list = []

        # other features
        bouncing_to_partner_list = []
        bouncing_to_self_list = []

        receiver_timepoint_list = []
        hitter_timepoint_list = []
        episode_list = []

        session_list = []

        for name, group in group_df:
            # n_data = len(group["hitter_pr_p2_al"]) - (n_segment - 1)
            # print(n_data)
            group_name = name[0]

            if len(group) > min_group_n:
                group.sort_values(by=['observation_label'])

                receiver_idx = np.concatenate([np.argwhere(group["success"].values == 1).flatten()[1:]])

                if len(receiver_idx) > 0:
                    hitter_idx = receiver_idx - 1

                    subjects = group[["id_subject1", "id_subject2"]].values[0]
                    age1 = self.single_summary_df[self.single_summary_df["Subject1"] == group["id_subject1"].values[0]][
                        "Age"].values
                    age2 = self.single_summary_df[self.single_summary_df["Subject1"] == group["id_subject2"].values[0]][
                        "Age"].values
                    gender1 = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == group["id_subject1"].values[0]][
                            "Gender"].values
                    gender2 = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == group["id_subject2"].values[0]][
                            "Gender"].values

                    ages = np.hstack([age1, age2])
                    genders = np.hstack([gender1, gender2])

                    s1_relationship = "R1"
                    s2_relationship = "R1"
                    all_df_s1 = self.double_summary_df[(self.double_summary_df["Subject1"] == subjects[0]) | (
                            self.double_summary_df["Subject2"] == subjects[0])].loc[:, "index_order"].values
                    all_df_s2 = self.double_summary_df[(self.double_summary_df["Subject1"] == subjects[1]) | (
                            self.double_summary_df["Subject2"] == subjects[1])].loc[:, "index_order"].values
                    g = self.df_summary[self.df_summary["file_name"] == group["session_id"].values[0]]

                    if (np.sum(all_df_s1 < g["index_order"].values[0]) == 1):
                        s1_relationship = "R2"
                    if (np.sum(all_df_s2 < g["index_order"].values[0]) == 1):
                        s2_relationship = "R2"

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

                    # me
                    me_double = self.me_df[self.me_df["session"] == group["session_id"].values[0]]
                    me_foot_sim, me_shoulder_arm_sim, me_whole_sim, me_foot_mean, me_shoulder_arm_mean, me_whole_mean = meSimilarityMean(
                        me_double.iloc[0], me_double.iloc[1])

                    # others
                    bouncing_to_partner = group["s1_bouncing_point_dist_p1"].fillna(0).values + group[
                        "s2_bouncing_point_dist_p1"].fillna(0).values
                    bouncing_to_partner = bouncing_to_partner[receiver_idx]
                    bouncing_to_self = group["hitter_position_to_bouncing_point"].values[receiver_idx]

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

                    # what the hitter does when the reciever takes an action
                    # distance when hitter hits - distance when receiver hits
                    # + not moving
                    # - moving back
                    hitter_at_and_after_hit = group["hitter_at_and_after_hit"].values[
                        receiver_idx]

                    # personal info
                    receiver = subjects[group["receiver"].values[receiver_idx].astype(int)]
                    hitter = subjects[group["hitter"].values[receiver_idx].astype(int)]

                    receiver_age = ages[group["receiver"].values[receiver_idx].astype(int)]
                    hitter_age = ages[group["hitter"].values[receiver_idx].astype(int)]

                    receiver_gender = genders[group["receiver"].values[receiver_idx].astype(int)]
                    hitter_gender = genders[group["hitter"].values[receiver_idx].astype(int)]

                    session = group["session_id"].values[receiver_idx]
                    receiver_skill = skill_subjects[group["receiver"].values[receiver_idx].astype(int)]
                    hitter_skill = skill_subjects[group["hitter"].values[receiver_idx].astype(int)]

                    individual_skill = 0.5 * (receiver_skill + hitter_skill)
                    individual_skill_sim = np.abs(receiver_skill - hitter_skill)
                    individual_skill_max = np.max(np.vstack([receiver_skill, hitter_skill]).T, axis=-1)

                    # gender
                    subject_1_gender = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == receiver[0]]["Gender"].values[0]
                    subject_2_gender = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == hitter[0]]["Gender"].values[0]
                    gender_sim = np.ones_like(receiver) * genderSim(subject_1_gender, subject_2_gender)

                    # height
                    subject_1_height = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == receiver[0]]["Height"].values[0]
                    subject_2_height = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == hitter[0]]["Height"].values[0]
                    height_sim = np.ones_like(receiver) * np.abs(subject_1_height - subject_2_height)

                    # weight
                    subject_1_weight = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == receiver[0]]["Weight"].values[0]
                    subject_2_weight = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == hitter[0]]["Weight"].values[0]
                    weight_sim = np.ones_like(receiver) * np.abs(subject_1_weight - subject_2_weight)

                    # age
                    subject_1_age = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == receiver[0]]["Age"].values[0]
                    subject_2_age = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == hitter[0]]["Age"].values[0]
                    age_sim = np.ones_like(receiver) * np.abs(subject_1_age - subject_2_age)

                    # relationship
                    subject_1_rel = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == receiver[0]][
                            s1_relationship].values[0]
                    subject_2_rel = \
                        self.single_summary_df[self.single_summary_df["Subject1"] == hitter[0]][s2_relationship].values[
                            0]
                    relationship_avg = np.ones_like(receiver) * (0.5 * (subject_1_rel + subject_2_rel))

                    # ECG
                    group_name_split = name[0].split("_")
                    trial_date = group_name_split[0]
                    trial_session = group_name_split[1]
                    trial_name = group_name_split[2]
                    trial_rr = double_ecg[
                        (double_ecg["date"] == trial_date) & (double_ecg["session"] == trial_session) & (
                                double_ecg["trial_name"] == trial_name)]
                    single_ecg_s1 = single_ecg[single_ecg["subject1"] == group["id_subject1"].values[0]]["rr1"].values[
                        0]
                    single_ecg_s2 = single_ecg[single_ecg["subject1"] == group["id_subject2"].values[0]]["rr1"].values[
                        0]
                    hf_sim, lfhf_sim, rmsdd_sim, hf_mean, lfhf_mean, rmsdd_mean = rrSimilarityMean(
                        trial_rr["rr1"].values[0],
                        trial_rr["rr2"].values[0], single_ecg_s1, single_ecg_s2)

                    # time point
                    receiver_timepoint = group["observation_label"].values[receiver_idx]
                    hitter_timepoint = group["observation_label"].values[receiver_idx]

                    episode_label = group["episode_label"].values[receiver_idx]

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

                    receiver_list.append(receiver)
                    hitter_list.append(hitter)
                    session_list.append(session)

                    receiver_age_list.append(receiver_age)
                    hitter_age_list.append(hitter_age)

                    receiver_gender_list.append(receiver_gender)
                    hitter_gender_list.append(hitter_gender)

                    receiver_skill_list.append(receiver_skill)
                    hitter_skill_list.append(hitter_skill)

                    receiver_timepoint_list.append(receiver_timepoint)
                    hitter_timepoint_list.append(hitter_timepoint)
                    episode_list.append(episode_label)

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
                    hitter_at_and_after_hit_list.append(hitter_at_and_after_hit)

                    team_spatial_position_list.append(team_spatial_position)

                    # personal info
                    individuals_skill_list.append(individual_skill)
                    individuals_skill_sim_list.append(individual_skill_sim)
                    individuals_skill_max_list.append(individual_skill_max)
                    gender_sim_list.append(gender_sim)
                    height_sim_list.append(height_sim)
                    weight_sim_list.append(weight_sim)
                    age_sim_list.append(age_sim)
                    relationship_list.append(relationship_avg)

                    # ecg
                    hf_sim_list.append(np.ones_like(receiver) * hf_sim)
                    lfhf_sim_list.append(np.ones_like(receiver) * lfhf_sim)
                    rmsdd_sim_list.append(np.ones_like(receiver) * rmsdd_sim)
                    hf_mean_list.append(np.ones_like(receiver) * hf_mean)
                    lfhf_mean_list.append(np.ones_like(receiver) * lfhf_mean)
                    rmsdd_mean_list.append(np.ones_like(receiver) * rmsdd_mean)

                    # me
                    me_foot_sim_list.append(np.ones_like(receiver) * me_foot_sim)
                    me_shoulder_arm_sim_list.append(np.ones_like(receiver) * me_shoulder_arm_sim)
                    me_whole_sim_list.append(np.ones_like(receiver) * me_whole_sim)
                    me_foot_mean_list.append(np.ones_like(receiver) * me_foot_mean)
                    me_shoulder_arm_mean_list.append(np.ones_like(receiver) * me_shoulder_arm_mean)
                    me_whole_mean_list.append(np.ones_like(receiver) * me_whole_mean)

                    # compensation
                    bouncing_to_partner_list.append(bouncing_to_partner)
                    bouncing_to_self_list.append(bouncing_to_self)

        fetures_summary = {

        }

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
                #
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

                "receiver_fixation_racket_latency": np.concatenate(receiver_fixation_racket_latency_list),
                "receiver_distance_eye_hand": np.concatenate(receiver_distance_eye_hand_list),

            })

        if "impact" in mod:
            fetures_summary.update({

                # contact
                "receiver_im_racket_ball_angle": np.concatenate(receiver_im_racket_ball_angle_list),
                "receiver_im_racket_ball_wrist": np.concatenate(receiver_im_racket_ball_wrist_list),
                "receiver_im_ball_wrist": np.concatenate(receiver_im_ball_wrist_list),

            })
        if "skill" in mod:
            fetures_summary.update({
                "individual_skill": np.concatenate(individuals_skill_list),
                "individual_skill_sim": np.concatenate(individuals_skill_sim_list),
                "individual_skill_max": np.concatenate(individuals_skill_max_list)
            })

        if "personal" in mod:
            fetures_summary.update({
                "gender_sim": np.concatenate(gender_sim_list).astype(float),
                # male-male: 0, female-female: 1, male-female: 2
                "height_sim": np.concatenate(height_sim_list).astype(float),  # height difference
                # "weight_sim": np.concatenate(weight_sim_list),  # weight difference
                "age_sim": np.concatenate(age_sim_list).astype(float),  # age same: 0, education differed: 1
                "relationship": np.concatenate(relationship_list).astype(float)
            })

        if "ecg" in mod:
            fetures_summary.update({
                "ecg_hf_sim": np.concatenate(hf_sim_list).astype(float),
                "ecg_lfhf_sim": np.concatenate(lfhf_sim_list).astype(float),
                "ecg_rmsdd_sim": np.concatenate(rmsdd_sim_list).astype(float),
                "ecg_hf_mean": np.concatenate(hf_mean_list).astype(float),
                "ecg_lfhf_mean": np.concatenate(lfhf_mean_list).astype(float),
                "ecg_rmsdd_mean": np.concatenate(rmsdd_mean_list).astype(float),
            })
        if "me" in mod:
            fetures_summary.update({
                "me_foot_sim": np.concatenate(me_foot_sim_list).astype(float),
                "me_shoulder_arm_sim": np.concatenate(me_shoulder_arm_sim_list).astype(float),
                "me_whole_sim": np.concatenate(me_whole_sim_list).astype(float),
                "me_foot_mean": np.concatenate(me_foot_mean_list).astype(float),
                "me_shoulder_arm_mean": np.concatenate(me_shoulder_arm_mean_list).astype(float),
                "me_whole_mean": np.concatenate(me_whole_mean_list).astype(float),
            })

        if "other" in mod:
            fetures_summary.update({
                "hitter_bouncing_to_partner": np.concatenate(bouncing_to_partner_list).astype(float),
                "hitter_bouncing_to_self": np.concatenate(bouncing_to_self_list).astype(float),
                "hitter_bouncing_to_ratio": np.concatenate(bouncing_to_self_list).astype(float) - np.concatenate(
                    bouncing_to_partner_list).astype(float),
                "hitter_at_and_after_hit": np.concatenate(hitter_at_and_after_hit_list),

            })

        if with_control:
            fetures_summary.update({"receiver": np.concatenate(receiver_list),
                                    "hitter": np.concatenate(hitter_list),
                                    "session": np.concatenate(session_list),
                                    "hitter_skill": np.concatenate(hitter_skill_list),
                                    "receiver_skill": np.concatenate(receiver_skill_list),

                                    "receiver_age": np.concatenate(receiver_age_list),
                                    "hitter_age": np.concatenate(hitter_age_list),

                                    "receiver_gender": np.concatenate(receiver_gender_list) == "Man",
                                    "hitter_gender": np.concatenate(hitter_gender_list) == "Man",

                                    })

        if timepoint:
            fetures_summary.update({"receiver_timepoint": np.concatenate(receiver_timepoint_list),
                                    "hitter_timepoint": np.concatenate(hitter_timepoint_list),
                                    "episode_label": np.concatenate(episode_list),

                                    })

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

        self.single_summary_df = pd.read_csv(SINGLE_SUMMARY_FILE_PATH)
        self.double_summary_df = pd.read_csv(DOUBLE_SUMMARY_FILE_PATH)
        self.ecg_df = pd.read_pickle(ECG_FEATURES_FILE_PATH)

        self.single_me = pd.read_pickle(SINGLE_ME_FEATURES_FILE_PATH)

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

    def getImpressionFeatures(self, n_index=10, group="control", mod="skill", return_group_skill=False,
                              return_control=False):
        if group == "lower":
            y = 0
        else:
            y = 1
        from scipy.spatial.distance import cdist

        def computeStyleSim(features_name, s1, s2, bins=7, is_int=False):

            if features_name == "rr":

                x1 = self.ecg_df[(self.ecg_df["subject1"] == s1) & (self.ecg_df["double_single"] == "S")]["rr1"].values[
                    0]
                x2 = self.ecg_df[(self.ecg_df["subject1"] == s2) & (self.ecg_df["double_single"] == "S")]["rr1"].values[
                    0]

                x1 = x1[30:]
                x2 = x2[30:]
                # frequency domain

                x1_freq = hrvanalysis.get_frequency_domain_features(x1, method="lomb")
                x2_freq = hrvanalysis.get_frequency_domain_features(x2, method="lomb")

                # spatial
                x1_spatial = hrvanalysis.get_time_domain_features(x1)
                x2_spatial = hrvanalysis.get_time_domain_features(x2)

                hf_sim = np.abs(x1_freq["hf"] - x2_freq["hf"])
                lfhf_sim = np.abs(x1_freq["lf_hf_ratio"] - x2_freq["lf_hf_ratio"])
                rmsdd_sim = np.abs(x1_spatial["rmssd"] - x2_spatial["rmssd"])

                return hf_sim, lfhf_sim, rmsdd_sim


            else:
                x1 = self.single_df[self.single_df["id_subject"] == s1][features_name].values
                x2 = self.single_df[self.single_df["id_subject"] == s2][features_name].values

                x1 = x1[~np.isnan(x1)]
                x2 = x2[~np.isnan(x2)]
                n_min = len(x2) if len(x1) > len(x2) else len(x1)
                x1 = x1[:n_min]
                x2 = x2[:n_min]
                # x = np.concatenate([x1, x2])

                # data1, bin_edges = np.histogram(x1, bins=bins, range=(np.min(x), np.max(x)), density=True)
                # data2 = np.histogram(x2, bins=bin_edges, density=True)[0]

                # data1 = data1 / np.sum(data1)
                # data2 = data2 / np.sum(data2)

                # relative comparison:  refer to evaluating the similarity or difference between two entities in relation to other pairs
                # Relative comparisons = "Is A more similar to B than C is to D?" (no need for correct scaling).
                # Absolute comparisons = "How much do A and B overlap?" (requires correct normalization).
                # bc_coeff = np.sum(np.sqrt(data1 * data2))
                # return bc_coeff

                # return jensenshannon(data1, data2)
                # return stats.kstest(x1, x2).statistic
                return wasserstein_distance(x1, x2)

        def computeMeanFeatures(features_name, s1, s2):
            if features_name == "rr":
                x1 = self.ecg_df[(self.ecg_df["subject1"] == s1) & (self.ecg_df["double_single"] == "S")][
                    "rr1"].values[0]
                x2 = self.ecg_df[(self.ecg_df["subject1"] == s2) & (self.ecg_df["double_single"] == "S")][
                    "rr1"].values[0]

                x1 = x1[30:]
                x2 = x2[30:]
                # frequency domain

                x1_freq = hrvanalysis.get_frequency_domain_features(x1, method="lomb")
                x2_freq = hrvanalysis.get_frequency_domain_features(x2, method="lomb")

                # spatial
                x1_spatial = hrvanalysis.get_time_domain_features(x1)
                x2_spatial = hrvanalysis.get_time_domain_features(x2)

                hf_mean = 0.5 * (x1_freq["hf"] + x2_freq["hf"])
                lfhf_mean = 0.5 * (x1_freq["lf_hf_ratio"] + x2_freq["lf_hf_ratio"])
                rmsdd_mean = 0.5 * (x1_spatial["rmssd"] + x2_spatial["rmssd"])

                return hf_mean, lfhf_mean, rmsdd_mean
            else:
                single_df = self.single_df
                x1 = single_df[single_df["id_subject"] == s1][features_name].values
                x2 = single_df[single_df["id_subject"] == s2][features_name].values
                # n_min = len(x2) if len(x1) > len(x2) else len(x1)
                # x1 = x1[:n_min]
                # x2 = x2[:n_min]
                return 0.5 * (np.nanmean(x1) + np.nanmean(x2))
                # return np.nanmean(np.concatenate([x1, x2]))

        # male-male: 0, female-female: 1, male-female: 2
        def genderSim(s1, s2):
            if ((s1 == "Man") & (s2 == "Man")):
                return 0
            elif ((s1 == "Woman") & (s2 == "Woman")):
                return 1
            else:
                return 2

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

        group_skill_list = []

        # subjects
        subject1_list = []
        subject2_list = []
        subject_skill_list = []
        subject_skill_sim_list = []
        subject_skill_max_list = []
        gender_list = []  # male-male: 0, female-female: 1, male-female: 2
        height_list = []  # height difference
        weight_list = []  # weight difference
        age_list = []  # education same: 0, education differed: 1
        relationship_list = []

        # ecg
        ecg_hf_sim_list = []
        ecg_lfhf_sim_list = []
        ecg_rmssd_sim_list = []
        ecg_hf_mean_list = []
        ecg_lfhf_mean_list = []
        ecg_rmssd_mean_list = []

        # me
        me_foot_sim_list = []
        me_shoulder_arm_sim_list = []
        me_whole_sim_list = []

        me_foot_mean_list = []
        me_shoulder_arm_mean_list = []
        me_whole_mean_list = []

        for _, g in self.df_summary.iterrows():
            s1 = g["Subject1"]
            s2 = g["Subject2"]
            s1_relationship = "R1"
            s2_relationship = "R1"
            all_df_s1 = self.double_summary_df[(self.double_summary_df["Subject1"] == s1) | (
                    self.double_summary_df["Subject2"] == s1)].loc[:, "index_order"].values
            all_df_s2 = self.double_summary_df[(self.double_summary_df["Subject1"] == s2) | (
                    self.double_summary_df["Subject2"] == s2)].loc[:, "index_order"].values

            if (np.sum(all_df_s1 < g["index_order"]) == 1):
                s1_relationship = "R2"
            if (np.sum(all_df_s2 < g["index_order"]) == 1):
                s2_relationship = "R2"
            me_single = self.single_me[(self.single_me["subject"] == s1) | (self.single_me["subject"] == s2)]
            # similarity
            # ECG similarity
            ecg_hf_sim, ecg_lfhf_sim, ecg_rmssd_sim = computeStyleSim("rr", s1, s2)

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
            ec_start_fs_sim = computeStyleSim("ec_start_fs", s1, s2)
            fixation_racket_latency_sim = computeStyleSim("fixation_racket_latency", s1, s2)
            distance_eye_hand_sim = computeStyleSim("distance_eye_hand", s1, s2)
            im_ball_updown_sim = computeStyleSim("im_ball_updown", s1, s2)

            # impact
            im_racket_ball_angle_sim = computeStyleSim("im_racket_ball_angle", s1, s2)
            im_racket_ball_wrist_sim = computeStyleSim("im_racket_ball_wrist", s1, s2)
            im_ball_wrist_sim = computeStyleSim("im_ball_wrist", s1, s2)

            # me

            me_foot_sim = np.abs(me_single.iloc[0]["me_foot"] - me_single.iloc[1]["me_foot"])
            me_shoulder_arm_sim = np.abs(me_single.iloc[0]["me_shoulder_arm"] - me_single.iloc[1]["me_shoulder_arm"])
            me_whole_sim = np.abs(me_single.iloc[0]["me_whole"] - me_single.iloc[1]["me_whole"])

            # means

            # ECG mean
            ecg_hf_mean, ecg_lfhf_mean, ecg_rmssd_mean = computeMeanFeatures("rr", s1, s2)

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

            # me

            me_foot_mean = 0.5 * (me_single.iloc[0]["me_foot"] + me_single.iloc[1]["me_foot"])
            me_shoulder_arm_mean = 0.5 * (me_single.iloc[0]["me_shoulder_arm"] + me_single.iloc[1]["me_shoulder_arm"])
            me_whole_mean = 0.5 * (me_single.iloc[0]["me_whole"] + me_single.iloc[1]["me_whole"])

            # skills
            subject_1_skill = self.single_df[self.single_df["id_subject"] == s1]["skill_subject"].values[0]
            subject_2_skill = self.single_df[self.single_df["id_subject"] == s2]["skill_subject"].values[0]

            skill_mean = 0.5 * (subject_1_skill + subject_2_skill)
            skill_sim = np.abs(subject_1_skill - subject_2_skill)
            skill_max = np.max([subject_1_skill, subject_2_skill])

            # gender
            subject_1_gender = self.single_summary_df[self.single_summary_df["Subject1"] == s1]["Gender"].values[0]
            subject_2_gender = self.single_summary_df[self.single_summary_df["Subject1"] == s2]["Gender"].values[0]
            gender_sim = genderSim(subject_1_gender, subject_2_gender)

            # height
            subject_1_height = self.single_summary_df[self.single_summary_df["Subject1"] == s1]["Height"].values[0]
            subject_2_height = self.single_summary_df[self.single_summary_df["Subject1"] == s2]["Height"].values[0]
            height_sim = np.abs(subject_1_height - subject_2_height)

            # weight
            subject_1_weight = self.single_summary_df[self.single_summary_df["Subject1"] == s1]["Weight"].values[0]
            subject_2_weight = self.single_summary_df[self.single_summary_df["Subject1"] == s2]["Weight"].values[0]
            weight_sim = np.abs(subject_1_weight - subject_2_weight)

            # age
            subject_1_age = self.single_summary_df[self.single_summary_df["Subject1"] == s1]["Age"].values[0]
            subject_2_age = self.single_summary_df[self.single_summary_df["Subject1"] == s2]["Age"].values[0]
            age_sim = np.abs(subject_1_age - subject_2_age)

            # relationship
            subject_1_rel = self.single_summary_df[self.single_summary_df["Subject1"] == s1][s1_relationship].values[0]
            subject_2_rel = self.single_summary_df[self.single_summary_df["Subject1"] == s2][s2_relationship].values[0]
            relationship_avg = 0.5 * (subject_1_rel + subject_2_rel)

            # group skill
            group_skill = g["skill"]

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

            # ecg
            ecg_hf_sim_list.append(ecg_hf_sim)
            ecg_lfhf_sim_list.append(ecg_lfhf_sim)
            ecg_rmssd_sim_list.append(ecg_rmssd_sim)

            # me
            me_foot_sim_list.append(me_foot_sim)
            me_shoulder_arm_sim_list.append(me_shoulder_arm_sim)
            me_whole_sim_list.append(me_whole_sim)

            # skill
            subject_skill_list.append(skill_mean)
            subject_skill_sim_list.append(skill_sim)
            subject_skill_max_list.append(skill_max)
            group_skill_list.append(group_skill)

            # personal

            gender_list.append(gender_sim)  # male-male: 0, female-female: 1, male-female: 2
            height_list.append(height_sim)  # height difference
            weight_list.append(weight_sim)  # weight difference
            age_list.append(age_sim)  # education same: 0, education differed: 1
            relationship_list.append(relationship_avg)

            # mean

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

            # ecg

            # mean
            ecg_hf_mean_list.append(ecg_hf_mean)
            ecg_lfhf_mean_list.append(ecg_lfhf_mean)
            ecg_rmssd_mean_list.append(ecg_rmssd_mean)

            me_foot_mean_list.append(me_foot_mean)
            me_shoulder_arm_mean_list.append(me_shoulder_arm_mean)
            me_whole_mean_list.append(me_whole_mean)

            # subject
            subject1_list.append(s1)
            subject2_list.append(s2)

        fetures_summary = {
            "labels": y
        }

        if "top-1" in mod:
            fetures_summary.update({
                "p1_al_prec_sim": p1_al_prec_sim_list,
            })

        if "top-2" in mod:
            fetures_summary.update({
                "p1_al_prec_sim": p1_al_prec_sim_list,
                "p1_cs_mean": p1_cs_mean_list,
            })

        if "top-3" in mod:
            fetures_summary.update({
                "p1_al_prec_sim": p1_al_prec_sim_list,
                "p1_cs_mean": p1_cs_mean_list,
                "im_racket_ball_wrist_mean": im_racket_ball_wrist_mean_list,

            })

        if "top-5" in mod:
            fetures_summary.update({
                "p1_al_prec_sim": p1_al_prec_sim_list,
                "p1_cs_mean": p1_cs_mean_list,
                "im_racket_ball_wrist_mean": im_racket_ball_wrist_mean_list,
                "im_racket_ball_angle_sim": im_racket_ball_angle_sim_list,
                "p1_al_prec_mean": p1_al_prec_mean_list,

            })
        if "skill" in mod:
            fetures_summary.update({
                "subject_skill": subject_skill_list,
                "subject_skill_sim": subject_skill_sim_list,
                # "subject_skill_max": subject_skill_max_list,
            })

        if "perception" in mod:
            fetures_summary.update({
                "p1_al_onset_sim": p1_al_on_sim_list,
                "p1_al_prec_sim": p1_al_prec_sim_list,
                "p1_al_mag_sim": p1_al_gM_sim_list,

                "p2_al_onset_sim": p2_al_on_sim_list,
                "p2_al_prec_sim": p2_al_prec_sim_list,
                "p2_al_mag_sim": p2_al_gM_sim_list,

                # "p3_fx_onset_sim": p3_fx_on_sim_list,
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
                # "p3_fx_onset_mean": p3_fx_on_mean_list,
                "p3_fx_du_mean": p3_fx_du_mean_list

            })

        if "action" in mod:
            fetures_summary.update({

                # action
                "ec_start_fs_sim": ec_start_fs_sim_list,
                # "fixation_racket_latency_sim": fixation_racket_latency_sim_list,
                "distance_eye_hand_sim": distance_eye_hand_sim_list,

                "ec_start_fs_mean": ec_start_fs_mean_list,
                # "fixation_racket_latency_mean": fixation_racket_latency_mean_list,
                "distance_eye_hand_mean": distance_eye_hand_mean_list,

            })

        if "impact" in mod:
            fetures_summary.update({
                # impact
                "im_racket_ball_angle_sim": im_racket_ball_angle_sim_list,
                "im_racket_ball_wrist_sim": im_racket_ball_wrist_sim_list,
                # "im_ball_wrist_sim": im_ball_wrist_sim_list,
                "im_ball_updown_sim": im_ball_updown_sim_list,

                "im_racket_ball_angle_mean": im_racket_ball_angle_mean_list,
                "im_racket_ball_wrist_mean": im_racket_ball_wrist_mean_list,
                # "im_ball_wrist_mean": im_ball_wrist_mean_list,
                "im_ball_updown_mean": im_ball_updown_mean_list,
            })

        if "personal" in mod:
            fetures_summary.update({
                "gender_sim": gender_list,  # male-male: 0, female-female: 1, male-female: 2
                "height_sim": height_list,  # height difference

                "age_sim": age_list,  # education same: 0, education differed: 1
                "relationship": relationship_list
            })

        if "ecg" in mod:
            fetures_summary.update({
                # sim
                # "ecg_hf_sim": ecg_hf_sim_list,
                "ecg_lfhf_sim": ecg_lfhf_sim_list,
                # "ecg_rmssd_sim": ecg_rmssd_sim_list,

                # mean
                # "ecg_hf_mean": ecg_hf_mean_list,
                "ecg_lfhf_mean": ecg_lfhf_mean_list,
                # "ecg_rmssd_mean": ecg_rmssd_mean_list,
            })
        if "me" in mod:
            fetures_summary.update({
                # sim
                # "me_foot_sim": me_foot_sim_list,
                # "me_shoulder_arm_sim": me_shoulder_arm_sim_list,
                "me_whole_sim": me_whole_sim_list,

                # mean
                # "me_foot_mean": me_foot_mean_list,
                # "me_shoulder_arm_mean": me_shoulder_arm_mean_list,
                "me_whole_mean": me_whole_mean_list,
            })
        if return_control:
            fetures_summary.update({
                "subject1": subject1_list,
                "subject2": subject2_list,

            })

        if return_group_skill:
            return pd.DataFrame(fetures_summary), group_skill_list
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
