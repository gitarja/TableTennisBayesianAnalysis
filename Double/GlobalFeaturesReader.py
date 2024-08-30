import numpy as np
from scipy.special import logit, expit
import pandas as pd
from Utils.Conf import SINGLE_FEATURES_FILE_PATH, NORMALIZE_X_DOUBLE_EPISODE_COLUMNS, HMM_MODEL_PATH
from scipy import stats
from sklearn.impute import KNNImputer
import torch
from scipy.ndimage import label


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
                 exclude_no_pair=False, hmm_probs=False):

        self.df_summary = pd.read_csv(file_summary_path)

        self.df = pd.read_pickle(file_path)

        if hmm_probs:
            self.df = self.timeSeriesFeatures()

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
                        pref = "_"+ prefix[0]
                    else:
                        reciver_stat_idx = stay_unstable_receiver_idx
                        hitter_stat_idx = stay_unstable_hitter_idx
                        hitter_stat_idx2 = stay_unstable_hitter_idx2
                        pref =  "_"+ prefix[1]


                    # recovered effort

                    # subject
                    if i == 0:
                        subject.append(group["id_subject1"].values[0])
                        subject_skill.append(group["skill_subject1"].values[0])
                        recover_bouncing_point_var_p1.append(group[hitter_stat_idx2]["s1_bouncing_point_dist_p1"].mean())

                    else:
                        subject.append(group["id_subject2"].values[0])
                        subject_skill.append(group["skill_subject2"].values[0])
                        recover_bouncing_point_var_p1.append(group[hitter_stat_idx2]["s2_bouncing_point_dist_p1"].mean())


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

                    recover_gaze_entropy.append(group[reciver_stat_idx]["receiver_gaze_entropy"].replace([np.inf, -np.inf], np.nan).dropna().mean())
                    recover_gaze_ball_relDiv.append(group[reciver_stat_idx]["receiver_gaze_ball_relDiv"].replace([np.inf, -np.inf], np.nan).dropna().mean())

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
        receiver_p2_al_onset = []
        receiver_p1_cs = []
        receiver_p2_cs = []
        receiver_pursuit = []
        receiver_pursuit_duration = []

        receiver_start_fs_std = []
        receiver_start_fs_mean = []
        receiver_racket_to_root_std = []
        receiver_racket_ball_force_std = []
        receiver_fs_ball_racket_dir_std = []
        receiver_ec_fs_ball_rball_dist = []

        # hitter
        hitter_p1_al = []
        hitter_p2_al = []
        hitter_p1_al_prec = []
        hitter_p1_al_onset = []
        hitter_p2_al_prec = []
        hitter_p2_al_onset = []
        hitter_p1_cs = []
        hitter_p2_cs = []
        hitter_pursuit = []
        hitter_pursuit_duration = []

        racket_movement_sim = []
        single_movement_sim = []

        bouncing_point_var_p1 = []
        bouncing_point_var_p2 = []

        stable_state = []
        unstable_duration = []

        group_skill = []
        subject_skill = []

        subject = []
        for name, group in group_df:
            # set the labels for unstable
            unstable_idx = ((group["stable_probs"] <= 0.5) & (group["stable_probs"] != -1)).astype(float)

            unstable_diff = np.pad(np.diff(unstable_idx), (0, 1), "edge")
            observable_diff = np.pad(np.diff(group["observation_label"].values), (0, 1), "edge")
            unstable_episode = np.asarray((unstable_diff ==0) & (unstable_idx == 1) & (observable_diff == 1))

            state_groups, num_groups = label(unstable_episode)

            # duration of states
            durations_list = []
            for i in np.unique(state_groups)[1:]:
                duration = np.sum(state_groups == i)
                durations_list.append(duration)

            # receiver
            for i in range(2):
                receiver_idx = group["receiver"] == i
                hitter_idx = group["hitter"] == i

                unstable_duration.append(np.mean(durations_list))
                receiver_p1_al.append(group[receiver_idx]["receiver_pr_p1_al"].mean())
                receiver_p2_al.append(group[receiver_idx]["receiver_pr_p2_al"].mean())
                receiver_pursuit.append(group[receiver_idx]["receiver_pr_p3_fx"].mean())
                receiver_pursuit_duration.append(group[receiver_idx]["receiver_pr_p3_fx_duration"].mean())
                receiver_p1_al_prec.append(group[receiver_idx]["receiver_pr_p1_al_prec"].mean())
                receiver_p1_al_onset.append(group[receiver_idx]["receiver_pr_p1_al_onset"].mean())
                receiver_p1_al_mag.append(group[receiver_idx]["receiver_pr_p1_al_mag"].mean())
                receiver_p2_al_prec.append(group[receiver_idx]["receiver_pr_p2_al_prec"].mean())
                receiver_p2_al_onset.append(group[receiver_idx]["receiver_pr_p2_al_onset"].mean())
                receiver_p1_cs.append(group[receiver_idx]["receiver_pr_p1_cs"].mean())
                receiver_p2_cs.append(group[receiver_idx]["receiver_pr_p2_cs"].mean())

                # AL in P1 precision

                # hitter
                hitter_p1_al.append(group[hitter_idx]["hitter_pr_p1_al"].mean())
                hitter_p2_al.append(group[hitter_idx]["hitter_pr_p2_al"].mean())
                hitter_pursuit.append(group[hitter_idx]["hitter_pr_p3_fx"].mean())
                hitter_pursuit_duration.append(group[hitter_idx]["hitter_pr_p3_fx_duration"].mean())
                hitter_p1_al_prec.append(group[hitter_idx]["hitter_pr_p1_al_prec"].mean())
                hitter_p1_al_onset.append(group[hitter_idx]["hitter_pr_p1_al_onset"].mean())
                hitter_p2_al_prec.append(group[hitter_idx]["hitter_pr_p2_al_prec"].mean())
                hitter_p2_al_onset.append(group[hitter_idx]["hitter_pr_p2_al_onset"].mean())
                hitter_p1_cs.append(group[hitter_idx]["hitter_pr_p1_cs"].mean())
                hitter_p2_cs.append(group[hitter_idx]["hitter_pr_p2_cs"].mean())

                # stable state
                stable_idx = np.argwhere(group[hitter_idx]["stable_probs"] > -1).flatten()
                stable_state.append(np.average(group[hitter_idx]["stable_probs"].values[stable_idx] > 0.5))


                # subject
                if i == 0:
                    subject.append(group["id_subject1"].values[0])
                    subject_skill.append(group["skill_subject1"].values[0])
                    bouncing_point_var_p1.append(group[hitter_idx]["s1_bouncing_point_dist_p1"].mean())
                    bouncing_point_var_p2.append(group[hitter_idx]["s1_bouncing_point_dist_p2"].mean())
                else:
                    subject.append(group["id_subject2"].values[0])
                    subject_skill.append(group["skill_subject2"].values[0])
                    bouncing_point_var_p1.append(group[hitter_idx]["s2_bouncing_point_dist_p1"].mean())
                    bouncing_point_var_p2.append(group[hitter_idx]["s2_bouncing_point_dist_p2"].mean())

                subject_1_sample = single_df.loc[single_df["id_subject"] == group["id_subject1"].values[0]]
                subject_2_sample = single_df.loc[single_df["id_subject"] == group["id_subject2"].values[0]]

                sim_score = stats.ks_2samp(subject_1_sample["ec_fs_ball_racket_ratio"].values,
                                           subject_2_sample["ec_fs_ball_racket_ratio"].values)

                single_movement_sim.append(sim_score.statistic)

                receiver_start_fs_std.append(group[receiver_idx]["receiver_ec_start_fs"].std())
                receiver_start_fs_mean.append(group[receiver_idx]["receiver_ec_start_fs"].mean())
                receiver_racket_to_root_std.append(group[receiver_idx]["receiver_racket_to_root"].std())
                receiver_fs_ball_racket_dir_std.append(group[receiver_idx]["receiver_ec_fs_ball_racket_dir"].std())

                receiver_racket_ball_force_std.append(
                    np.std(group[receiver_idx]["receiver_im_ball_force"] / group[receiver_idx][
                        "receiver_im_racket_force"]))

                racket_movement_sim.append(group[receiver_idx]["hand_movement_sim_dtw"].mean())

                # fixed effect
                group_skill.append(self.df_summary[self.df_summary["file_name"] == name[0]]["skill"].values[0])

        fetures_summary = {
            "receiver_p1_al": np.asarray(receiver_p1_al),
            "receiver_p2_al": np.asarray(receiver_p2_al),
            "receiver_pursuit": np.asarray(receiver_pursuit),
            "receiver_pursuit_duration": np.asarray(receiver_pursuit_duration),
            "receiver_p1_al_prec": np.asarray(receiver_p1_al_prec),
            "receiver_p1_al_onset": np.asarray(receiver_p1_al_onset),
            "receiver_p1_al_mag": np.asarray(receiver_p1_al_mag),
            "receiver_p2_al_prec": np.asarray(receiver_p2_al_prec),
            "receiver_p2_al_onset": np.asarray(receiver_p2_al_onset),
            "receiver_p1_cs": np.asarray(receiver_p1_cs),
            "receiver_p2_cs": np.asarray(receiver_p2_cs),

            "hitter_p1_al": np.asarray(hitter_p1_al),
            "hitter_p2_al": np.asarray(hitter_p2_al),
            "hitter_pursuit": np.asarray(hitter_pursuit),
            "hitter_pursuit_duration": np.asarray(hitter_pursuit_duration),
            "hitter_p1_al_prec": np.asarray(hitter_p1_al_prec),
            "hitter_p1_al_onset": np.asarray(hitter_p1_al_onset),
            "hitter_p2_al_prec": np.asarray(hitter_p2_al_prec),
            "hitter_p2_al_onset": np.asarray(hitter_p2_al_onset),
            "hitter_p1_cs": np.asarray(hitter_p1_cs),
            "hitter_p2_cs": np.asarray(hitter_p2_cs),

            "receiver_start_fs_std": np.asarray(receiver_start_fs_std),
            "receiver_start_fs_mean": np.asarray(receiver_start_fs_mean),
            "receiver_racket_to_root_std": np.asarray(receiver_racket_to_root_std),
            "receiver_fs_ball_racket_dir_std": np.asarray(receiver_fs_ball_racket_dir_std),
            "receiver_racket_ball_force_ratio_std": np.asarray(receiver_racket_ball_force_std),

            "hand_mov_sim": np.asarray(racket_movement_sim),
            "single_mov_sim": np.asarray(single_movement_sim),

            "bouncing_point_var_p1": np.asarray(bouncing_point_var_p1),
            "bouncing_point_var_p2": np.asarray(bouncing_point_var_p2),

            "stable_percentage": np.asarray(stable_state),
            "unstable_duration": np.asarray(unstable_duration),


            "group_skill": np.asarray(group_skill),
            "subject": np.asarray(subject),
            "subject_skill": np.asarray(subject_skill),
            "group": group_label
        }

        return pd.DataFrame(fetures_summary)


    def getSegmentateFeatures(self, group_label="control", n_segment=5):

        # group_idx = -1
        # if group_label == "control":
        #     group_idx = 0
        # elif group_label == "inefficient":
        #     group_idx = 1

        mean_value = self.df['receiver_pr_p1_al_prec'].mean()

        # Replace NaNs in column S2 with the
        # mean of values in the same column
        self.df['receiver_pr_p1_al_prec'].fillna(value=mean_value, inplace=True)

        group_df = self.df.groupby(['session_id'])

        hand_mov_sim = []
        hitter_pf_rate = []
        receiver_p1_al_prec = []
        stable_state = []
        th_segments = []
        subject_skill = []
        subject = []

        for name, group in group_df:
            # n_data = len(group["hitter_pr_p2_al"]) - (n_segment - 1)
            # print(n_data)
            group.sort_values(by=['observation_label'])
            hitter_pursuit_seg = group["hitter_pr_p3_fx"]
            hand_mov_sim_seg = group["hand_movement_sim_dtw"]
            receiver_p1_al_prec_seg = group["receiver_pr_p1_al_prec"]
            stable_state_seg = group["stable_probs"] > 0.5

            for i in range(2):
                receiver_idx = group["receiver"] == i
                hitter_idx = group["hitter"] == i
                n_data = len(hand_mov_sim_seg[receiver_idx].rolling(window=n_segment).mean().values[n_segment - 1:])


                hand_mov_sim.append(hand_mov_sim_seg[receiver_idx].rolling(window=n_segment).mean().values[n_segment-1:])

                hitter_pf_rate.append(hitter_pursuit_seg[hitter_idx].rolling(window=n_segment).sum().values[n_segment-1:])
                receiver_p1_al_prec.append(receiver_p1_al_prec_seg[receiver_idx].rolling(window=n_segment).mean().values[n_segment-1:])
                stable_state.append(stable_state_seg[hitter_idx].rolling(window=n_segment).mean().values[n_segment-1:])

                th_segments.append(np.log(group["observation_label"][receiver_idx].rolling(window=n_segment).mean().values[n_segment-1:] + 1))
                # th_segments.append(np.ones(shape=(n_data, )))
                if i == 0:
                    subject.append(group[receiver_idx]["id_subject1"].values[:n_data])
                    subject_skill.append(group[receiver_idx]["skill_subject1"].values[:n_data])

                else:
                    subject.append(group[receiver_idx]["id_subject2"].values[:n_data])
                    subject_skill.append(group[receiver_idx]["skill_subject2"].values[:n_data])


        # normalize racket movement sim
        # racket_mov_sim = np.concatenate(racket_mov_sim)
        # racket_mov_sim_norm = (racket_mov_sim - np.mean(racket_mov_sim)) / np.std(racket_mov_sim)

        fetures_summary = {
            "hand_mov_sim": np.concatenate(hand_mov_sim),
            "hitter_pf_rate": np.concatenate(hitter_pf_rate),
            "receiver_al_prec": np.concatenate(receiver_p1_al_prec),
            "stable_rate": np.concatenate(stable_state),
            "th_segments": np.concatenate(th_segments),
            "subject": np.concatenate(subject),
            "subject_skill": np.concatenate(subject_skill),

            "group": group_label,
        }

        return pd.DataFrame(fetures_summary)

    def timeSeriesFeatures(self):

        df_summary = self.df_summary
        df = self.df
        df_summary = df_summary[(df_summary["norm_score"] > 0.55) & (df_summary["Tobii_percentage"] > 65)]

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
        model = torch.load(HMM_MODEL_PATH)
        for i, g in grouped_episodes:
            g = g.sort_values(by=['observation_label'])
            # g = g.loc[g["receiver"].values==1]
            signal1 = g["ball_speed_after_hit"].values
            signal2 = g["ball_dir_after_hit"].values
            signal3 = g["bouncing_point_dist_p1"].values
            signal4 = g["bouncing_point_dist_p2"].values

            if len(signal1) > 0:
                X = np.vstack([signal1, signal2, signal3, signal4]).T
                X = np.expand_dims(X, axis=0)
                probs = model.predict_proba(X).numpy()[0][:, 0]  # 0: stable 1: unstable
                preds = model.predict(X).numpy()[0]

                conditions = np.argwhere(
                    (self.df["session_id"] == g["session_id"].values[0]) & self.df["observation_label"].isin(
                        g["observation_label"].values)).flatten()
                self.df.loc[conditions, "stable_probs"] = probs
                self.df.loc[conditions, "unstable_preds"] = preds

        return self.df


if __name__ == '__main__':
    from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH

    # control group
    reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                        file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                        include_subjects=None, exclude_failure=True,
                                        exclude_no_pair=False)

    reader.timeSeriesFeatures()
