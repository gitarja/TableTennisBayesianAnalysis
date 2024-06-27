import numpy as np
from scipy.special import logit, expit
import pandas as pd
from Utils.Conf import SINGLE_FEATURES_FILE_PATH
from scipy import stats
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

    def __init__(self, file_path="", file_summary_path="", include_subjects=["test"], exclude_failure=True, exclude_no_pair = False):
        df = pd.read_pickle(file_path)

        self.df_summary = pd.read_csv(file_summary_path)

        # select subjects subjects
        df = df.loc[df["session_id"].isin(include_subjects), :]



        if exclude_failure:
            # 0: failure
            # -1: stop
            df = df.loc[(df["success"] != 0)| (df["success"] != -1)]
        else:
            df = df.loc[df["success"] != -1]

        if exclude_no_pair:
            df = df.loc[df["pair_idx"] != -1]


        self.df = df



    def getGlobalFeatures(self, group_label="control"):

        single_df = pd.read_pickle(SINGLE_FEATURES_FILE_PATH)
        group_df = self.df.groupby(['session_id'])

        # receiver
        receiver_p1_al = []
        receiver_p2_al = []
        receiver_p1_al_prec = []
        receiver_p1_al_onset = []
        receiver_p2_al_prec = []
        receiver_p2_al_onset = []
        receiver_p1_cs = []
        receiver_p2_cs = []
        receiver_pursuit = []
        receiver_pursuit_duration = []


        receiver_start_fs_std = []
        receiver_racket_to_root_std = []
        receiver_fs_ball_racket_dir_std = []


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

        bouncing_point_var = []
        s1_bouncing_point_var = []
        s2_bouncing_point_var = []

        group_skill = []


        subject1 = []
        subject2 = []
        for name, group in group_df:

            # receiver
            receiver_p1_al.append(group["receiver_pr_p1_al"].mean())
            receiver_p2_al.append(group["receiver_pr_p2_al"].mean())
            receiver_pursuit.append(group["receiver_pr_p3_fx"].mean())
            receiver_pursuit_duration.append(group["receiver_pr_p3_fx_duration"].mean())
            receiver_p1_al_prec.append(group["receiver_pr_p1_al_prec"].mean())
            receiver_p1_al_onset.append(group["receiver_pr_p1_al_onset"].mean())
            receiver_p2_al_prec.append(group["receiver_pr_p2_al_prec"].mean())
            receiver_p2_al_onset.append(group["receiver_pr_p2_al_onset"].mean())
            receiver_p1_cs.append(group["receiver_pr_p1_cs"].mean())
            receiver_p2_cs.append(group["receiver_pr_p2_cs"].mean())




            # AL in P1 precision


            # hitter
            hitter_p1_al.append(group["hitter_pr_p1_al"].mean())
            hitter_p2_al.append(group["hitter_pr_p2_al"].mean())
            hitter_pursuit.append(group["hitter_pr_p3_fx"].mean())
            hitter_pursuit_duration.append(group["hitter_pr_p3_fx_duration"].mean())
            hitter_p1_al_prec.append(group["hitter_pr_p1_al_prec"].mean())
            hitter_p1_al_onset.append(group["hitter_pr_p1_al_onset"].mean())
            hitter_p2_al_prec.append(group["hitter_pr_p2_al_prec"].mean())
            hitter_p2_al_onset.append(group["hitter_pr_p2_al_onset"].mean())
            hitter_p1_cs.append(group["hitter_pr_p1_cs"].mean())
            hitter_p2_cs.append(group["hitter_pr_p2_cs"].mean())

            # bouncing point variance

            bouncing_point_var.append(0.5 * (group["s1_bouncing_point_dist"].mean() + group["s2_bouncing_point_dist"].mean()))
            s1_bouncing_point_var.append(group["s1_bouncing_point_dist"].mean())
            s2_bouncing_point_var.append(group["s2_bouncing_point_dist"].mean())

            # subject

            subject1.append(group["id_subject1"].values[0])
            subject2.append(group["id_subject2"].values[0])



            subject_1_sample = single_df.loc[single_df["id_subject"] == group["id_subject1"].values[0]]
            subject_2_sample = single_df.loc[single_df["id_subject"] == group["id_subject2"].values[0]]

            sim_score = stats.ks_2samp(subject_1_sample["ec_fs_ball_racket_ratio"].values, subject_2_sample["ec_fs_ball_racket_ratio"].values)

            single_movement_sim.append(sim_score.statistic)

            receiver_start_fs_std.append(group["receiver_ec_start_fs"].std())
            receiver_racket_to_root_std.append(group["receiver_racket_to_root"].std())
            receiver_fs_ball_racket_dir_std.append(group["receiver_ec_fs_ball_racket_dir"].std())

            racket_movement_sim.append(group["hand_movement_sim_lcss"].mean())


            # fixed effect
            group_skill.append(self.df_summary[self.df_summary["file_name"] ==name[0]]["skill"].values[0])



        fetures_summary = {
            "receiver_p1_al": np.asarray(receiver_p1_al),
            "receiver_p2_al": np.asarray(receiver_p2_al),
            "receiver_pursuit":  np.asarray(receiver_pursuit),
            "receiver_pursuit_duration": np.asarray(receiver_pursuit_duration),
            "receiver_p1_al_prec": np.asarray(receiver_p1_al_prec),
            "receiver_p1_al_onset": np.asarray(receiver_p1_al_onset),
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
            "receiver_racket_to_root_std": np.asarray(receiver_racket_to_root_std),
            "receiver_fs_ball_racket_dir_std": np.asarray(receiver_fs_ball_racket_dir_std),
            "hand_mov_sim": np.asarray(racket_movement_sim),
            "single_mov_sim": np.asarray(single_movement_sim),

            "bouncing_point_var": np.asarray(bouncing_point_var),
            "s1_bouncing_point_var": np.asarray(s1_bouncing_point_var),
            "s2_bouncing_point_var": np.asarray(s2_bouncing_point_var),

            "group_skill": np.asarray(group_skill),
            "subject1": np.asarray(subject1),
            "subject2": np.asarray(subject2),
            "group": group_label
        }

        return pd.DataFrame(fetures_summary)


    def getSegmentateFeatures(self, group_label="control", n_segment=5):
        mean_value = self.df['receiver_pr_p1_al_prec'].mean()

        # Replace NaNs in column S2 with the
        # mean of values in the same column
        self.df['receiver_pr_p1_al_prec'].fillna(value=mean_value, inplace=True)


        group_df = self.df.groupby(['session_id'])

        forward_sim = []
        racket_mov_sim = []
        hitter_pf_rate = []
        hitter_al1_rate = []
        hitter_al2_rate = []
        hitter_pf_duration = []
        receiver_p1_al_prec = []
        th_segments = []
        session_id = []


        for name, group in group_df:
            n_data = len(group["hitter_pr_p2_al"])- (n_segment-1)
            # print(n_data)
            group.sort_values(by=['observation_label'])

            group['hitter_pr_p3_fx_duration_clean'] = group["hitter_pr_p3_fx_duration"].fillna(0)
            forward_sim.append(group["forward_swing_sim"].rolling(window=n_segment).mean().values[n_segment-1:])
            racket_mov_sim.append(group["hand_movement_sim_lcss"].rolling(window=n_segment).mean().values[n_segment-1:])
            hitter_pf_rate.append(group["hitter_pr_p3_fx"].rolling(window=n_segment).sum().values[n_segment-1:])
            hitter_al1_rate.append(group["hitter_pr_p1_al"].rolling(window=n_segment).mean().values[n_segment-1:])
            hitter_al2_rate.append(group["hitter_pr_p2_al"].rolling(window=n_segment).mean().values[n_segment-1:])
            hitter_pf_duration.append(group["hitter_pr_p3_fx_duration_clean"].rolling(window=n_segment).mean().values[n_segment-1:])

            receiver_p1_al_prec.append(group["receiver_pr_p1_al_prec"].rolling(window=n_segment).mean().values[n_segment-1:])
            th_segments.append(np.log2(group["observation_label"].rolling(window=n_segment).mean().values[n_segment-1:]+1))
            session_id.append(group.session_id.values[n_segment-1:])



        #normalize racket movement sim
        # racket_mov_sim = np.concatenate(racket_mov_sim)
        # racket_mov_sim_norm = (racket_mov_sim - np.mean(racket_mov_sim)) / np.std(racket_mov_sim)


        fetures_summary = {

            "forward_swing_sim": np.concatenate(forward_sim),
            "racket_mov_sim": np.concatenate(racket_mov_sim),
            "hitter_pf_rate": np.concatenate(hitter_pf_rate).astype(int),
            "hitter_al1_rate": np.concatenate(hitter_al1_rate),
            "hitter_al2_rate": np.concatenate(hitter_al2_rate),
            "hitter_pf_duration": np.concatenate(hitter_pf_duration),
            "receiver_p1_al_prec": np.concatenate(receiver_p1_al_prec),
            "th_segments": np.concatenate(th_segments),

            "group": group_label,
            "session_id": np.concatenate(session_id),
        }

        return pd.DataFrame(fetures_summary)


