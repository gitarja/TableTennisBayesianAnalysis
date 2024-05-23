import numpy as np
from scipy.special import logit, expit
import pandas as pd

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

    def __init__(self, file_path="", include_subjects=["test"], exclude_failure=True, exclude_no_pair = False):
        df = pd.read_pickle(file_path)

        # select subjects subjects
        df = df.loc[df["session_id"].isin(include_subjects), :]



        if exclude_failure:
            df = df.loc[(df["success"] != 0)| (df["success"] != -1)]
        else:
            df = df.loc[df["success"] != -1]

        if exclude_no_pair:
            df = df.loc[df["pair_idx"] != -1]


        self.df = df



    def getGlobalFeatures(self, group_label="control"):

        group_df = self.df.groupby(['session_id'])
        receiver_p1_al = []
        receiver_p2_al = []
        receiver_pursuit = []
        receiver_pursuit_duration = []
        receiver_start_fs_std = []
        receiver_racket_to_root_std = []
        receiver_fs_ball_racket_dir_std = []


        hitter_p1_al = []
        hitter_p2_al = []

        hitter_pursuit = []
        hitter_pursuit_duration = []

        subject1 = []
        subject2 = []
        for name, group in group_df:

            # percentage of AL in phase 1 and 2
            receiver_p1_al.append(group["receiver_pr_p1_al"].mean())
            receiver_p2_al.append(group["receiver_pr_p2_al"].mean())
            receiver_pursuit.append(group["receiver_pr_p3_fx"].mean())
            receiver_pursuit_duration.append(group["receiver_pr_p3_fx_duration"].mean())

            hitter_p1_al.append(group["hitter_pr_p1_al"].mean())
            hitter_p2_al.append(group["hitter_pr_p2_al"].mean())
            # duration of pursuit
            hitter_pursuit.append(group["hitter_pr_p3_fx"].mean())
            hitter_pursuit_duration.append(group["hitter_pr_p3_fx_duration"].mean())


            subject1.append(group["id_subject1"].values[0])
            subject2.append(group["id_subject2"].values[0])

            receiver_start_fs_std.append(group["receiver_ec_start_fs"].std())
            receiver_racket_to_root_std.append(group["receiver_racket_to_root"].std())
            receiver_fs_ball_racket_dir_std.append(group["receiver_ec_fs_ball_racket_dir"].std())



        fetures_summary = {
            "receiver_p1_al": np.asarray(receiver_p1_al),
            "receiver_p2_al": np.asarray(receiver_p2_al),
            "receiver_pursuit":  np.asarray(receiver_pursuit),
            "receiver_pursuit_duration": np.asarray(receiver_pursuit_duration),
            "hitter_p1_al": np.asarray(hitter_p1_al),
            "hitter_p2_al": np.asarray(hitter_p2_al),
            "hitter_pursuit": np.asarray(hitter_pursuit),
            "hitter_pursuit_duration": np.asarray(hitter_pursuit_duration),
            "subject1": np.asarray(subject1),
            "subject2": np.asarray(subject2),
            "receiver_start_fs_std": np.asarray(receiver_start_fs_std),
            "receiver_racket_to_root_std": np.asarray(receiver_racket_to_root_std),
            "receiver_fs_ball_racket_dir_std": np.asarray(receiver_fs_ball_racket_dir_std),
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
            racket_mov_sim.append(group["racket_movement_sim_lcc"].rolling(window=n_segment).mean().values[n_segment-1:])
            hitter_pf_rate.append(group["hitter_pr_p3_fx"].rolling(window=n_segment).sum().values[n_segment-1:])
            hitter_al1_rate.append(group["hitter_pr_p1_al"].rolling(window=n_segment).mean().values[n_segment-1:])
            hitter_al2_rate.append(group["hitter_pr_p2_al"].rolling(window=n_segment).mean().values[n_segment-1:])
            hitter_pf_duration.append(group["hitter_pr_p3_fx_duration_clean"].rolling(window=n_segment).mean().values[n_segment-1:])

            receiver_p1_al_prec.append(group["receiver_pr_p1_al_prec"].rolling(window=n_segment).mean().values[n_segment-1:])
            th_segments.append(np.log2(group["observation_label"].rolling(window=n_segment).mean().values[n_segment-1:]+1))
            session_id.append(group.session_id.values[n_segment-1:])



        fetures_summary = {

            "forward_swing_sim": np.concatenate(forward_sim),
            "racket_mov_sim": np.concatenate(racket_mov_sim),
            "hitter_pf_rate": np.concatenate(hitter_pf_rate),
            "hitter_al1_rate": np.concatenate(hitter_al1_rate),
            "hitter_al2_rate": np.concatenate(hitter_al2_rate),
            "hitter_pf_duration": np.concatenate(hitter_pf_duration),
            "receiver_p1_al_prec": np.concatenate(receiver_p1_al_prec),
            "th_segments": np.concatenate(th_segments),

            "group": group_label,
            "session_id": np.concatenate(session_id),
        }

        return pd.DataFrame(fetures_summary)


