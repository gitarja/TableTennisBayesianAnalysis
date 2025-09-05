import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling

from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import os
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH
import seaborn as sns

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20

if __name__ == '__main__':

    inefficient_group, efficient_group = groupLabeling()

    # inefficient group
    inefficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                    file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                    include_subjects=inefficient_group, exclude_failure=True,
                                                    exclude_no_pair=False, hmm_probs=True)

    efficient_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                                  file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                                  include_subjects=efficient_group, exclude_failure=True,
                                                  exclude_no_pair=False, hmm_probs=True)
    mod = "skill_personal_perception_action_impact_ecg_me_other"
    inefficient_features = inefficient_reader.getStableUnstableFailureFeatures(group_name="inefficient",
                                                                               success_failure=True,
                                                                               mod=mod,
                                                                               with_control=True, timepoint=True,
                                                                               min_group_n=3)
    inefficient_features["group"] = "inefficient"
    # efficient group

    efficient_features = efficient_reader.getStableUnstableFailureFeatures(group_name="efficient", success_failure=True,
                                                                           mod=mod,
                                                                           with_control=True, timepoint=True,
                                                                           min_group_n=3)
    efficient_features["group"] = "efficient"

    df = pd.concat([inefficient_features, efficient_features])
    type_features = "contact"
    if type_features == "eyes":
        features = [

            "hitter_p1_al",
            "hitter_p2_al",
            "hitter_fx",
            "receiver_p1_al",
            "receiver_p2_al",
            "receiver_p3_fx",
            "hitter_p1_al_onset",
            "hitter_p1_al_prec",
            "hitter_p1_al_mag",
            "hitter_p1_cs",
            "hitter_p2_al_onset",
            "hitter_p2_al_prec",
            "hitter_p2_al_mag",
            "hitter_p2_cs",
            "hitter_fx_duration",
            "receiver_p1_al_onset",
            "receiver_p1_al_prec",
            "receiver_p1_al_mag",
            "receiver_p1_cs",
            "receiver_p2_al_onset",
            "receiver_p2_al_prec",
            "receiver_p2_al_mag",
            "receiver_p2_cs",
            "receiver_p3_fx_duration",

        ]
        fig, axes = plt.subplots(2, 4, figsize=(8.27, 11.69))  # Adjust figsize as needed
    if type_features == "contact":
        features = [

            "receiver_start_fs",
            "receiver_distance_eye_hand",

            "receiver_im_racket_ball_wrist",
            "receiver_im_racket_ball_angle",
            "receiver_im_ball_updown",
            "hitter_bouncing_to_partner",

            "hitter_at_and_after_hit"

        ]
        fig, axes = plt.subplots(2, 4, figsize=(8.27, 3.89))  # Adjust figsize as needed

    if type_features == "others":
        df = df.drop_duplicates(subset=["session"])
        features = [
            "ecg_lfhf_sim",
            "ecg_lfhf_mean",

            "me_whole_sim",
            "me_whole_mean",

            "individual_skill",
            "individual_skill_sim",
            "gender_sim",
            "age_sim",
            "height_sim",
            "relationship",

        ]
        fig, axes = plt.subplots(3, 4, figsize=(8.27, 5.85))  # Adjust figsize as needed
    plt.rcParams["patch.force_edgecolor"] = False

    n_cols = len(features)

    i = 0
    for f in features:
        row, col_idx = divmod(i, 4)
        ax = axes[row, col_idx]
        sns.histplot(df[f], ax=ax, fill=True, edgecolor=None, element="step")
        ax.set_title(f, fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=8)
        i += 1

    # Hide any unused axes (in case < 48 columns)
    for j in range(len(df.columns), 24):
        row, col_idx = divmod(j, 8)
        axes[row, col_idx].axis('off')

    plt.tight_layout()

    results_path = "F:\\users\\prasetia\\Personal-OneDrive\\OneDrive\\ExperimentResults\\DoubleTennis\\Final2\\Figures\\Material\\"
    plt.savefig(os.path.join(results_path, type_features + "_distribution.pdf"))
