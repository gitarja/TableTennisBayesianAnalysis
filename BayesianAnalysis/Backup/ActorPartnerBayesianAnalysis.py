import matplotlib.pyplot as plt

from Utils.GroupClassification import groupLabeling
import numpy as np
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_SUMMARY_FILE_PATH, features_explanation
from Double.GlobalFeaturesReader import GlobalDoubleFeaturesReader
import pandas as pd
import pymc as pm
from Utils.Conf import N_CORE, N_TUNE, N_CHAINS, N_SAMPLES, DOUBLE_SUMMARY_FEATURES_PATH, DOUBLE_RESULTS_PATH_ANOVA, \
    DOUBLE_SUMMARY_FILE_PATH, TARGET_ACC, DOUBLE_RESULTS_PATH_APM_TTEST
import arviz as az
import pickle
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from ActorPartnerModel import CenteredModel

# PREV_ANALYZED_FEATURES = ["onset_forward_swing"]
# NEXT_ANALYZED_FEATURES = ["onset_forward_swing"]
# HITTER_BOOL = [False]


PREV_ANALYZED_FEATURES = [ "ball_updown", "onset_forward_swing"]
NEXT_ANALYZED_FEATURES = [  "ball_updown", "onset_forward_swing"]
HITTER_BOOL = [False, False, False]

# PREV_ANALYZED_FEATURES = ["al_prec_p1"]
# NEXT_ANALYZED_FEATURES = ["al_prec_p1"]
# HITTER_BOOL = [ False]

if __name__ == '__main__':

    lower_group, upper_group = groupLabeling()

    # inefficient group
    lower_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                              file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                              include_subjects=lower_group, exclude_failure=True,
                                              exclude_no_pair=False, hmm_probs=True)

    higher_reader = GlobalDoubleFeaturesReader(file_path=DOUBLE_SUMMARY_FEATURES_PATH,
                                               file_summary_path=DOUBLE_SUMMARY_FILE_PATH,
                                               include_subjects=upper_group, exclude_failure=True,
                                               exclude_no_pair=False, hmm_probs=True)

    higher_features = higher_reader.getCoupledFeatures(group_name="efficient",
                                                       success_failure=True,
                                                       mod="skill_personal_perception_action_impact",
                                                       with_control=True)

    lower_features = lower_reader.getCoupledFeatures(group_name="inefficient",
                                                     success_failure=True,
                                                     mod="skill_personal_perception_action_impact",
                                                     with_control=True)

    lower_features["group"] = "lower"
    higher_features["group"] = "higher"
    df = pd.concat([lower_features, higher_features])

    df.loc[:, "lower"] = df.group == "lower"
    df.loc[:, "higher"] = df.group == "higher"

    for prev_feature, next_feature, hitter in zip(PREV_ANALYZED_FEATURES, NEXT_ANALYZED_FEATURES, HITTER_BOOL):

        clean_df = df.dropna(subset=[prev_feature + "_prev", next_feature + "_next"])

        # standarize features
        scaler = StandardScaler()
        all_features = np.concatenate([clean_df[prev_feature + "_prev"].values.reshape(-1, 1), clean_df[next_feature + "_next"].values.reshape(-1, 1)])
        scaler.fit(all_features)
        clean_df[prev_feature + "_prev"] = scaler.transform(clean_df[prev_feature + "_prev"].values.reshape(-1, 1)).flatten()
        clean_df[next_feature + "_next"] = scaler.transform(clean_df[next_feature + "_next"].values.reshape(-1, 1)).flatten()

        print(np.nanmean(clean_df[prev_feature+ "_prev"].values))
        print(np.nanmean(clean_df[next_feature + "_next"].values))

        # check sanity
        # az.plot_dist(clean_df[next_feature + "_next"])
        # plt.show()
        subjects = pd.concat([clean_df["hitter"], clean_df["receiver"]])
        if hitter:
            turn_take_subjects = clean_df["receiver_idx"]

        else:
            turn_take_subjects = clean_df["hitter_idx"]

        actor_subjects_idx, _ = pd.factorize(clean_df["hitter"])
        partner_subjects_idx, _ = pd.factorize(clean_df["receiver"])
        subjects_idx, subjects_unique = pd.factorize(subjects)
        turn_take_unique_idx, turn_take_unique = pd.factorize(turn_take_subjects)

        coords = {"subject_idx": subjects_unique, "turn_take_idx": turn_take_unique,
                  "obs": range(len(clean_df[next_feature + "_next"])),
                  "group": ["lower", "higher"]}

        model = CenteredModel(coords, clean_df, actor_subjects_idx, partner_subjects_idx, turn_take_unique_idx,
                              prev_feature, next_feature, hitter=False)
        with model:
            # debug the model
            print(model.debug())
            #pm.model_to_graphviz(model).view()
            # Inference!
            idata = pm.sample_prior_predictive()
            idata.extend(
                pm.sample(random_seed=100, target_accept=TARGET_ACC, idata_kwargs={"log_likelihood": True},
                          draws=N_SAMPLES,
                          chains=N_CHAINS, tune=N_TUNE, cores=N_CORE, compile_kwargs=dict(mode="NUMBA"))
            )
            idata.extend(pm.sample_posterior_predictive(idata))

        # az.plot_posterior(
        #     idata, var_names=["global_higher_influenceActorPartner_diff", "global_lower_influenceActorPartner_diff",
        #                       "global_influenceActorPartner_diff"], figsize=(15, 10),
        # )
        # plt.show()

        # save the model
        file_name = "idata_"
        with open(DOUBLE_RESULTS_PATH_APM_TTEST + file_name + prev_feature + "_prev_" + next_feature + "_next_" + str(
                N_SAMPLES) + ".pkl", 'wb') as handle:
            print("write data into: " + file_name + prev_feature + "_prev_" + next_feature + "_next_" + str(
                N_SAMPLES) + ".pkl")
            pickle.dump(idata, handle, protocol=pickle.HIGHEST_PROTOCOL)
