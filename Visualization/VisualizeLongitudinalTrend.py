from Utils.Conf import DOUBLE_RESULTS_PATH_LONGITUDINAL, features_explanation
import pickle
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import numpy as np
from scipy.special import expit
import seaborn as sns
from matplotlib.lines import Line2D
# sns.set_style("white")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams.update({'xtick.labelsize': 25, 'ytick.labelsize': 25})
# save the model
n = 5
features = [
    # "receiver_start_fs",
    # "hitter_p1_al_mag",
    # "receiver_im_racket_dir",
    # "receiver_fixation_racket_latency",
    # "hitter_p2_al_prec",
    # "hitter_p2_al_mag",
    # "receiver_p2_al_mag",
    # "hitter_at_and_after_hit",
    # "hitter_p1_cs",
    # "hitter_p2_al_onset",
    # "hand_movement_sim",
    # "receiver_p1_al_onset",
    # "hitter_p1_al_prec",
    # "receiver_p1_al_mag",
    # "receiver_p2_al_prec",
    # "receiver_p2_al_onset",
    # "hitter_fx_onset",
    # "receiver_distance_eye_hand",
    # "hitter_p1_al_onset",
    # "receiver_p1_al_prec",
    # "hitter_fx_duration",
    # "receiver_p3_fx_onset",
    "receiver_im_ball_updown",
    # "receiver_p3_fx_duration"

]

# features = ["receiver_fixation_racket_latency"]
for analyzed_features in features:
    print(analyzed_features)


    with open(DOUBLE_RESULTS_PATH_LONGITUDINAL + "idata_" + analyzed_features + "_" + str(n) + ".pkl", 'rb') as handle:
        idata = pickle.load(handle)
    # hierarchical_loo = az.plot_ppc(idata, kind="cumulative")
    # plt.show()
    # compared posterior to observed
    # az.plot_trace(idata, var_names=[     "global_higher", "global_lower",
    #         "global_higher_seg", "global_lower_seg"])
    # plt.show()

    print(az.summary(
        idata,
        var_names=[
            "global_higher", "global_lower",
            "global_higher_seg", "global_lower_seg",
            "global_th_segment",
            "global_skill_slope"
        ],
    ))


    # global_higher = posterior["global_higher"].mean(dim="sample")
    # global_higher_seg = posterior["global_higher_seg"].mean(dim="sample")
    #
    # global_lower = posterior["global_lower"].mean(dim="sample")
    # global_lower_seg = posterior["global_lower_seg"].mean(dim="sample")
    #
    # global_skill_slope = posterior["global_skill_slope"].mean(dim="sample")

    posterior = az.extract(idata.posterior)
    global_higher = posterior["global_higher"]
    global_higher_seg = posterior["global_higher_seg"]

    global_lower = posterior["global_lower"]
    global_lower_seg = posterior["global_lower_seg"]

    global_skill_slope = posterior["global_skill_slope"]

    subjects_intercept = posterior["subjects_intercept"].mean(dim="subject_idx")
    subjects_intercept_seg = posterior["subjects_intercept_seg"].mean(dim="subject_idx")

    global_intercept = posterior["global_intercept"] + subjects_intercept
    global_th_segment = posterior["global_th_segment"] + subjects_intercept_seg

    time = 100
    time_xi = xr.DataArray(np.arange(time) / 100)
    # plot line
    fig, ax = plt.subplots(figsize=(12, 8))

    y_higher = global_intercept + global_higher * 1 + global_higher_seg * (
        time_xi) + global_th_segment * time_xi + global_skill_slope
    y_higher_mean = global_intercept.mean() + global_higher.mean() * 1 + global_higher_seg.mean() * (
        time_xi) + global_th_segment.mean() * time_xi + global_skill_slope.mean()

    y_lower = global_intercept + global_lower * 1 + global_lower_seg * (
        time_xi) + global_th_segment * time_xi + global_skill_slope
    y_lower_mean = global_intercept.mean() + global_lower.mean() * 1 + global_lower_seg.mean() * (
        time_xi) + global_th_segment.mean() * time_xi + global_skill_slope.mean()

    # ax.plot(
    #     time_xi,
    #     y_higher.values.reshape(8000, n).T,
    #     color="#66c2a5",
    #     linewidth=0.05,
    #     alpha=0.05,
    # )
    #
    # ax.plot(
    #     time_xi,
    #     y_lower.values.reshape(8000, n).T,
    #     color="#fb8072",
    #     linewidth=0.05,
    #     alpha=0.05,
    # )


    az.plot_hdi(
        time_xi,
        y_higher.values.reshape(4, 2000, time),
        hdi_prob=0.95,
        fill_kwargs={"alpha": 0.1, "linewidth": 0.1},
        color="#69A87F",
    )

    az.plot_hdi(
        time_xi,
        y_lower.values.reshape(4, 2000, time),
        hdi_prob=0.95,
        fill_kwargs={"alpha": 0.1, "linewidth": 0.1},
        color="#B5152C",
    )

    ax.plot(
        time_xi,
        y_higher_mean,
        color="#69A87F",
        lw=3,
        linestyle='dashed'
    )

    ax.plot(
        time_xi,
        y_lower_mean,
        color="#B5152C",
        lw=3,
        linestyle='dashed'
    )
    ax.set_ylabel(features_explanation[analyzed_features], fontsize=28)
    ax.set_xlabel(r"T of episode / 100", fontsize=28)
    ax.set_frame_on(False)
    ax.set_ylim([0.19, 0.25])

    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
    ax.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # plt.show()
    plt.savefig(DOUBLE_RESULTS_PATH_LONGITUDINAL + "trends\\"+analyzed_features+".pdf", format='pdf', transparent=True, bbox_inches='tight')
    plt.close()

