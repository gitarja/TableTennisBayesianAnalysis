import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from Utils.Conf import features_explanation
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

sns.set_theme()
sns.set(font_scale=5)
sns.set(font="Arial")
sns.set_style("white")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 30


def plotSHAP(shap_values, x, columns, results_path="", prefix="", alpha=0.15):
    # plt.rcParams["font.family"] = "Arial"
    # plt.rcParams["font.size"] = 2
    # combing shap and x
    shap_columns = ["shap_" + c for c in columns]
    summary_df = pd.DataFrame(np.concatenate([x.values, shap_values], axis=-1), columns=columns + shap_columns)
    explanation = shap.Explanation(values=shap_values, data=x.values, feature_names=columns)
    polynomial_features = PolynomialFeatures(degree=2)
    for c in columns:

        try:
            ref_value = explanation[:, c]

            xmin = np.nanpercentile(ref_value.data, 0.5)
            xmax = np.nanpercentile(ref_value.data, 99.5)
            ymin = np.min(ref_value.values)
            ymax = np.max(ref_value.values)
            y_abs_max = np.max(np.abs(ref_value.values))

            if c == "receiver_skill":
                xmax = 1
            elif c == "receiver_p3_fx_onset":
                xmin = 0

            shap.plots.scatter(ref_value, show=False, alpha=alpha, xmin=xmin, xmax=xmax, dot_size=10)
            snipset_summary = summary_df.loc[(summary_df[c] >= xmin) & (summary_df[c] <= xmax)]
            if (c != "receiver_p1_al") & (c != "receiver_p2_al") & (c != "receiver_p3_fx") & (c != "hitter_p1_al") & (
                    c != "hitter_p2_al") & (c != "hitter_fx") & (c != "hitter_p1_cs") & (c != "hitter_p2_cs") & (
                    c != "receiver_p1_cs") & (c != "receiver_p2_cs"):
                a = sns.regplot(data=snipset_summary, x=c, y="shap_" + c, order=2, color="#81b1d3",
                                line_kws=dict(color="#252525"), scatter=False)
                # xp = polynomial_features.fit_transform(np.expand_dims(snipset_summary[c].values, -1))
                # model = sm.OLS(np.expand_dims(snipset_summary["shap_" + c].values, -1), xp).fit()
                # print(model.summary())

                # fig, ax = plt.subplots()

                # ax.set(xlim=(xmin, xmax))
            plt.ylabel(r'Influence on'  "\n" 'success of coordination')
            plt.xlabel("\n" + features_explanation[c])
            plt.axhline(y=0., color="#525252", linestyle=":")
            if c == "receiver_skill":
                plt.xlim(0.5, 1.01)
            plt.ylim(-1 * y_abs_max, y_abs_max)

            sns.despine()
            plt.tight_layout()
            # plt.show()

            # plt.savefig(results_path + "\\" + c +"_"+ prefix + ".png", format='png')
            plt.savefig(results_path + "\\" + c + "_" + prefix + ".pdf", format='pdf')
            plt.close()
        except:
            print(c)


def plotShapSummary(shap_values, x, results_path="", prefix=""):
    shap.summary_plot(shap_values, x, max_display=40, show=False)
    plt.tight_layout()
    plt.savefig(results_path + "\\" + "summary_shap.png", format='png')
    plt.close()


def plotShapAbsoulte(shap_values, x, y=None, columns=None, results_path="", prefix="", max=12):
    explanation = shap.Explanation(values=shap_values, data=x.values, feature_names=columns)
    if y is not None:
        clustering = shap.utils.hclust(x, y)
        shap.plots.bar(explanation, max_display=max, show=False, clustering=clustering, clustering_cutoff=0.5)
    else:
        shap.plots.bar(explanation, max_display=max, show=False)
    plt.tight_layout()
    # plt.savefig(results_path + "\\" + "bar_shap.png", format='png')
    plt.savefig(results_path + "\\" + "bar_shap.pdf", format='pdf')
    plt.close()


def plotShapInteraction(shap_values, x, columns, results_path, ref, show_column):
    sns.set_theme()
    sns.set_style("white")
    # combing shap and x
    shap_columns = ["shap_" + c for c in columns]
    summary_df = pd.DataFrame(np.concatenate([x.values, shap_values], axis=-1), columns=columns + shap_columns)
    explanation = shap.Explanation(values=shap_values, data=x.values, feature_names=columns)

    for c in show_column:
        ref_value = explanation[:, ref]
        shap.plots.scatter(ref_value, color=explanation[:, c],
                           xmin=np.nanpercentile(ref_value.data, 0.5), xmax=np.nanpercentile(ref_value.data, 99.5),

                           alpha=0.2, show=False)
        plt.ylabel('SHAP values')
        plt.axhline(y=0., color="black", linestyle=":")

        plt.savefig(results_path + "\\" + ref + "_" + c + ".png", format='png')
        plt.close()


def plotShapComparisonBar(features_name, shap_order1, shap_order2, shap_order3):
    def combinedDF(df):
        features = df["features"]
        features_list = []
        group_list = []
        ineff_df = df["ineff"]
        avg_df = df["avg"]
        eff_df = df["eff"]
        group_list.extend("ineff" for _ in range(len(df)))
        group_list.extend("avg" for _ in range(len(df)))
        group_list.extend("eff" for _ in range(len(df)))
        features_list.extend(features for _ in range(3))

        new_df = pd.DataFrame(
            {"features": np.concatenate(features_list), "SHAP": np.concatenate([ineff_df, avg_df, eff_df]),
             "group": group_list})

        return new_df

    ineff_shap_abs = []
    avg_shap_abs = []
    eff_shap_abs = []
    all_shap_abs = []

    for i in range(len(features_name)):
        shap_ineff = np.average(np.abs((shap_order1[:, i])))
        shap_avg = np.average(np.abs((shap_order2[:, i])))
        shap_eff = np.average(np.abs((shap_order3[:, i])))

        shap_all = np.average(np.abs(np.concatenate([shap_order1[:, i], shap_order2[:, i], shap_order3[:, i]])))

        ineff_shap_abs.append(shap_ineff)
        avg_shap_abs.append(shap_avg)
        eff_shap_abs.append(shap_eff)
        all_shap_abs.append(shap_all)

    df = pd.DataFrame(
        {"features": features_name, "inefficient": ineff_shap_abs, "average": avg_shap_abs, "efficient": eff_shap_abs,
         "all": all_shap_abs})
    # df = df[df["features"].str.contains("_im_") == False] # remove im
    df = df.sort_values(by=['all'], ascending=False)

    df_top10 = df.iloc[:15]
    baseline = np.median(np.average([df["inefficient"], df["average"], df["efficient"]]))
    colors = ["#bdc9e1", "#74a9cf", "#045a8d"]
    # Set your custom color palette
    pallete = sns.color_palette(colors)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    sns.barplot(ax=axes[0], x=df_top10.inefficient, y=df_top10.features, color=colors[0])
    axes[0].axvline(baseline, ls='--', color="#969696")
    sns.barplot(ax=axes[1], x=df_top10.average, y=df_top10.features, color=colors[1])
    axes[1].axvline(baseline, ls='--', color="#969696")
    sns.barplot(ax=axes[2], x=df_top10.efficient, y=df_top10.features, color=colors[2])
    axes[2].axvline(baseline, ls='--', color="#969696")
    # df_top10.plot(x='features', kind='barh', color=pallete)
    plt.show()


def plotShapComparison(features_name, shap_order1, shap_order2, shap_order_diff, color, size, label2="EFFICIENT"):
    GREY96 = "0.96"
    GREY30 = "0.3"
    GREY15 = "0.15"
    GREY05 = "#bdbdbd"
    # Initialize plot
    fig, ax = plt.subplots(figsize=(9, 11))

    # Adjust figure margins, this is going to be useful later.
    fig.subplots_adjust(left=0.05, right=0.90, top=0.9, bottom=0.075)

    # Set background color

    fig.set_facecolor(GREY96)
    ax.set_facecolor(GREY96)

    for y0, y1, c, s1, df in zip(shap_order1, shap_order2, color, size * 200, shap_order_diff):
        if abs(df) == 0:
            c = GREY05
        ax.plot([1, 2], [y0, y1], c=c, lw=1)
        if df < 0:
            ax.scatter(1, y0, c=c, s=s1, zorder=10)
            ax.scatter(2, y1, c=c, s=1, zorder=10)
        else:
            ax.scatter(1, y0, c=c, s=1, zorder=10)
            ax.scatter(2, y1, c=c, s=s1, zorder=10)

    # Space between the dot and the label
    TEXT_HPADDING = 0.08

    # Space between the line and the dot
    LINE_HPADDING1 = 0.02
    # Space between the line and the label
    LINE_HPADDING2 = 0.07

    for i, t in enumerate(zip(shap_order_diff, features_name)):
        # Take the vertical adjustment for the name

        od, name = t
        # If it is odd, plot on the left
        if od < 0:
            # Add label
            x = 1 - TEXT_HPADDING
            y = shap_order1[i]
            ax.text(
                x, y, name, size=11,
                color=GREY15, ha="right", va="center"
            )

            # Add line connect dot with label
            x = [1 - LINE_HPADDING2, 1 - LINE_HPADDING1]
            y = [shap_order1[i], shap_order1[i]]
            ax.plot(x, y, color=GREY15, lw=0.5)

        # If it is even, plot on the right
        else:
            # Add label
            x = 2 + TEXT_HPADDING
            y = shap_order2[i]
            ax.text(
                x, y, name, size=11,
                color=GREY15, va="center"
            )
            # Add line connect dot with label
            x = [2 + LINE_HPADDING1, 2 + LINE_HPADDING2]
            y = [shap_order2[i], shap_order2[i]]
            ax.plot(x, y, color=GREY15, lw=0.5)

    # Remove all spines
    ax.set_frame_on(False)
    # Remove horizontal and vertical ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Expand horizontal limits to (0.5, 2.5)
    ax.set_xlim(0.5, 2.5)

    # Horizontal lines in the background
    # ax.hlines(np.arange(10, 41), 1.1, 1.9, alpha=0.2, lw=0.5, color=GREY15, zorder=0)

    # Add annotations
    # Note the dots added with ax.scatter are the text backgrounds
    for y in np.arange(0, 41, 5):
        ax.scatter(1.5, y, s=1200, color=GREY96)
        ax.text(
            1.5, y, str(y),
            size=10, name="Faune", color="darkgreen", weight="bold",
            alpha=0.2, va="center", ha="center"
        )

    ax.text(
        0.9, 41, "AVG",
        name="Faune", size=27, color="darkgreen",
        weight="bold", alpha=0.3, ha="right"
    )

    ax.text(
        2.1, 41, label2, name="Faune", size=27, color="darkgreen",
        weight="bold", alpha=0.3, ha="left"
    )
    plt.show()
