import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

def plotSHAP(shap_values, x, columns, results_path="", prefix=""):
    sns.set_theme()
    sns.set(font_scale=1.7)
    sns.set_style("white")
    # combing shap and x
    shap_columns = ["shap_"+c for c in columns]
    summary_df = pd.DataFrame(np.concatenate([x.values, shap_values], axis=-1), columns=columns + shap_columns)
    explanation = shap.Explanation(values=shap_values, data=x.values, feature_names=columns)
    for c in columns:
        ref_value = explanation[:, c]
        if c =="receiver_skill":
            xmax = 1.
        else:
            xmax = np.nanpercentile(ref_value.data, 99)

        shap.plots.scatter(ref_value, show=False,  alpha=0.3, xmin=np.nanpercentile(ref_value.data, 1), xmax=xmax)
        # g = sns.scatterplot(x=c, y="shap_" + c, data=summary_df, s=7, color="#636363", edgecolor = None, alpha=0.5)
        #
        plt.ylabel('SHAP values')
        plt.axhline(y=0., color="black", linestyle=":")
        if c=="receiver_skill":
            plt.xlim(0.5, 1.1)

        # sns.despine()
        # plt.show()
        plt.tight_layout()
        plt.savefig(results_path + "\\" + c +"_"+ prefix + ".png", format='png')
        plt.close()


def plotShapSummary(shap_values, x, results_path="", prefix=""):
    shap.summary_plot(shap_values, x, max_display=40, show=False)
    plt.tight_layout()
    plt.savefig(results_path + "\\" + "summary_shap.png", format='png')
    plt.close()

def plotShapInteraction(shap_values, x, columns, results_path, ref, show_column):
    sns.set_theme()
    sns.set_style("white")
    # combing shap and x
    shap_columns = ["shap_"+c for c in columns]
    summary_df = pd.DataFrame(np.concatenate([x.values, shap_values], axis=-1), columns=columns + shap_columns)
    explanation = shap.Explanation(values=shap_values, data=x.values, feature_names=columns)

    for c in show_column:
        ref_value = explanation[:, ref]
        shap.plots.scatter(ref_value, color=explanation[:, c],
                           xmin=np.nanpercentile(ref_value.data, 0.5), xmax=np.nanpercentile(ref_value.data, 99.5),

                           alpha=0.2, show=False)
        plt.ylabel('SHAP values')
        plt.axhline(y=0., color="black", linestyle=":")

        plt.savefig(results_path + "\\" + ref + "_" + c +  ".png", format='png')
        plt.close()



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