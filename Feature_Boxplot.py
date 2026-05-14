# %% Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from matplotlib import rcParams

# ---------- Global style ----------
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["axes.linewidth"] = 1.2

# %% Functions for boxplot whiskers
def get_boxplot_whisker_top(values):
    values = np.asarray(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    valid_values = values[values <= upper_fence]
    return valid_values.max() if valid_values.size else values.max()


def get_boxplot_whisker_bottom(values):
    values = np.asarray(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    valid_values = values[values >= lower_fence]
    return valid_values.min() if valid_values.size else values.min()


def draw_single_boxplot(ax, rad_data, tan_data, ylabel, show_legend=False, panel_label=None):
    locations = ["C3", "FC3", "C4", "FC4"]

    data_rows = []
    for i, loc in enumerate(locations):
        for val in rad_data[:, i]:
            data_rows.append({"Location": loc, "Orientation": "Radial", "Value": val})
        for val in tan_data[:, i]:
            data_rows.append({"Location": loc, "Orientation": "Tangential", "Value": val})
    df = pd.DataFrame(data_rows)

    colors = {"Radial": "#5DADE2", "Tangential": "#EB984E"}

    sns.boxplot(
        data=df,
        x="Location",
        y="Value",
        hue="Orientation",
        palette=colors,
        width=0.6,
        dodge=True,
        showfliers=False,
        linewidth=1.5,
        boxprops={"edgecolor": "black", "alpha": 0.8},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        medianprops={"linewidth": 2, "color": "black"},
        ax=ax,
    )

    visible_bottoms = []
    visible_tops = []

    for i, _loc in enumerate(locations):
        _, p_val = ttest_rel(rad_data[:, i], tan_data[:, i])

        rad_whisker_bottom = get_boxplot_whisker_bottom(rad_data[:, i])
        tan_whisker_bottom = get_boxplot_whisker_bottom(tan_data[:, i])
        rad_whisker_top = get_boxplot_whisker_top(rad_data[:, i])
        tan_whisker_top = get_boxplot_whisker_top(tan_data[:, i])
        whisker_bottom = min(rad_whisker_bottom, tan_whisker_bottom)
        whisker_top = max(rad_whisker_top, tan_whisker_top)

        visible_bottoms.append(whisker_bottom)
        visible_tops.append(whisker_top)

    visible_y_min = min(visible_bottoms)
    visible_y_max = max(visible_tops)
    visible_y_range = visible_y_max - visible_y_min
    if visible_y_range == 0:
        visible_y_range = max(abs(visible_y_max), 1.0)

    h_offset = visible_y_range * 0.045
    top_annotation = visible_y_max

    for i, _loc in enumerate(locations):
        _, p_val = ttest_rel(rad_data[:, i], tan_data[:, i])

        rad_whisker_top = get_boxplot_whisker_top(rad_data[:, i])
        tan_whisker_top = get_boxplot_whisker_top(tan_data[:, i])
        whisker_top = max(rad_whisker_top, tan_whisker_top)

        bracket_top = whisker_top + h_offset

        ax.plot(
            [i - 0.2, i - 0.2, i + 0.2, i + 0.2],
            [bracket_top, bracket_top + 0.4 * h_offset, bracket_top + 0.4 * h_offset, bracket_top],
            lw=1.2,
            c="k",
        )

        marker = "***" if p_val < 0.001 else f"p={p_val:.3f}"
        ax.text(
            i,
            bracket_top + 0.5 * h_offset,
            marker,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
        top_annotation = max(top_annotation, bracket_top + 0.8 * h_offset)

    lower_margin = 0.03 * visible_y_range
    upper_margin = 0.04 * visible_y_range
    ax.set_ylim(visible_y_min - lower_margin, top_annotation + upper_margin)

    ax.set_ylabel(ylabel, fontsize=20)
    ax.set(xlabel=None)

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=18, length=4, width=1.2, direction="out")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)

    if not show_legend:
        ax.get_legend().remove()
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=18, frameon=False)

    if panel_label is not None:
        ax.text(
            -0.18,
            1.05,
            panel_label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="left",
        )

def plot_all_features():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    std_res = np.load("Std_Result.npz")
    draw_single_boxplot(
        axes[0],
        std_res["rad_std"],
        std_res["tan_std"],
        ylabel="Standard Deviation (fT)",
        panel_label="(a)",
    )

    skew_res = np.load("skewness_result.npz")
    draw_single_boxplot(
        axes[1],
        skew_res["skewness_rad"],
        skew_res["skewness_tan"],
        ylabel="Skewness",
        show_legend=True,
        panel_label="(b)",
    )

    kurt_res = np.load("kurtosis_result.npz")
    draw_single_boxplot(
        axes[2],
        kurt_res["kurtosis_rad"],
        kurt_res["kurtosis_tan"],
        ylabel="Kurtosis",
        panel_label="(c)",
    )

    ent_res = np.load("sampen_results.npz")
    draw_single_boxplot(
        axes[3],
        ent_res["rad_sampen"],
        ent_res["tan_sampen"],
        ylabel="SampEn",
        panel_label="(d)",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    plt.savefig("Features_Boxplot_nopoint.pdf", dpi=600, bbox_inches="tight")
    plt.show()

# %% plot features
plot_all_features()

# %%
