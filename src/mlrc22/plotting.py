from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from pydvl.reporting.plots import shaded_mean_std

__all__ = [
    "plot_constraint_accuracy_over_coalitions",
    "plot_flip_accuracy_over_removal_percentages",
    "plot_flipped_utility_over_removal_percentages",
    "plot_clean_data_utility_percentage",
    "plot_clean_data_vs_noisy_data_utility",
    "plot_utility_over_removal_percentages",
]


def shaded_mean_confidence_interval(
    data: pd.DataFrame,
    abscissa: Sequence[Any] | None = None,
    alpha: float = 0.05,
    mean_color: str | None = "dodgerblue",
    shade_color: str | None = "lightblue",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Modified version of the `shaded_mean_std()` function defined in pyDVL."""
    assert len(data.shape) == 2
    mean = data.mean(axis=0)
    quantiles = data.quantile([alpha, 1 - alpha], axis=0)

    if ax is None:
        fig, ax = plt.subplots()
    if abscissa is None:
        abscissa = list(range(data.shape[1]))

    ax.fill_between(
        abscissa,
        quantiles.loc[alpha],
        quantiles.loc[1 - alpha],
        alpha=0.3,
        color=shade_color,
    )
    ax.plot(abscissa, mean, color=mean_color, **kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def plot_constraint_accuracy_over_coalitions(
    accuracies_df: pd.DataFrame,
    *,
    scorer_names: list[str],
    experiment_output_dir: Path,
    method_name: str,
) -> None:
    for scorer in scorer_names:
        df = accuracies_df[accuracies_df["scorer"] == scorer]
        fig, ax = plt.subplots()
        sns.barplot(
            data=df,
            x="fraction",
            y="accuracy",
            hue="dataset",
            palette={
                "House": "indianred",
                "Medical": "darkorchid",
                "Chemical": "dodgerblue",
            },
            ax=ax,
        )
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=3,
            title=None,
            frameon=False,
        )
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel("Fraction of Samples")
        ax.set_ylabel("Accuracy")
        fig.tight_layout()
        fig.savefig(
            experiment_output_dir
            / f"{method_name}_accuracy_over_coalitions_{scorer}.pdf",
            bbox_inches="tight",
        )


def plot_clean_data_utility_percentage(
    results_df: pd.DataFrame,
    *,
    noise_fractions: list[float],
    noise_levels: list[float],
    experiment_output_dir: Path,
) -> None:
    mean_colors = ["limegreen", "indianred", "dodgerblue"]
    shade_colors = ["seagreen", "firebrick", "lightskyblue"]

    fig, ax = plt.subplots()

    for i, noise_fraction in enumerate(noise_fractions):
        df = results_df[(results_df["noise_fraction"] == noise_fraction)]
        df = (
            df.groupby("noise_level")["clean_values_percentage"]
            .apply(lambda df: df.reset_index(drop=True))
            .unstack()
        )
        shaded_mean_std(
            df,
            abscissa=noise_levels,
            mean_color=mean_colors[i],
            shade_color=shade_colors[i],
            xlabel="Noise Level",
            ylabel="Percentage of the Total Utility",
            label=f"Noise Fraction: {noise_fraction:.1f}",
            ax=ax,
        )

    plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=2
    )
    fig.tight_layout()
    fig.savefig(
        experiment_output_dir / f"clean_data_utility_percentage.pdf",
        bbox_inches="tight",
    )


def plot_clean_data_vs_noisy_data_utility(
    results_df: pd.DataFrame,
    *,
    noise_fractions: list[float],
    noise_levels: list[float],
    experiment_output_dir: Path,
) -> None:
    mean_colors = ["limegreen", "indianred", "dodgerblue"]
    shade_colors = ["seagreen", "firebrick", "lightskyblue"]

    for noise_fraction in noise_fractions:
        df = results_df[(results_df["noise_fraction"] == noise_fraction)]
        df = (
            df.groupby("noise_level")[
                ["total_clean_utility", "total_noisy_utility", "total_utility"]
            ]
            .apply(lambda df: df.reset_index(drop=True))
            .unstack()
        )
        fig, ax = plt.subplots()
        for i, (column, ylabel) in enumerate(
            zip(
                ["total_clean_utility", "total_noisy_utility", "total_utility"],
                ["Clean Data", "Noisy Data", "Total Utility"],
            )
        ):
            shaded_mean_std(
                df[[column]],
                abscissa=noise_levels,
                mean_color=mean_colors[i],
                shade_color=shade_colors[i],
                xlabel="Noise Level",
                ylabel="Utility",
                label=ylabel,
                ax=ax,
            )
        plt.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=3
        )
        fig.tight_layout()
        fig.savefig(
            experiment_output_dir
            / f"clean_data_vs_noisy_data_utility_{noise_fraction:.2f}.pdf",
            bbox_inches="tight",
        )


def plot_flip_accuracy_over_removal_percentages(
    scores_df: pd.DataFrame,
    *,
    label_flip_percentages: list[float],
    experiment_output_dir: Path,
) -> None:
    for flip_percentage in label_flip_percentages:
        fig, ax = plt.subplots()
        sns.boxplot(
            data=scores_df,
            x="method",
            y="flip_accuracy",
            hue="scorer",
            palette={
                "accuracy": "indianred",
                "average_precision": "darkorchid",
                "f1": "dodgerblue",
            },
            ax=ax,
        )
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=3,
            title=None,
            frameon=False,
        )
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel("Method")
        ax.set_ylabel("Flipped Data Points Accuracy")
        fig.tight_layout()
        fig.savefig(
            experiment_output_dir
            / f"flip_accuracy_over_removal_percentages_{flip_percentage:.2f}.pdf",
            bbox_inches="tight",
        )


def plot_flipped_utility_over_removal_percentages(
    scores_df: pd.DataFrame,
    *,
    scorer_names: list[str],
    label_flip_percentages: list[float],
    method_names: list[str],
    removal_percentages: list[float],
    experiment_output_dir: Path,
) -> None:
    mean_colors = ["darkorchid", "limegreen", "dodgerblue"]
    shade_colors = ["plum", "seagreen", "lightskyblue"]

    for scorer in scorer_names:
        if scorer == "accuracy":
            ylabel = "Accuracy"
        elif scorer == "f1":
            ylabel = "F1 Score"
        else:
            ylabel = "Average Precision"

        for flip_percentage in label_flip_percentages:
            fig, ax = plt.subplots()

            for i, method_name in enumerate(method_names):
                df = scores_df[
                    (scores_df["method"] == method_name)
                    & (scores_df["scorer"] == scorer)
                    & (scores_df["flip_percentage"] == flip_percentage)
                ].drop(columns=["method", "scorer", "flip_percentage", "flip_accuracy"])
                shaded_mean_std(
                    df,
                    abscissa=removal_percentages,
                    mean_color=mean_colors[i],
                    shade_color=shade_colors[i],
                    xlabel="Percentage Removal",
                    ylabel=ylabel,
                    label=f"{method_name}",
                    ax=ax,
                )
            plt.legend(loc="lower left")
            fig.tight_layout()
            fig.savefig(
                experiment_output_dir
                / f"utility_over_removal_percentages_{scorer}_{flip_percentage:.2f}.pdf",
                bbox_inches="tight",
            )


def plot_utility_over_removal_percentages(
    scores_df: pd.DataFrame,
    *,
    method_names: list[str],
    budget_list: list[int],
    removal_percentages: list[float],
    experiment_output_dir: Path,
) -> None:
    mean_colors = ["dodgerblue", "darkorange", "limegreen", "indianred", "darkorchid"]
    shade_colors = ["lightskyblue", "gold", "seagreen", "firebrick", "plum"]

    for type in ["best", "worst"]:
        for budget in budget_list:
            fig, ax = plt.subplots()

            for i, method_name in enumerate(method_names):
                df = scores_df[
                    (scores_df["method"] == method_name)
                    & (scores_df["budget"] == budget)
                    & (scores_df["type"] == type)
                ].drop(columns=["method", "budget", "type"])

                shaded_mean_confidence_interval(
                    df,
                    abscissa=removal_percentages,
                    mean_color=mean_colors[i],
                    shade_color=shade_colors[i],
                    xlabel="Percentage Removal",
                    ylabel="Accuracy",
                    label=f"{method_name}",
                    ax=ax,
                )
            plt.legend(loc="lower left")
            fig.tight_layout()
            fig.savefig(
                experiment_output_dir
                / f"utility_over_removal_percentages_{type}_{budget}.pdf",
                bbox_inches="tight",
            )
