import io
import logging
from contextlib import redirect_stderr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydvl.reporting.plots import shaded_mean_std
from pydvl.reporting.scores import compute_removal_score
from pydvl.utils import Utility
from pydvl.utils.config import ParallelConfig
from pydvl.value.least_core import montecarlo_least_core
from pydvl.value.results import ValuationResult
from pydvl.value.shapley import ShapleyMode, compute_shapley_values
from sklearn.naive_bayes import GaussianNB
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import tqdm_logging_redirect

from mlrc22.constants import OUTPUT_DIR, RANDOM_SEED
from mlrc22.utils import create_enron_spam_datasets, set_random_seed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

sns.set_theme(style="whitegrid", palette="pastel")
sns.set_context("paper", font_scale=1.5)

EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "fixing_mislabeled_data"
EXPERIMENT_OUTPUT_DIR.mkdir(exist_ok=True)

mean_colors = ["dodgerblue", "indianred", "limegreen"]
shade_colors = ["lightskyblue", "firebrick", "seagreen"]


def plot_flip_accuracy_over_removal_percentages(
    scores_df: pd.DataFrame, *, label_flip_percentages: list[float]
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
            EXPERIMENT_OUTPUT_DIR
            / f"flip_accuracy_over_removal_percentages_{flip_percentage:.2f}.pdf",
            bbox_inches="tight",
        )


def plot_utility_over_removal_percentages(
    scores_df: pd.DataFrame,
    *,
    scorer_names: list[str],
    label_flip_percentages: list[float],
    method_names: list[str],
    removal_percentages: list[float],
) -> None:
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
                EXPERIMENT_OUTPUT_DIR
                / f"utility_over_removal_percentages_{scorer}_{flip_percentage:.2f}.pdf",
                bbox_inches="tight",
            )


def run():
    parallel_config = ParallelConfig(backend="ray", logging_level=logging.ERROR)

    scorer_names = ["accuracy", "f1", "average_precision"]
    removal_percentages = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    label_flip_percentages = [0.10, 0.20, 0.30]
    method_names = ["Random", "Least Core", "TMC Shapley"]

    n_iterations = 5000
    logger.info(f"Using number of iterations {n_iterations}")

    n_repetitions = 5
    logger.info(f"Using number of repetitions {n_repetitions}")

    n_jobs = 4
    logger.info(f"Using number of jobs {n_jobs}")

    model = GaussianNB()

    random_state = np.random.RandomState(RANDOM_SEED)

    all_scores = []

    with tqdm_logging_redirect():
        for flip_percentage in tqdm(
            label_flip_percentages, desc="Flip Percentage", leave=True
        ):
            logger.info(f"{flip_percentage=}")
            logger.info(f"Creating datasets")
            (
                training_dataset,
                testing_dataset,
                flipped_indices,
            ) = create_enron_spam_datasets(flip_percentage, random_state=random_state)
            logger.info(f"Training dataset size: {len(training_dataset)}")
            logger.info(f"Testing dataset size: {len(testing_dataset)}")

            for scorer_name in tqdm(scorer_names, desc="Scorer", leave=False):
                logger.info(f"{scorer_name=}")
                logger.info("Creating utilities")
                training_utility = Utility(
                    data=training_dataset,
                    model=model,
                    scoring=scorer_name,
                    enable_cache=False,
                )

                testing_utility = Utility(
                    data=testing_dataset,
                    model=model,
                    scoring=scorer_name,
                    enable_cache=False,
                )

                for method_name in tqdm(method_names, desc="Method", leave=False):
                    logger.info(f"{method_name=}")
                    for _ in trange(
                        n_repetitions,
                        desc=f"Repetitions '{method_name}'",
                        leave=False,
                    ):
                        if method_name == "Random":
                            values = ValuationResult.from_random(
                                size=len(training_utility.data)
                            )
                        elif method_name == "Least Core":
                            values = montecarlo_least_core(
                                training_utility,
                                epsilon=0.0,
                                n_iterations=n_iterations,
                                n_jobs=n_jobs,
                                config=parallel_config,
                                options={
                                    "max_iters": 10000,
                                },
                            )
                        else:
                            f = io.StringIO()
                            with redirect_stderr(f):
                                values = compute_shapley_values(
                                    training_utility,
                                    # The budget for TMC Shapley is less because
                                    # for each iteration it goes over all indices
                                    # of an entire permutation of indices
                                    n_iterations=n_iterations
                                    // len(training_utility.data),
                                    n_jobs=n_jobs,
                                    config=parallel_config,
                                    mode=ShapleyMode.TruncatedMontecarlo,
                                )
                        # Sort values in increasing order
                        values.sort()
                        # The data points with flipped labels should have the lowest valuation
                        # So to verify that we compute the percentage of data points
                        # with the lowest valuation that correspond to the flipped data points
                        lowest_value_indices = values.indices[: len(flipped_indices)]
                        assert lowest_value_indices.shape == flipped_indices.shape
                        flip_accuracy = np.in1d(
                            lowest_value_indices, flipped_indices
                        ).mean()

                        scores = compute_removal_score(
                            u=testing_utility,
                            values=values,
                            percentages=removal_percentages,
                            remove_best=False,
                            progress=False,
                        )
                        scores["method"] = method_name
                        scores["scorer"] = scorer_name
                        scores["flip_percentage"] = flip_percentage
                        scores["flip_accuracy"] = flip_accuracy
                        all_scores.append(scores)

    scores_df = pd.DataFrame(all_scores)

    scores_df.to_csv(EXPERIMENT_OUTPUT_DIR / "scores.csv", index=False)

    plot_utility_over_removal_percentages(
        scores_df,
        scorer_names=scorer_names,
        label_flip_percentages=label_flip_percentages,
        method_names=method_names,
    )
    plot_flip_accuracy_over_removal_percentages(
        scores_df, label_flip_percentages=label_flip_percentages
    )


if __name__ == "__main__":
    set_random_seed(RANDOM_SEED)
    run()
