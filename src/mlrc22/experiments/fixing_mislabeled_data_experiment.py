import io
import logging
from contextlib import redirect_stderr

import numpy as np
import pandas as pd
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
from mlrc22.dataset import create_enron_spam_datasets
from mlrc22.plotting import (
    plot_flipped_data_accuracy,
    plot_flipped_utility_over_removal_percentages,
)
from mlrc22.utils import set_random_seed, setup_logger, setup_plotting

logger = setup_logger()

setup_plotting()
set_random_seed(RANDOM_SEED)


def run():
    logger.info("Starting Fixing Mislabeled Data Experiment")

    experiment_output_dir = OUTPUT_DIR / "fixing_mislabeled_data"
    experiment_output_dir.mkdir(exist_ok=True)

    parallel_config = ParallelConfig(backend="ray", logging_level=logging.ERROR)

    scorer_names = ["accuracy", "f1", "average_precision"]
    removal_percentages = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    label_flip_percentages = [0.20, 0.30]
    method_names = ["Random", "Least Core", "TMC Shapley"]

    n_iterations = 5000
    logger.info(f"Using number of iterations {n_iterations}")

    n_repetitions = 5
    logger.info(f"Using number of repetitions {n_repetitions}")

    n_jobs = 4
    logger.info(f"Using number of jobs {n_jobs}")

    model = GaussianNB()

    all_scores = []

    random_state = np.random.RandomState(RANDOM_SEED)

    with tqdm_logging_redirect():
        for flip_percentage in tqdm(
            label_flip_percentages, desc="Flip Percentage", leave=True
        ):
            logger.info(f"{flip_percentage=}")
            for _ in trange(
                n_repetitions,
                desc="Repetitions",
                leave=False,
            ):
                logger.info(f"Creating datasets")
                (
                    training_dataset,
                    testing_dataset,
                    flipped_indices,
                ) = create_enron_spam_datasets(
                    flip_percentage, random_state=random_state
                )
                logger.info(f"Training dataset size: {len(training_dataset)}")
                logger.info(f"Testing dataset size: {len(testing_dataset)}")

                for method_name in tqdm(method_names, desc="Method", leave=False):
                    logger.info(f"{method_name=}")
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
                                    "solver": "SCS",
                                    "max_iters": 30000,
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

    scores_df.to_csv(experiment_output_dir / "scores.csv", index=False)

    plot_flipped_utility_over_removal_percentages(
        scores_df,
        scorer_names=scorer_names,
        label_flip_percentages=label_flip_percentages,
        method_names=method_names,
        removal_percentages=removal_percentages,
        experiment_output_dir=experiment_output_dir,
    )
    plot_flipped_data_accuracy(
        scores_df,
        label_flip_percentages=label_flip_percentages,
        experiment_output_dir=experiment_output_dir,
    )

    logger.info("Finished Fixing Mislabeled Data Experiment")


if __name__ == "__main__":
    run()
