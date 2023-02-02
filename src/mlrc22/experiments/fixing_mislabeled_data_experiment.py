import io
import logging
from contextlib import redirect_stderr
from time import time

import numpy as np
import pandas as pd
import pydvl
from pydvl.reporting.scores import compute_removal_score
from pydvl.utils import Utility, init_parallel_backend
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
    plot_values_histogram,
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

    n_repetitions = 10
    logger.info(f"Using number of repetitions {n_repetitions}")

    n_jobs = 56
    logger.info(f"Using number of jobs {n_jobs}")

    model = GaussianNB()

    all_values_df = None

    random_state = np.random.RandomState(RANDOM_SEED)
    all_scores = []
    delayed_values = []
    parallel_backend = init_parallel_backend(parallel_config)

    def _fun(u, testing_utility, data, method_name, scorer_name, flip_percentage):
        if method_name in ("Random", "TMC Shapley"):
            values = data
        else:
            values = pydvl.value.least_core.montecarlo._solve_linear_problem(u, data)
        return dict(
            values=values,
            method_name=method_name,
            scorer_name=scorer_name,
            flip_percentage=flip_percentage,
            testing_utility=testing_utility,
        )

    fun = parallel_backend.wrap(_fun)

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
                            show_warnings=False
                        )

                        testing_utility = Utility(
                            data=testing_dataset,
                            model=model,
                            scoring=scorer_name,
                            enable_cache=False,
                            show_warnings=False
                        )

                        if method_name == "Random":
                            values = ValuationResult.from_random(
                                size=len(training_utility.data)
                            )
                        elif method_name == "Least Core":
                            values = montecarlo_least_core(
                                training_utility,
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
                        delayed_values.append(
                            dict(
                                u=training_utility,
                                testing_utility=testing_utility,
                                data=values,
                                method_name=method_name,
                                scorer_name=scorer_name,
                                flip_percentage=flip_percentage,
                            )
                        )

    logger.info(f"Waiting for {len(delayed_values)} promised values")
    start = time()
    resolved_values = parallel_backend.get(
            [fun.remote(**p) for p in delayed_values]
            )
    logger.info(
            f"Resolved {len(resolved_values)} promised values in "
            f"{time() - start:.2f} seconds"
            )

    for result in resolved_values:

        values = result["values"]
        method_name = result["method_name"]
        scorer_name = result["scorer_name"]
        flip_percentage = result["flip_percentage"]
        testing_utility = result["testing_utility"]

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

        # Save raw values
        column_name = f"{method_name}_{flip_percentage}"
        df = (
            values.to_dataframe(column=column_name)
            .drop(columns=[f"{column_name}_stderr"])
            .T
        )
        df = df[sorted(df.columns)]
        df["method"] = method_name
        df["scorer"] = scorer_name
        df["flip_percentage"] = flip_percentage

        if all_values_df is None:
            all_values_df = df.copy()
        else:
            all_values_df = pd.concat([all_values_df, df])

    scores_df = pd.DataFrame(all_scores)

    scores_df.to_csv(experiment_output_dir / "scores.csv", index=False)

    all_values_df.to_csv(experiment_output_dir / "values.csv", index=False)

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

    plot_values_histogram(
        all_values_df,
        hue_column="flip_percentage",
        method_names=method_names,
        experiment_output_dir=experiment_output_dir,
    )

    logger.info("Finished Fixing Mislabeled Data Experiment")


if __name__ == "__main__":
    run()
