import io
import logging
from contextlib import redirect_stderr

import numpy as np
import pandas as pd
from pydvl.utils import Utility
from pydvl.utils.config import ParallelConfig
from pydvl.value.least_core import montecarlo_least_core
from pydvl.value.results import ValuationResult
from pydvl.value.shapley import ShapleyMode, compute_shapley_values
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import tqdm_logging_redirect

from mlrc22.constants import OUTPUT_DIR, RANDOM_SEED
from mlrc22.dataset import create_synthetic_dataset
from mlrc22.plotting import (
    plot_clean_data_utility_percentage,
    plot_clean_data_vs_noisy_data_utility,
    plot_noisy_data_accuracy,
)
from mlrc22.utils import set_random_seed, setup_logger, setup_plotting

logger = setup_logger()

setup_plotting()
set_random_seed(RANDOM_SEED)


def run():
    logger.info("Starting Noisy Data Experiment")

    experiment_output_dir = OUTPUT_DIR / "noisy_data"
    experiment_output_dir.mkdir(exist_ok=True)

    parallel_config = ParallelConfig(backend="ray", logging_level=logging.ERROR)

    n_features = 50
    n_train_samples = 200
    n_test_samples = 5000

    noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0]
    noise_fraction = 0.2
    logger.info(f"{noise_fraction=}")

    method_names = ["Random", "Least Core", "TMC Shapley"]

    n_iterations = 5000
    logger.info(f"Using number of iterations {n_iterations}")

    n_repetitions = 5
    logger.info(f"Using number of repetitions {n_repetitions}")

    n_jobs = 8
    logger.info(f"Using number of jobs {n_jobs}")

    random_state = np.random.RandomState(RANDOM_SEED)

    all_values_df = None

    all_results = []

    with tqdm_logging_redirect():
        for _ in trange(n_repetitions, desc="Repetitions", leave=True):
            for noise_level in tqdm(noise_levels, desc="Noise Level", leave=False):
                logger.info(f"{noise_level=}")
                dataset, noisy_indices = create_synthetic_dataset(
                    n_features=n_features,
                    n_train_samples=n_train_samples,
                    n_test_samples=n_test_samples,
                    random_state=random_state,
                    noise_level=noise_level,
                    noise_fraction=noise_fraction,
                )
                logger.info(f"Number of samples in dataset: {len(dataset)}")

                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(solver="liblinear"),
                )
                logger.info(f"Creating utility")
                utility = Utility(
                    data=dataset,
                    model=model,
                    enable_cache=False,
                )

                for method_name in tqdm(method_names, desc="Method", leave=False):
                    logger.info(f"{method_name=}")
                    if method_name == "Random":
                        values = ValuationResult.from_random(size=len(utility.data))
                    elif method_name == "Least Core":
                        values = montecarlo_least_core(
                            utility,
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
                                utility,
                                # The budget for TMC Shapley is less because
                                # for each iteration it goes over all indices
                                # of an entire permutation of indices
                                n_iterations=n_iterations // len(utility.data),
                                n_jobs=n_jobs,
                                config=parallel_config,
                                mode=ShapleyMode.TruncatedMontecarlo,
                            )

                    # Sort values in increasing order
                    values.sort()
                    # The noisy data points should have the lowest valuation
                    # So to verify that we compute the percentage of data points
                    # with the lowest valuation that correspond to the noisy data points
                    lowest_value_indices = values.indices[: len(noisy_indices)]
                    assert lowest_value_indices.shape == noisy_indices.shape
                    noisy_indices_accuracy = np.in1d(
                        lowest_value_indices, noisy_indices
                    ).mean()

                    # Save raw values
                    column_name = f"{method_name}"
                    df = (
                        values.to_dataframe(column=column_name)
                        .drop(columns=[f"{column_name}_stderr"])
                        .T
                    )
                    df = df[sorted(df.columns)]
                    df["method"] = method_name

                    if all_values_df is None:
                        all_values_df = df.copy()
                    else:
                        all_values_df = pd.concat([all_values_df, df])

                    # Get the actual values
                    values = values.values

                    mask = np.ones(len(values), dtype=bool)
                    mask[noisy_indices] = False

                    shifted_values = values - np.min(values)
                    total_shifted_utility = np.sum(shifted_values)
                    total_shifted_clean_values = np.sum(shifted_values[mask])
                    shifted_clean_values_percentage = (
                        total_shifted_clean_values / total_shifted_utility
                    )

                    total_utility = np.sum(values)
                    total_clean_utility = np.sum(values[mask])
                    total_noisy_utility = total_utility - total_clean_utility

                    results = {
                        "noise_level": noise_level,
                        "noise_fraction": noise_fraction,
                        "clean_values_percentage": shifted_clean_values_percentage,
                        "total_clean_utility": total_clean_utility,
                        "total_noisy_utility": total_noisy_utility,
                        "total_utility": total_utility,
                        "noisy_accuracy": noisy_indices_accuracy,
                        "method": method_name,
                    }
                    all_results.append(results)
    results_df = pd.DataFrame(all_results)

    results_df.to_csv(experiment_output_dir / "results.csv", index=False)

    all_values_df.to_csv(experiment_output_dir / "values.csv", index=False)

    plot_clean_data_utility_percentage(
        results_df,
        method_names=method_names,
        noise_levels=noise_levels,
        experiment_output_dir=experiment_output_dir,
    )

    plot_clean_data_vs_noisy_data_utility(
        results_df,
        method_names=method_names,
        noise_fraction=noise_fraction,
        noise_levels=noise_levels,
        experiment_output_dir=experiment_output_dir,
    )

    plot_noisy_data_accuracy(results_df, experiment_output_dir=experiment_output_dir)

    logger.info("Finished Noisy Data Experiment")


if __name__ == "__main__":
    run()
