import io
import logging
from contextlib import redirect_stderr
from time import time

import numpy as np
import pandas as pd
import pydvl.value.least_core
from pydvl.utils import Utility, init_parallel_backend
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
    plot_values_histogram,
)
from mlrc22.utils import set_random_seed, setup_logger, setup_plotting

logger = setup_logger()

setup_plotting()
set_random_seed(RANDOM_SEED)


def run():
    logger.info("Starting Noisy Data Experiment")

    experiment_output_dir = OUTPUT_DIR / "noisy_data"
    experiment_output_dir.mkdir(exist_ok=True)

    parallel_config = ParallelConfig(
        backend="ray", logging_level=logging.ERROR, n_local_workers=1
    )

    n_features = 50
    n_train_samples = 200
    n_test_samples = 5000

    noise_levels = [0.0]  # , 0.5, 1.0, 2.0, 3.0]
    noise_fraction = 0.2
    logger.info(f"{noise_fraction=}")

    method_names = ["Least Core", "TMC Shapley", "Random"]

    n_iterations = 300
    logger.info(f"Using number of iterations {n_iterations}")

    n_repetitions = 1  # 0
    logger.info(f"Using number of repetitions {n_repetitions}")

    n_jobs = 1
    logger.info(f"Using number of jobs {n_jobs}")

    random_state = np.random.RandomState(RANDOM_SEED)

    all_values_df = None

    all_results = []
    promised_values = []
    parallel_backend = init_parallel_backend(parallel_config)

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
                    LogisticRegression(solver="liblinear", n_jobs=1),
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
                        fun = lambda: dict(
                            values=values,
                            method=method_name,
                            noise_level=noise_level,
                            noise_fraction=noise_fraction,
                        )
                    elif method_name == "Least Core":
                        problem = montecarlo_least_core(
                            utility,
                            n_iterations=n_iterations,
                            n_jobs=n_jobs,
                            config=parallel_config,
                            options={
                                "solver": "SCS",
                                "max_iters": 30000,
                            },
                        )
                        fun = lambda: dict(
                            values=pydvl.value.least_core.montecarlo._solve_linear_problem(
                                utility, problem
                            ),
                            method=method_name,
                            noise_level=noise_level,
                            noise_fraction=noise_fraction,
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
                            fun = lambda: dict(
                                values=values,
                                method=method_name,
                                noise_level=noise_level,
                                noise_fraction=noise_fraction,
                            )

                    promised_values.append(parallel_backend.wrap(fun).remote())

        logger.info(f"Waiting for {len(promised_values)} promised values")
        start = time()
        resolved_values = parallel_backend.get(promised_values)
        logger.info(
            f"Resolved {len(resolved_values)} promised values in {time() - start:.2f} seconds"
        )

        for result in tqdm(resolved_values, desc="Waiting for values", leave=True):
            values = result["values"]
            method_name = result["method"]
            noise_level = result["noise_level"]
            noise_fraction = result["noise_fraction"]

            # Sort values in increasing order
            values.sort()
            # The noisy data points should have the lowest valuation
            # So to verify that we compute the percentage of data points
            # with the lowest valuation that correspond to the noisy data points
            lowest_value_indices = values.indices[: len(noisy_indices)]
            assert lowest_value_indices.shape == noisy_indices.shape
            noisy_indices_accuracy = np.in1d(lowest_value_indices, noisy_indices).mean()

            # Save raw values
            column_name = f"{method_name}"
            df = (
                values.to_dataframe(column=column_name)
                .drop(columns=[f"{column_name}_stderr"])
                .T
            )
            df = df[sorted(df.columns)]
            df["method"] = method_name
            df["noise_level"] = noise_level
            df["noise_fraction"] = noise_fraction

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

    plot_values_histogram(
        all_values_df,
        hue_column="noise_level",
        method_names=method_names,
        experiment_output_dir=experiment_output_dir,
    )

    plot_noisy_data_accuracy(results_df, experiment_output_dir=experiment_output_dir)

    logger.info("Finished Noisy Data Experiment")


if __name__ == "__main__":
    run()
