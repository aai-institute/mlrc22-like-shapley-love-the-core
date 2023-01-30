import logging

import numpy as np
import pandas as pd
from pydvl.utils import Utility
from pydvl.utils.config import ParallelConfig
from pydvl.value.least_core import montecarlo_least_core
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
)
from mlrc22.utils import set_random_seed, setup_logger, setup_plotting

logger = setup_logger()

setup_plotting()
set_random_seed(RANDOM_SEED)


def run():
    experiment_output_dir = OUTPUT_DIR / "noisy_data"
    experiment_output_dir.mkdir(exist_ok=True)

    parallel_config = ParallelConfig(backend="ray", logging_level=logging.ERROR)

    n_features = 50
    n_train_samples = 200
    n_test_samples = 5000

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    noise_fractions = [0.2, 0.4]

    n_iterations = 5000
    logger.info(f"Using number of iterations {n_iterations}")

    n_repetitions = 5
    logger.info(f"Using number of repetitions {n_repetitions}")

    n_jobs = 8
    logger.info(f"Using number of jobs {n_jobs}")

    random_state = np.random.RandomState(RANDOM_SEED)

    all_results = []

    with tqdm_logging_redirect():
        for noise_fraction in tqdm(noise_fractions, desc="Noise Fraction", leave=True):
            logger.info(f"{noise_fraction=}")
            for noise_level in tqdm(noise_levels, desc="Noise Level", leave=False):
                logger.info(f"{noise_level=}")

                for _ in trange(n_repetitions, desc="Repetitions", leave=False):
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
                    logger.info("Computing approximate Least Core values")

                    try:
                        valuation = montecarlo_least_core(
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
                        values = valuation.values
                    except ValueError:
                        values = np.empty(len(dataset))
                        values[:] = np.nan

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
                    }
                    all_results.append(results)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(experiment_output_dir / "results.csv", index=False)

    plot_clean_data_utility_percentage(
        results_df,
        noise_fractions=noise_fractions,
        noise_levels=noise_levels,
        experiment_output_dir=experiment_output_dir,
    )

    plot_clean_data_vs_noisy_data_utility(
        results_df,
        noise_fractions=noise_fractions,
        noise_levels=noise_levels,
        experiment_output_dir=experiment_output_dir,
    )


if __name__ == "__main__":
    run()
