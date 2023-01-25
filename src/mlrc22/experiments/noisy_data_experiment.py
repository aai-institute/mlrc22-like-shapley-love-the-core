import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydvl.reporting.plots import shaded_mean_std
from pydvl.utils import Utility
from pydvl.utils.config import ParallelConfig
from pydvl.value.least_core import montecarlo_least_core
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import tqdm_logging_redirect

from mlrc22.constants import OUTPUT_DIR, RANDOM_SEED
from mlrc22.utils import create_synthetic_dataset, set_random_seed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

sns.set_theme(style="whitegrid", palette="pastel")
sns.set_context("paper", font_scale=1.5)

EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "noisy_data"
EXPERIMENT_OUTPUT_DIR.mkdir(exist_ok=True)

mean_colors = ["limegreen", "indianred", "dodgerblue"]
shade_colors = ["seagreen", "firebrick", "lightskyblue"]


def plot_clean_data_utility_percentage(
    results_df: pd.DataFrame, *, noise_fractions: list[float], noise_levels: list[float]
) -> None:
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
        EXPERIMENT_OUTPUT_DIR / f"clean_data_utility_percentage.pdf",
        bbox_inches="tight",
    )


def plot_clean_data_vs_noisy_data_utility(
    results_df: pd.DataFrame, *, noise_fractions: list[float], noise_levels: list[float]
) -> None:
    for noise_fraction in noise_fractions:
        df = results_df[(results_df["noise_fraction"] == noise_fraction)]
        df = (
            df.groupby("noise_level")[
                ["total_clean_values", "total_noisy_values", "total_utility"]
            ]
            .apply(lambda df: df.reset_index(drop=True))
            .unstack()
        )
        fig, ax = plt.subplots()
        for i, (column, ylabel) in enumerate(
            zip(
                ["total_clean_values", "total_noisy_values", "total_utility"],
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
            EXPERIMENT_OUTPUT_DIR
            / f"clean_data_vs_noisy_data_utility_{noise_fraction:.2f}.pdf",
            bbox_inches="tight",
        )


def run():
    parallel_config = ParallelConfig(backend="ray", logging_level=logging.ERROR)

    n_features = 50
    n_train_samples = 200
    n_test_samples = 5000

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    noise_fractions = [0.1, 0.3]

    n_iterations = 5000
    logger.info(f"Using number of iterations {n_iterations}")

    n_repetitions = 5
    logger.info(f"Using number of repetitions {n_repetitions}")

    n_jobs = 4
    logger.info(f"Using number of jobs {n_jobs}")

    random_state = np.random.RandomState(RANDOM_SEED)

    all_results = []

    with tqdm_logging_redirect():
        for noise_fraction in tqdm(noise_fractions, desc="Noise Fraction", leave=True):
            for noise_level in tqdm(noise_levels, desc="Noise Level", leave=True):
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
                    LogisticRegression(solver="liblinear", random_state=random_state),
                )
                logger.info(f"Creating utility")
                utility = Utility(
                    data=dataset,
                    model=model,
                    enable_cache=False,
                )
                for _ in trange(n_repetitions, desc="Repetitions", leave=False):
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

                    shifted_values = values + np.min(values)
                    total_utility = np.sum(shifted_values)
                    total_clean_values = np.sum(shifted_values[mask])
                    clean_values_percentage = total_clean_values / total_utility

                    results = {
                        "noise_level": noise_level,
                        "noise_fraction": noise_fraction,
                        "clean_values_percentage": clean_values_percentage,
                        "total_clean_values": total_clean_values,
                        "total_noisy_values": total_utility - total_clean_values,
                        "total_utility": total_utility,
                    }
                    all_results.append(results)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(EXPERIMENT_OUTPUT_DIR / "results.csv", index=False)

    plot_clean_data_utility_percentage(
        results_df, noise_fractions=noise_fractions, noise_levels=noise_levels
    )

    plot_clean_data_vs_noisy_data_utility(
        results_df, noise_fractions=noise_fractions, noise_levels=noise_levels
    )


if __name__ == "__main__":
    set_random_seed(RANDOM_SEED)
    run()
