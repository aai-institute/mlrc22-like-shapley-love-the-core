import logging

import numpy as np
import pandas as pd
from pydvl.utils import Utility, powerset
from pydvl.utils.config import ParallelConfig
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from mlrc22.constants import OUTPUT_DIR, RANDOM_SEED
from mlrc22.dataset import (
    create_breast_cancer_dataset,
    create_house_voting_dataset,
    create_wine_dataset,
)
from mlrc22.nucleolus import exact_nucleolus, montecarlo_nucleolus
from mlrc22.plotting import plot_constraint_accuracy_over_coalitions
from mlrc22.utils import set_random_seed, setup_logger, setup_plotting

logger = setup_logger()

setup_plotting()
set_random_seed(RANDOM_SEED)


def run():
    logger.info("Starting Feature Valuation - Nucleolus Experiment")

    experiment_output_dir = OUTPUT_DIR / "feature_valuation_nucleolus"
    experiment_output_dir.mkdir(exist_ok=True)

    parallel_config = ParallelConfig(backend="ray", logging_level=logging.ERROR)

    scorer_names = ["accuracy", "f1", "average_precision"]
    fractions = [0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.20]

    n_repetitions = 10
    logger.info(f"Using number of repetitions {n_repetitions}")

    n_jobs = 8
    logger.info(f"Using number of jobs {n_jobs}")

    all_values_df = None

    accuracies = []

    random_state = np.random.RandomState(RANDOM_SEED)

    with tqdm_logging_redirect():
        for dataset_name in tqdm(["Medical", "Chemical"], desc="Dataset", leave=True):
            if dataset_name == "House":
                dataset = create_house_voting_dataset(random_state=random_state)
            elif dataset_name == "Medical":
                dataset = create_breast_cancer_dataset(random_state=random_state)
            elif dataset_name == "Chemical":
                dataset = create_wine_dataset(random_state=random_state)
            else:
                raise ValueError(f"Unknown dataset '{dataset_name}'")
            logger.info(f"Creating dataset '{dataset_name}'")

            logger.info(f"Number of features in dataset: {len(dataset)}")
            powerset_size = 2 ** len(dataset)

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(solver="liblinear"),
            )

            estimated_values = {fraction: [] for fraction in fractions}

            for scorer_name in tqdm(scorer_names, desc="Scorer", leave=False):
                logger.info(f"Creating utility with scorer: {scorer_name}")
                utility = Utility(
                    data=dataset,
                    model=model,
                    enable_cache=False,
                )
                logger.info("Computing exact Nucleolus values")
                exact_values = exact_nucleolus(
                    utility,
                    options={
                        "solver": "ECOS",
                        "max_iters": 30000,
                    },
                    progress=True,
                )
                breakpoint()

                logger.info("Computing approximate Nucleolus values")

                for fraction in tqdm(fractions, desc="Fractions", leave=False):
                    n_iterations = int(fraction * (2 ** len(dataset)))
                    logger.info(
                        f"Using number of iterations {n_iterations} for fraction {fraction}"
                    )
                    for _ in range(n_repetitions):
                        values = montecarlo_nucleolus(
                            utility,
                            epsilon=0.0,
                            n_iterations=n_iterations,
                            n_jobs=n_jobs,
                            config=parallel_config,
                            options={
                                "solver": "ECOS",
                                "max_iters": 30000,
                            },
                        )
                        estimated_values[fraction].append(values)

                        # Save raw values
                        column_name = "nucleolus"
                        df = (
                            values.to_dataframe(column=column_name)
                            .drop(columns=[f"{column_name}_stderr"])
                            .T
                        )
                        df = df[sorted(df.columns)]
                        df["fraction"] = fraction

                        if all_values_df is None:
                            all_values_df = df.copy()
                        else:
                            all_values_df = pd.concat([all_values_df, df])

                # This is inspired the code in pyDVL's exact_least_core() function
                # This creates the components of the following inequality:
                # $\sum_{i\in S} x_{i} + d(S) \geq v(S) &, \forall S \subseteq N$
                logger.info("Creating components of the Nucleolus constraints")
                constraints = np.zeros((powerset_size, len(dataset)))

                utility_values = np.zeros(powerset_size)
                with tqdm_logging_redirect():
                    for i, subset in tqdm(
                        enumerate(powerset(utility.data.indices)),
                        total=powerset_size,
                        desc="Subsets",
                    ):
                        utility_values[i] = utility(subset)
                        indices = np.zeros(len(dataset), dtype=bool)
                        indices[list(subset)] = True
                        constraints[i, indices] = 1

                logger.info("Computing accuracy for each fraction")

                for fraction, values_list in estimated_values.items():
                    for values in values_list:
                        sorted_results = sorted(
                            values, key=lambda x: getattr(x, "index", 0)
                        )
                        sorted_values = np.array(
                            [getattr(x, "value", x) for x in sorted_results]
                        )
                        subsidies = exact_values.subsidies
                        left_hand_side = constraints @ sorted_values + subsidies
                        accuracy = np.mean(left_hand_side >= utility_values)
                        accuracies.append(
                            {
                                "fraction": fraction,
                                "scorer": scorer_name,
                                "accuracy": accuracy,
                                "dataset": dataset_name,
                            }
                        )

    accuracies_df = pd.DataFrame(accuracies)

    accuracies_df.to_csv(experiment_output_dir / "accuracies.csv", index=False)

    all_values_df.to_csv(experiment_output_dir / "values.csv", index=False)

    plot_constraint_accuracy_over_coalitions(
        accuracies_df,
        scorer_names=scorer_names,
        method_name="nucleolus",
        experiment_output_dir=experiment_output_dir,
        use_log_scale=False,
    )

    plot_constraint_accuracy_over_coalitions(
        accuracies_df,
        scorer_names=scorer_names,
        method_name="nucleolus",
        experiment_output_dir=experiment_output_dir,
        use_log_scale=True,
    )

    logger.info("Finished Feature Valuation - Nucleolus Experiment")


if __name__ == "__main__":
    run()
