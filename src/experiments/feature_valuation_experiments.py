import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydvl.utils import Utility, powerset
from pydvl.value.least_core import exact_least_core, montecarlo_least_core
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from experiments.constants import OUTPUT_DIR, RANDOM_SEED
from experiments.utils import (
    create_breast_cancer_dataset,
    create_house_voting_dataset,
    create_wine_dataset,
    set_random_seed,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

sns.set_theme(style="whitegrid", palette="pastel")
sns.set_context("paper", font_scale=1.5)

EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "feature_valuation"
EXPERIMENT_OUTPUT_DIR.mkdir(exist_ok=True)


def run():
    accuracies = []

    for dataset_name in ["House", "Medical", "Chemical"]:
        if dataset_name == "House":
            dataset = create_house_voting_dataset()
        elif dataset_name == "Medical":
            dataset = create_breast_cancer_dataset()
        elif dataset_name == "Chemical":
            dataset = create_wine_dataset()
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}'")
        logger.info(f"Creating dataset '{dataset_name}")

        logger.info(f"Number of features in dataset: {len(dataset)}")
        powerset_size = 2 ** len(dataset)

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

        logger.info("Creating utility")
        utility = Utility(
            data=dataset,
            model=model,
            enable_cache=False,
        )

        logger.info("Computing exact Least Core values")
        exact_values = exact_least_core(utility, progress=True)

        logger.info(
            "Computing estimated Least Core values using fractions of the number of subsets"
        )
        fractions = [0.01, 0.02, 0.05, 0.075, 0.1, 0.15]

        estimated_values = {fraction: [] for fraction in fractions}

        n_repetitions = 10

        with tqdm_logging_redirect():
            for fraction in tqdm(fractions, desc="Fractions"):
                max_iterations = int(fraction * (2 ** len(dataset)))
                logger.info(
                    f"Using number of iterations {max_iterations} for fraction {fraction}"
                )
                for _ in range(n_repetitions):
                    try:
                        values = montecarlo_least_core(
                            utility,
                            epsilon=0.0,
                            max_iterations=max_iterations,
                            n_jobs=4,
                        )
                    except ValueError:
                        values = np.empty(len(dataset))
                        values[:] = np.nan
                        estimated_values[fraction].append(values)
                    else:
                        estimated_values[fraction].append(values)

        # This is inspired the code in pyDVL's exact_least_core() function
        # This creates the components of the following inequality:
        # $\sum_{i\in S} x_{i} + e \geq v(S) &, \forall S \subseteq N$
        logger.info("Creating components of least core constraints")
        constraints = np.zeros((powerset_size, len(dataset) + 1))
        constraints[:, -1] = 1

        utility_values = np.zeros(powerset_size)
        with tqdm_logging_redirect():
            for i, subset in tqdm(
                enumerate(powerset(utility.data.indices)),
                total=powerset_size,
                desc="Subsets",
            ):
                utility_values[i] = utility(subset)
                indices = np.zeros(len(dataset) + 1, dtype=bool)
                indices[list(subset)] = True
                constraints[i, indices] = 1

        logger.info("Computing accuracy for each fraction")

        for fraction, values_list in estimated_values.items():
            for values in values_list:
                sorted_results = sorted(values, key=lambda x: getattr(x, "index", 0))
                sorted_values = np.array(
                    [getattr(x, "value", x) for x in sorted_results]
                    + [exact_values.least_core_value]
                )
                left_hand_side = constraints @ sorted_values
                accuracy = np.mean(left_hand_side >= utility_values)
                accuracies.append(
                    {
                        "fraction": fraction,
                        "accuracy": accuracy,
                        "dataset": dataset_name,
                    }
                )

    accuracies_df = pd.DataFrame(accuracies)

    accuracies_df.to_csv(EXPERIMENT_OUTPUT_DIR / "accuracies.csv", index=False)

    fig, ax = plt.subplots()
    sns.barplot(
        data=accuracies_df,
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
    fig.savefig(EXPERIMENT_OUTPUT_DIR / "least_core_accuracy_over_coalitions.eps")


if __name__ == "__main__":
    set_random_seed(RANDOM_SEED)
    run()
