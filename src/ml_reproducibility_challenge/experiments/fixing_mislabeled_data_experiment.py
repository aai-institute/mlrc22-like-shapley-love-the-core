import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydvl.reporting.plots import shaded_mean_std
from pydvl.reporting.scores import compute_removal_score
from pydvl.utils import Utility
from pydvl.utils.config import ParallelConfig
from pydvl.value.least_core import montecarlo_least_core
from pydvl.value.results import ValuationResult
from sklearn.naive_bayes import GaussianNB
from tqdm.auto import trange
from tqdm.contrib.logging import tqdm_logging_redirect

from ml_reproducibility_challenge.constants import OUTPUT_DIR, RANDOM_SEED
from ml_reproducibility_challenge.utils import (
    create_enron_spam_datasets,
    set_random_seed,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

sns.set_theme(style="whitegrid", palette="pastel")
sns.set_context("paper", font_scale=1.5)

EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "fixing_mislabeled_data"
EXPERIMENT_OUTPUT_DIR.mkdir(exist_ok=True)

mean_colors = ["dodgerblue", "indianred"]
shade_colors = ["lightskyblue", "firebrick"]


def run():
    parallel_config = ParallelConfig(backend="ray", address="ray://127.0.0.1:10001")

    logger.info(f"Creating datasets")
    training_dataset, testing_dataset = create_enron_spam_datasets()

    logger.info(f"Training dataset size: {len(training_dataset)}")
    logger.info(f"Testing dataset size: {len(testing_dataset)}")

    model = GaussianNB()

    logger.info("Creating utilities")
    training_utility = Utility(
        data=training_dataset,
        model=model,
        enable_cache=False,
    )

    testing_utility = Utility(
        data=testing_dataset,
        model=model,
        enable_cache=False,
    )

    removal_percentages = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    max_iterations = 5000
    logger.info(f"Using number of iterations {max_iterations}")

    n_repetitions = 5
    logger.info(f"Using number of repetitions {n_repetitions}")

    method_names = ["Random", "Least Core"]
    all_scores = []

    with tqdm_logging_redirect():
        for method_name in method_names:
            for _ in trange(n_repetitions, desc=f"Repetitions '{method_name}'"):
                if method_name == "Random":
                    values = ValuationResult.from_random(
                        size=len(training_utility.data)
                    )
                else:
                    values = montecarlo_least_core(
                        training_utility,
                        epsilon=0.01,
                        max_iterations=max_iterations,
                        n_jobs=4,
                        config=parallel_config,
                    )
                scores = compute_removal_score(
                    u=testing_utility,
                    values=values,
                    percentages=removal_percentages,
                    progress=True,
                )
                scores["method"] = method_name
                all_scores.append(scores)

    scores_df = pd.DataFrame(all_scores)

    scores_df.to_csv(EXPERIMENT_OUTPUT_DIR / "scores.csv", index=False)

    fig, ax = plt.subplots()

    for i, method_name in enumerate(method_names):
        shaded_mean_std(
            scores_df[scores_df["method"] == method_name].drop(columns=["method"]),
            abscissa=removal_percentages,
            mean_color=mean_colors[i],
            shade_color=shade_colors[i],
            xlabel="Percentage Removal",
            ylabel="Accuracy",
            label=f"{method_name}",
            title="Test performance as we remove more and more training data with"
            "the worst valuation guided by the least core vs. random selection.",
            ax=ax,
        )
    plt.legend()
    fig.savefig(EXPERIMENT_OUTPUT_DIR / "accuracy_over_removal_percentages.eps")


if __name__ == "__main__":
    set_random_seed(RANDOM_SEED)
    run()
