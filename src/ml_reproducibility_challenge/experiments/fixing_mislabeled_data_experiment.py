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
from tqdm.auto import tqdm, trange
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

    scorer_names = ["accuracy", "f1", "average_precision"]
    removal_percentages = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    label_flip_percentages = [0.10, 0.20, 0.30]
    method_names = ["Random", "Least Core"]

    n_iterations = 5000
    logger.info(f"Using number of iterations {n_iterations}")

    n_repetitions = 5
    logger.info(f"Using number of repetitions {n_repetitions}")

    logger.info("Using Gaussian Naive Bayes model")
    model = GaussianNB()

    all_scores = []

    with tqdm_logging_redirect():
        for flip_percentage in tqdm(
            label_flip_percentages, desc="Flip Percentage", position=0, leave=False
        ):
            logger.info(f"{flip_percentage=}")
            logger.info(f"Creating datasets")
            training_dataset, testing_dataset = create_enron_spam_datasets(
                flip_percentage, random_state=RANDOM_SEED
            )
            logger.info(f"Training dataset size: {len(training_dataset)}")
            logger.info(f"Testing dataset size: {len(testing_dataset)}")

            for scorer_name in tqdm(
                scorer_names, desc="Scorer", position=1, leave=False
            ):
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

                for method_name in tqdm(
                    method_names, desc="Method", position=2, leave=True
                ):
                    logger.info(f"{method_name=}")
                    for _ in trange(
                        n_repetitions,
                        desc=f"Repetitions '{method_name}'",
                        position=3,
                        leave=False,
                    ):
                        if method_name == "Random":
                            values = ValuationResult.from_random(
                                size=len(training_utility.data)
                            )
                        else:
                            values = montecarlo_least_core(
                                training_utility,
                                epsilon=0.0,
                                n_iterations=n_iterations,
                                n_jobs=4,
                                config=parallel_config,
                                options={
                                    "max_iters": 10000,
                                },
                            )
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
                        all_scores.append(scores)

    scores_df = pd.DataFrame(all_scores)

    scores_df.to_csv(EXPERIMENT_OUTPUT_DIR / "scores.csv", index=False)

    for scorer in scorer_names:
        for flip_percentage in label_flip_percentages:
            fig, ax = plt.subplots()

            for i, method_name in enumerate(method_names):
                df = scores_df[
                    (scores_df["method"] == method_name)
                    & (scores_df["scorer"] == scorer)
                    & (scores_df["flip_percentage"] == flip_percentage)
                ].drop(columns=["method", "scorer", "flip_percentage"])
                shaded_mean_std(
                    df,
                    abscissa=removal_percentages,
                    mean_color=mean_colors[i],
                    shade_color=shade_colors[i],
                    xlabel="Percentage Removal",
                    ylabel="Accuracy",
                    label=f"{method_name}",
                    ax=ax,
                )
            plt.legend(loc="lower left")
            fig.tight_layout()
            fig.savefig(
                EXPERIMENT_OUTPUT_DIR
                / f"accuracy_over_removal_percentages_{scorer}_{flip_percentage:.2f}.pdf"
            )


if __name__ == "__main__":
    set_random_seed(RANDOM_SEED)
    run()
