import io
import logging
from contextlib import redirect_stderr

import numpy as np
import pandas as pd
from pydvl.reporting.scores import compute_removal_score
from pydvl.utils import Utility
from pydvl.utils.config import ParallelConfig
from pydvl.value.least_core import montecarlo_least_core
from pydvl.value.loo import naive_loo
from pydvl.value.results import ValuationResult
from pydvl.value.shapley import ShapleyMode, compute_shapley_values
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import tqdm_logging_redirect

from mlrc22.constants import OUTPUT_DIR, RANDOM_SEED
from mlrc22.dataset import create_synthetic_dataset
from mlrc22.plotting import plot_utility_over_removal_percentages
from mlrc22.utils import set_random_seed, setup_logger, setup_plotting

logger = setup_logger()

setup_plotting()
set_random_seed(RANDOM_SEED)


def run():
    experiment_output_dir = OUTPUT_DIR / "data_valuation_synthetic"
    experiment_output_dir.mkdir(exist_ok=True)

    parallel_config = ParallelConfig(backend="ray", logging_level=logging.ERROR)

    n_features = 50
    n_train_samples = 200
    n_test_samples = 5000

    removal_percentages = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    method_names = [
        "TMC Shapley",
        "Group Testing Shapley",
        "Least Core",
        "Leave One Out",
        "Random",
    ]

    budget_list = [10000, 50000]

    n_repetitions = 20
    logger.info(f"Using number of repetitions {n_repetitions}")

    n_jobs = 8
    logger.info(f"Using number of jobs {n_jobs}")

    all_scores = []

    random_state = np.random.RandomState(RANDOM_SEED)

    with tqdm_logging_redirect():
        for budget in tqdm(budget_list, desc="Budget", leave=True):
            logger.info(f"Using number of iterations {budget}")
            for _ in trange(
                n_repetitions,
                desc="Repetitions",
                leave=False,
            ):

                dataset, _ = create_synthetic_dataset(
                    n_features=n_features,
                    n_train_samples=n_train_samples,
                    n_test_samples=n_test_samples,
                    random_state=random_state,
                )

                for method_name in tqdm(method_names, desc="Method", leave=False):
                    logger.info(f"{method_name=}")

                    # We do not set the random_state in the model itself
                    # because we are testing the method and not the model
                    model = make_pipeline(
                        StandardScaler(), LogisticRegression(solver="liblinear")
                    )

                    logger.info("Creating utility")
                    utility = Utility(
                        data=dataset,
                        model=model,
                        score_range=(0.0, 1.0),
                        enable_cache=False,
                    )

                    if method_name == "Random":
                        values = ValuationResult.from_random(size=len(utility.data))
                    elif method_name == "Leave One Out":
                        values = naive_loo(utility, progress=False)
                    elif method_name == "Least Core":
                        values = montecarlo_least_core(
                            utility,
                            epsilon=0.0,
                            n_iterations=budget,
                            n_jobs=n_jobs,
                            config=parallel_config,
                            options={
                                "solver": "SCS",
                                "max_iters": 30000,
                            },
                        )
                    else:
                        if method_name == "Group Testing Shapley":
                            mode = ShapleyMode.GroupTesting
                            n_iterations = budget
                            kwargs = {"epsilon": 0.1}
                        else:
                            mode = ShapleyMode.TruncatedMontecarlo
                            # The budget for TMCShapley methods is less because
                            # for each iteration it goes over all indices
                            # of an entire permutation of indices
                            n_iterations = budget // len(utility.data)
                            kwargs = {}
                        f = io.StringIO()
                        with redirect_stderr(f):
                            values = compute_shapley_values(
                                utility,
                                n_iterations=n_iterations,
                                n_jobs=n_jobs,
                                config=parallel_config,
                                mode=mode,
                                **kwargs,
                            )

                    # Remove worst data points
                    scores = compute_removal_score(
                        u=utility,
                        values=values,
                        percentages=removal_percentages,
                        remove_best=False,
                    )
                    scores["method"] = method_name
                    scores["budget"] = budget
                    scores["type"] = "worst"
                    all_scores.append(scores)

                    # Remove best data points
                    scores = compute_removal_score(
                        u=utility,
                        values=values,
                        percentages=removal_percentages,
                        remove_best=True,
                    )
                    scores["method"] = method_name
                    scores["budget"] = budget
                    scores["type"] = "best"
                    all_scores.append(scores)

    scores_df = pd.DataFrame(all_scores)

    scores_df.to_csv(experiment_output_dir / "scores.csv", index=False)

    plot_utility_over_removal_percentages(
        scores_df,
        budget_list=budget_list,
        method_names=method_names,
        removal_percentages=removal_percentages,
        experiment_output_dir=experiment_output_dir,
    )


if __name__ == "__main__":
    run()
