import logging
import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydvl.utils import Utility
from pydvl.utils.config import ParallelConfig
from pydvl.utils.parallel import MapReduceJob
from pydvl.value.least_core._common import (
    _solve_egalitarian_least_core_quadratic_program,
    _solve_least_core_linear_program,
)
from pydvl.value.least_core.montecarlo import _montecarlo_least_core, _reduce_func
from pydvl.value.results import ValuationResult, ValuationStatus

logger = logging.getLogger(__name__)


__all__ = ["montecarlo_least_core"]


def montecarlo_least_core(
    u: Utility,
    n_iterations: int,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    *,
    epsilon: float = 0.0,
    options: Optional[dict] = None,
    progress: bool = False,
) -> ValuationResult:
    """Taken verbatim from pyDVL. The only change is in removing the check for
    n_iterations < len(u.data) because it isn't necessary.

    This is only needed for the feature valuation experiment because for one
    of the fractions we use n_iterations < len(u.data)
    """
    n = len(u.data)

    if n_iterations > 2**n:
        warnings.warn(
            f"Passed n_iterations is greater than the number subsets! Setting it to 2^{n}",
            RuntimeWarning,
        )
        n_iterations = 2**n

    if options is None:
        options = {}

    iterations_per_job = max(1, n_iterations // n_jobs)

    logger.debug("Instantiating MapReduceJob")
    map_reduce_job: MapReduceJob["Utility", Tuple["NDArray", "NDArray"]] = MapReduceJob(
        inputs=u,
        map_func=_montecarlo_least_core,
        reduce_func=_reduce_func,
        map_kwargs=dict(
            n_iterations=iterations_per_job,
            progress=progress,
        ),
        n_jobs=n_jobs,
        config=config,
    )
    logger.debug("Calling MapReduceJob instance")
    utility_values, A_lb = map_reduce_job()

    if np.any(np.isnan(utility_values)):
        warnings.warn(
            f"Calculation returned {np.sum(np.isnan(utility_values))} nan values out of {utility_values.size}",
            RuntimeWarning,
        )

    logger.debug("Building vectors and matrices for linear programming problem")
    b_lb = utility_values
    A_eq = np.ones((1, n))
    # We explicitly add the utility value for the entire dataset
    b_eq = np.array([u(u.data.indices)])

    logger.debug("Removing possible duplicate values in lower bound array")
    A_lb, unique_indices = np.unique(A_lb, return_index=True, axis=0)
    b_lb = b_lb[unique_indices]

    _, subsidy = _solve_least_core_linear_program(
        A_eq=A_eq, b_eq=b_eq, A_lb=A_lb, b_lb=b_lb, epsilon=epsilon, **options
    )

    values: Optional[NDArray[np.float_]]

    if subsidy is None:
        logger.debug("No values were found")
        status = ValuationStatus.Failed
        values = np.empty(n)
        values[:] = np.nan
        subsidy = np.nan

        return ValuationResult(
            algorithm="montecarlo_least_core",
            status=status,
            values=values,
            subsidy=subsidy,
            stderr=None,
            data_names=u.data.data_names,
        )

    values = _solve_egalitarian_least_core_quadratic_program(
        subsidy,
        A_eq=A_eq,
        b_eq=b_eq,
        A_lb=A_lb,
        b_lb=b_lb,
        **options,
    )

    if values is None:
        logger.debug("No values were found")
        status = ValuationStatus.Failed
        values = np.empty(n)
        values[:] = np.nan
        subsidy = np.nan
    else:
        status = ValuationStatus.Converged

    return ValuationResult(
        algorithm="montecarlo_least_core",
        status=status,
        values=values,
        subsidy=subsidy,
        stderr=None,
        data_names=u.data.data_names,
    )
