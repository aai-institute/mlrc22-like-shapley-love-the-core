import logging
import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydvl.utils import Utility, maybe_progress, powerset
from pydvl.utils.config import ParallelConfig
from pydvl.utils.parallel import MapReduceJob
from pydvl.value.least_core._common import _solve_least_core_linear_program
from pydvl.value.least_core.montecarlo import _montecarlo_least_core, _reduce_func
from pydvl.value.results import ValuationResult, ValuationStatus

__all__ = ["exact_nucleolus", "montecarlo_nucleolus"]

logger = logging.getLogger(__name__)


def _find_matching_indices(
    x: NDArray[np.float_], y: NDArray[np.float_]
) -> NDArray[np.int_]:
    """Finds indices in x of rows that are both in x and y.
    This assumes that all rows in y are contained in x and that both
    are 2D arrays.
    """
    all_indices = []
    for row in y:
        indices = np.where(
            np.apply_along_axis(lambda x: np.equal(x, row).all(), axis=1, arr=x)
        )[0]
        all_indices.append(indices)
    return np.hstack(all_indices)


def _solve_nucleolus(
    A_eq: NDArray[np.float_],
    b_eq: NDArray[np.float_],
    A_lb: NDArray[np.float_],
    b_lb: NDArray[np.float_],
    *,
    epsilon: float = 0.0,
    progress: bool = False,
    **options,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    # Copy these here because we need the original ones to keep track of indices
    reduced_A_eq = A_eq.copy()
    reduced_b_eq = b_eq.copy()
    reduced_A_lb = A_lb.copy()
    reduced_b_lb = b_lb.copy()

    subsidies = np.zeros_like(b_lb)

    for i in maybe_progress(
        range(len(A_lb)),
        progress,
        total=len(A_lb),
    ):
        logger.debug(f"iteration {i}")
        values, subsidy = _solve_least_core_linear_program(
            A_eq=reduced_A_eq,
            b_eq=reduced_b_eq,
            A_lb=reduced_A_lb,
            b_lb=reduced_b_lb,
            epsilon=epsilon,
            **options,
        )

        if subsidy is None:
            continue

        excess = reduced_b_lb - (reduced_A_lb @ values) - subsidy

        # Find indices that satisfy the constraints
        satisfied_indices = np.where(np.isclose(excess, 0.0))[0]
        not_satisfied_indices = np.where(~np.isclose(excess, 0.0))[0]

        # Update subsidies
        matched_indices = _find_matching_indices(A_lb, reduced_A_lb[satisfied_indices])
        subsidies[matched_indices] = subsidy

        if len(not_satisfied_indices) == 0:
            break

        if len(satisfied_indices) == 0:
            raise RuntimeError("Could not solve Nucleolus")

        # Add satisfied lower bound constraints to the equality constraints
        reduced_A_eq = np.concatenate(
            [reduced_A_eq, reduced_A_lb[satisfied_indices]], axis=0
        )
        reduced_b_eq = np.concatenate(
            [reduced_b_eq, reduced_b_lb[satisfied_indices] - subsidy], axis=0
        )

        # Keep only not satisfied lower bound constraints
        reduced_A_lb = reduced_A_lb[not_satisfied_indices]
        reduced_b_lb = reduced_b_lb[not_satisfied_indices]
    return values, subsidies


def exact_nucleolus(
    u: Utility, *, options: Optional[dict] = None, progress: bool = False, **kwargs
) -> ValuationResult:
    """Code heavily inspired by pyDVL"""
    n = len(u.data)

    # Arbitrary choice, will depend on time required, caching, etc.
    if n > 20:
        warnings.warn(f"Large dataset! Computation requires 2^{n} calls to model.fit()")

    if options is None:
        options = {}

    powerset_size = 2**n

    logger.debug("Building vectors and matrices for linear programming problem")
    A_eq = np.ones((1, n))
    A_lb = np.zeros((powerset_size, n))

    logger.debug("Iterating over all subsets")
    utility_values = np.zeros(powerset_size)
    for i, subset in enumerate(
        maybe_progress(
            powerset(u.data.indices),
            progress,
            total=powerset_size - 1,
            position=0,
        )
    ):
        indices = np.zeros(n, dtype=bool)
        indices[list(subset)] = True
        A_lb[i, indices] = 1
        utility_values[i] = u(subset)

    b_lb = utility_values
    b_eq = utility_values[-1:]

    values, subsidies = _solve_nucleolus(
        A_eq=A_eq, b_eq=b_eq, A_lb=A_lb, b_lb=b_lb, progress=progress, **options
    )

    return ValuationResult(
        algorithm="exact_nucleolus",
        status=ValuationStatus.Converged,
        values=values,
        subsidies=subsidies,
        stderr=None,
        data_names=u.data.data_names,
    )


def montecarlo_nucleolus(
    u: Utility,
    n_iterations: int,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    *,
    epsilon: float = 0.0,
    options: Optional[dict] = None,
    progress: bool = False,
) -> ValuationResult:
    """Code heavily inspired by pyDVL"""
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

    values, subsidies = _solve_nucleolus(
        A_eq=A_eq, b_eq=b_eq, A_lb=A_lb, b_lb=b_lb, **options
    )

    return ValuationResult(
        algorithm="montecarlo_nucleolus",
        status=ValuationStatus.Converged,
        values=values,
        subsidies=subsidies,
        stderr=None,
        data_names=u.data.data_names,
    )
