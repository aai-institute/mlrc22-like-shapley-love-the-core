# ML-Reproducibility-Challenge-2022

This repository contains code to reproduce the paper
[`If You Like Shapley Then You’ll Love the Core`](https://ojs.aaai.org/index.php/AAAI/article/view/16721)
for the [ML Reproducibility Challenge 2022](https://paperswithcode.com/rc2022).

# Getting Started

We use Python version 3.10 for this repository.

We use [Poetry](https://python-poetry.org/) for dependency management. More specifically version `1.2.0`.

After installing Poetry, run the following command to create a virtual environment and install
all dependencies:

```shell
poetry install
```

You can then activate the virtual environment using:

```shell
poetry shell
```

## Ray

The main dependency used to compute the results uses [ray]() internally
to parallelize its computations.

Unfortunately, starting a ray cluster directly is known to mess up with
the random seed set in the main process (See this
[issue](https://github.com/ray-project/ray/issues/10145) for more information).

Therefore, we decided to include a docker-compose file in this repository
to start the ray cluster separately.

To start the cluster simply use:

```shell
docker-compose up --build --scale ray-worker=4
```

This will start a head container along with 4 worker containers.

Once it's started you can navigate to this address http://127.0.0.1:8080
to view the ray dashboard.

# Experiments

We use [DVC](https://dvc.org/) to run the experiments and track their results.

To reproduce all results use:

```shell
dvc repro
```

## Feature Valuation

This experiments uses 3 small scale datasets with a number of features between
10 and 14. It uses monte carlo least core to compute feature valuations
and then computes, for a varying number of computational budgets, the percentage
of all feature coalitions that satisfy the least core constraints with respect
to the true deficit $e^{*}$ (i.e. the exact least core value).

To reproduce the results of this experiment use:

```shell
dvc repro feature-valuation
```

You can find the results under [output/feature_valuation](output/feature_valuation).

# Contributing

Make sure to install the pre-commit hooks:

```shell
pre-commit install
```
