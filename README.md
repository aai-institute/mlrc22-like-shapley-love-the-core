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

# Experiments

## Feature Valuation

```shell
python -m experiments.feature_valuation_experiments
```

# Contributing

Make sure to install the pre-commit hooks:

```shell
pre-commit install
```
