[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/aai-institute/mlrc22-like-shapley-love-the-core/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/aai-institute/mlrc22-like-shapley-love-the-core)

# MLRC 2022: If you lik Shapley, then you'll love the core

This repository contains code to reproduce the paper
[`If You Like Shapley Then You’ll Love the Core`](http://procaccia.info/wp-content/uploads/2020/12/core.pdf)
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

We use [DVC](https://dvc.org/) to run the experiments and track their results.

To reproduce all results use:

```shell
dvc repro
```

## Feature Valuation

### Least Core

To reproduce the results of this experiment use:

```shell
dvc repro feature-valuation-least-core
```

You can find the results under [output/feature_valuation_least_core](output/feature_valuation_least_core).


## Data Valuation

### Synthetic Data

To reproduce the results of this experiment use:

```shell
dvc repro data-valuation-synthetic
```

You can find the results under [output/data_valuation_synthetic](output/data_valuation_synthetic).

### Dog vs Fish Dataset

> **Note**:
> This experiment requires downloading the [imagenet-1k](https://huggingface.co/datasets/imagenet-1k) dataset from
> [HuggingFace Datasets](https://huggingface.co/datasets).
> For that you need to first create an account and then login using
> the [huggingface-cli](https://huggingface.co/docs/huggingface_hub/quick-start#login) tool.

To reproduce the results of this experiment use:

```shell
dvc repro data-valuation-dog-vs-fish
```

You can find the results under [output/data_valuation_dog_vs_fish](output/data_valuation_dog_vs_fish).

## Fixing Misalabeled Data

To reproduce the results of this experiment use:

```shell
dvc repro fixing-mislabeled-data
```

You can find the results under [output/fixing_mislabeled_data](output/fixing_mislabeled_data).

## Noisy Data

To reproduce the results of this experiment use:

```shell
dvc repro noisy-data
```

You can find the results under [output/noisy_data](output/noisy_data).

# Contributing

Make sure to install the pre-commit hooks:

```shell
pre-commit install
```
