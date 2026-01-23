### Producing quantified predictions of magnetospheric region for spacecraft in the near-Mercury environment based on MESSENGER observations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18348157.svg)](https://doi.org/10.5281/zenodo.18348157)

A software repository to determine the likelihood of observing solar wind, magnetosheath, and magnetosphere regions at any position in the near-Mercury environment.

## Clone this repository

**HTTPS**
```shell
git clone https://github.com/daraghhollman/bepi-region-prediction.git
cd bepi-region-prediction
```

**SSH**
```shell
git clone git@github.com:daraghhollman/bepi-region-prediction.git
cd bepi-region-prediction
```

## Setup

Some initial setup is required before this repository can be used:

* Downloading a MESSENGER bow shock and magnetopause crossing list
* Downloading BepiColombo SPICE kernels

This is all handled automatically by running the following scripts:

> [!TIP]
> For portability, we **strongly** recommend and use [uv](https://docs.astral.sh/uv/) to manage dependencies and versions. If you do not wish to use uv, dependencies can be found within [pyproject.toml](./pyproject.toml) and a `requirements.txt` file can be made with `pip compile pyproject.toml -o requirements.txt`.

```shell
uv run python src/setup/init.py
```

> [!NOTE]
> Note that while we use BepiColombo SPICE kernels as examples throughout this work, the outputs are applicable to any spacecraft in Mercury's magnetospheric environment, including MESSENGER.


### Creating maps of relative region occurrence

```shell
# Creates a dataset containing MESSENGER region observations at a 20 minutes time cadence.
uv run python src/determine_messenger_regions.py

# Bins the above observations spatially and determines a probabilitiy and uncertainty for each bin.
uv run python src/create_probabilitiy_maps.py
```

In the short future, we hope to include a Zenodo data repository accompanying this work so that these steps can be skipped, and a user can jump straight into the examples section.

## Examples

The two examples included in the publication can be found under `src/figure-creation/`, however, we instead recommend to first look at the examples directory: `src/examples/`, which includes worked examples in Python notebooks.
