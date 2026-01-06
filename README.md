# Title

Short description...

## Installation

### Clone this repository

Download this repository and the required ESA Bitbucket repository for BepiColombo SPICE kernels.

**HTTPS**
```shell
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/daraghhollman/bepi-region-prediction.git
cd bepi-region-prediction
```

**SSH**
```shell
git clone --depth 1 --recurse-submodules --shallow-submodules git@github.com:daraghhollman/bepi-region-prediction.git
cd bepi-region-prediction
```

### Setup

Some initial setup is required before this repository can be used:

* Downloading a MESSENGER bow shock and magnetopause crossing list

This is all handled automatically by running the following script:

> [!TIP]
> For portability, we **strongly** recommend and use [uv](https://docs.astral.sh/uv/) to manage dependencies and versions. If you do not wish to use uv, dependencies can be found within [pyproject.toml](./pyproject.toml) and a `requirements.txt` file can be made with `pip compile pyproject.toml -o requirements.txt`.

```shell
uv run python src/setup/init.py
```
