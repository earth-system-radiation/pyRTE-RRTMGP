# Installation and Setup

pyRTE-RRTMGP is currently in early stages of development and is not yet available on conda forge.

To install the package, you can clone the repository and build and install the conda package locally.

## Prerequisites

pyRTE-RRTMGP is currently only tested on x86_64 architecture with Linux and MacOS.
The package might also work with other architectures (such as Apple Silicone or Linux ARM), but this is not tested.

To build and install the package, you need the conda package manager. If you don't have conda installed, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).

The package source code is hosted [on GitHub](https://github.com/earth-system-radiation/pyRTE-RRTMGP) and uses git submodules to include the [RTE+RRTMGP Fortran software](https://earth-system-radiation.github.io/rte-rrtmgp/). The easiest way to install pyRTE-RRTMGP is to use `git`. You can install git from [here](https://git-scm.com/downloads).

## Installation with conda (recommended)

1. Clone the repository:

    ```bash
    git clone git@github.com:earth-system-radiation/pyRTE-RRTMGP.git
    ```

    or

    ```bash
    git clone https://github.com/earth-system-radiation/pyRTE-RRTMGP.git
    ```

    After cloning the repository, enter the repository directory:

    ```bash
    cd pyRTE-RRTMGP
    ```

2. Update the submodules:

    ```bash
    git submodule update --init --recursive
    ```

3. Make sure you have conda installed. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).
    To make sure your conda setup is working, run the command below:

    ```bash
    conda --version
    ```

    If this runs without errors, you are good to go.

4. Install the conda build requirements (if you haven't already):

    ```bash
    conda install conda-build conda-verify
    ```

5. Build the conda package locally:

    ```bash
    conda build conda.recipe
    ```

6. Install the package in your current conda environment:

    ```bash
    conda install -c ${CONDA_PREFIX}/conda-bld/ pyrte
    ```

    Note: This will install the package in your current conda environment. If you want to install the package in a different environment, activate your environment before running the `conda install` command above.

## Installation with pip

You also have the option to build and install the package with pip. However, this is not recommended as it requires you to have a working Fortran compiler and other prerequisites installed on your system.

To install with pip, you first need to clone the repo (``git clone git@github.com:earth-system-radiation/pyRTE-RRTMGP.git``) and update the submodules (``git submodule update --init --recursive``) as described in the conda installation instructions (above)[#installation-with-conda-recommended].

Then, make sure you have the necessary dependencies installed:

```bash

Then, make sure you have the necessary dependencies installed:

```bash
sudo apt install -y \
    libnetcdff-dev \
    gfortran-10 \
    python3-dev \
    cmake
```

Next, run pip to compile the Fortran code, build, and install the package:

``` bash
pip install -e .
```

Once built, the module will be located in a folder called `pyrte`.
