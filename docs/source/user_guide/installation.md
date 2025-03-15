(installation)=
# Installing pyRTE-RRTMGP

This section contains instructions for installing pyRTE-RRTMGP. On Linux (x86_64) platforms, you can install the package with `conda`. For other platforms, you need to build the package from source.

```{warning}
This project is a work in progress. The Python API is not yet stable and is subject to change.

See the [Contributor Guide](../contributor_guide/contribute.md) for information on how to contribute to this effort.
```

## Installation with Conda (recommended)

```{include} ../../../README.md
:start-after: <!-- start-installation-section -->
:end-before: <!-- end-installation-section -->
```

For **platforms other than Linux for x64 processors**, you can {ref}`install the package with pip <install-pip>` or build it from source using the instructions in the {ref}`local-install`.

(install-pip)=
## Installation with pip

You also have the option to build and install the package with ``pip``. This should work with macOS and Linux systems but requires you to install the system dependencies manually before installing with ``pip``.

Before installing the package, you should **create a virtual environment** to avoid conflicts with other packages. You can create a virtual environment in a folder of your choice using the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then, follow the instructions below for your respective platform to **install the system dependencies**:

* Debian/Ubuntu: On Debian/Ubuntu systems, you can use a tool like `apt` to install the dependencies: ``sudo apt install build-essential gfortran cmake git``
* Other Linux distributions: Install the dependencies using the package manager of your distribution.
* Mac OS: On MacOS systems you can use a tool like `brew` to install the dependencies: ``brew install git gcc cmake``

After installing the system dependencies, you can install the package directly from the git repository:

```bash
pip install git+https://github.com/earth-system-radiation/pyRTE-RRTMGP
```

## Local installation for development

To build and test the package locally for development purposes, follow the instructions in {ref}`local-install`.
