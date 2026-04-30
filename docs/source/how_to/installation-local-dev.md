# Installing a Local Development Environment

(local-install)=
## How to install (for developers)

Building and testing the package locally requires [mamba](https://mamba.readthedocs.io/) on your system.

Building pyRTE-RRTMGP locally **currently works with Linux (Ubuntu) and macOS (Intel/ARM) only**.

Go set up a local development environment:

1. **Clone the pyRTE GitHub repository**:

    ```bash
    git clone https://github.com/earth-system-radiation/pyRTE-RRTMGP.git
    cd pyRTE-RRTMGP
    ```

2.  **Install the software dependencies**
    ```bash
    mamba env create -f dev-environment.yml
    ```

    will, when executed from the root directory of the local repo, install pyRTE's dependencies and the package itself in editable mode, which means that any changes you make to the package will be reflected in the documentation build right after you make them, without needing to reinstall the package.


### How to Set up Pre-Commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to maintain consistent code formatting (using [flake8](https://flake8.pycqa.org/en/latest/), [isort](https://pycqa.github.io/isort/), and [black](https://black.readthedocs.io/en/stable/)) and run static type checking with [mypy](https://github.com/python/mypy) before each commit.

Install the pre-commit hooks from the the root directory of your local clone of the repository:

```bash
pre-commit install
```

After installing pre-commit, you can also run the checks manually with `pre-commit run --all-files`.

Please ensure that new contributions pass the formatting and mypy checks before submitting pull requests.

### How to Run and Update Tests

pyRTE-RRTMTP uses [pytest](https://docs.pytest.org/) for testing. To run the tests, first install the required testing dependencies (optimally in a dedicated virtual environment):

```bash
mamba env create -f dev-environment.yml
mamba activate pyrte_dev
```

Then, run the tests:

```bash
pytest tests
```

New features should be accompanied with new tests.
