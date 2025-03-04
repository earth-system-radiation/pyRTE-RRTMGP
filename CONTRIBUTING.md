# Contributing to pyRTE-RRTMGP

Thanks for considering making a contribution to pyRTE-RRTMGP!

## How to Report a Bug?

Please file a bug report on the [GitHub page](https://github.com/earth-system-radiation/pyRTE-RRTMGP/issues/new/choose).
If possible, your issue should include a [minimal, complete, and verifiable example](https://stackoverflow.com/help/mcve) of the bug.

## How to Propose a New Feature?

Please file a feature request on the [GitHub page](https://github.com/earth-system-radiation/pyRTE-RRTMGP/issues/new/choose).

## How to Set up a Local Development Environment?

Please follow the instructions for [installing pyRTE-RRTMTP with pip or conda in the documentation](https://pyrte-rrtmgp.readthedocs.io/en/latest/user_guide/installation.html).

### Static Type Checking
This project employs [mypy](https://github.com/python/mypy) for static type checking to improve code reliability and maintainability. Mypy is integrated into our continuous integration workflow, and you can also run it locally using:

```bash
pip install mypy
mypy .
```
Please ensure that new contributions pass the mypy checks before submitting pull requests.

Static type checking is run as part of `pre-commit`, the use of which we encourage. 


## How to Contribute to the Documentation?

The documentation uses [Sphinx](https://www.sphinx-doc.org/en/master/) and is located in the `docs` directory.

To build the documentation locally, first install the required documentation dependencies (optimally in a dedicated virtual environment):

```bash
pip install -r docs/requirements-doc.txt
```

Then, build the documentation:

```bash
cd docs
make html
```

The built documentation will be located in `docs/build/html`.

## How to Run and Update Tests?

pyRTE-RRTMTP uses [pytest](https://docs.pytest.org/en/stable/) for testing. To run the tests, first install the required testing dependencies (optimally in a dedicated virtual environment):

```bash
pip install -r tests/requirements-test.txt
```

Then, run the tests:

```bash
pytest tests
```

## How to Contribute a Patch That Fixes a Bug?

Please fork this repository, branch from `main`, make your changes, and open a
GitHub [pull request](https://github.com/earth-system-radiation/pyRTE-RRTMTP/pulls)
against the `main` branch.

## How to Contribute New Features?

Please fork this repository, branch from `main`, make your changes, and open a
GitHub [pull request](https://github.com/earth-system-radiation/pyRTE-RRTMTP/pulls)
against the `main` branch.

Pull Requests for new features should include tests and documentation.

## How to Make a New Release?

For maintainers:

1. To make a new release, update the version number in `pyproject.toml` and `conda.recipe/meta.yaml`. Also update `CITATION.cff` as necessary.
2. Then, create a new release off the `main` branch on GitHub, using the "Draft a new release" button in [https://github.com/earth-system-radiation/pyRTE-RRTMGP/releases](https://github.com/earth-system-radiation/pyRTE-RRTMGP/releases).
3. Create a new tag with the version number, and use the "Generate release notes" button to create the release notes.
4. Review and update the the release notes as necessary, and publish the release and set it as the latest release.

The documentation on [https://pyrte-rrtmgp.readthedocs.io/](https://pyrte-rrtmgp.readthedocs.io/) will update automatically.
