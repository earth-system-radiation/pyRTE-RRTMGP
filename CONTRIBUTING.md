# Contributing to pyRTE-RRTMGP

Thanks for considering making a contribution to pyRTE-RRTMGP!

## Did you find a bug?

Please file an issue on the [Github page](https://github.com/earth-system-radiation/pyRTE-RRTMTP/issues).
If possible, your issue should include a [minimal, complete, and verifiable example](https://stackoverflow.com/help/mcve) of the bug.

## Would you like to propose a new feature?

Please file an issue on the [Github page](https://github.com/earth-system-radiation/pyRTE-RRTMTP/issues).

## Would you like to set up a local development environment?

Please follow the instructions for [installing pyRTE-RRTMTP with pip in the documentation](https://pyrte-rrtmgp.readthedocs.io/en/latest/user_guide/installation.html).

## Would you like to contribute to the documentation?

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

## Did you write a patch that fixes a bug?

Please fork this repository, branch from `develop`, make your changes, and open a
Github [pull request](https://github.com/earth-system-radiation/pyRTE-RRTMTP/pulls)
against the `develop` branch.

## Did you add functionality?

Please fork this repository, branch from `develop`, make your changes, and open a
Github [pull request](https://github.com/earth-system-radiation/pyRTE-RRTMTP/pulls)
against the `develop` branch.
Pull Requests for new features should include tests and documentation.
