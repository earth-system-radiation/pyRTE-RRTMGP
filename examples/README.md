# pyRTE user documentation

This directory/book contains examples highlighting the use of pyRTE. They are the best place 
to learn about using pyRTE. Installation instructions and detailed technical documentaton 
are available on [ReadTheDocs](https://pyrte-rrtmgp.readthedocs.io/en/latest/).

The examples are written as annotated Python files in `py:percent` format. 
They are run as part of continuous integrations so should always be consistent with underlyng code. 

Conda environments with the dependencies needed to run the notebooks are defined in the 
`example-notebooks*.yml` files. `example-notebooks-scripts` installs `pyRTE` from the current code base; 
`example-notebooks-interactive` installs the latest `pyRTE` release on conda. 

To create notebooks for exploring interactively convert the `*.py` files to Jupyter notebooks with 
```
cd examples
jupytext --to ipynb **/*.py
```

Files in the `examples/` directory are also coallated in a 
Jupyterbook published on Github. 