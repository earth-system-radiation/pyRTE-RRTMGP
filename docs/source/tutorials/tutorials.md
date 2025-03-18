(tutorials)=
# pyRTE-RRTMGP Tutorials

The pyRTE-RRTMGP repository contains a number of tutorials in the form of  [Jupyter notebooks](https://docs.jupyter.org). These notebooks are located in the `examples` folder of the repository.

Currently, the following tutorials are available:

* `examples/sw_example.ipynb`: A tutorial on how to use pyRTE-RRTMGP to solve shortwave radiative transfer equations.
* `examples/lw_example.ipynb`: A tutorial on how to use pyRTE-RRTMGP to solve longwave radiative transfer equations.
* `examples/dask_example.ipynb`: A tutorial on how to use pyRTE-RRTMGP with [Dask](https://docs.dask.org/en/stable/) to solve radiative transfer equations with parallel computing.
* `examples/all_sky_example.ipynb`: A tutorial on how to use pyRTE-RRTMGP to solve radiative transfer equations for an all-sky scenario.

The tutorials are designed to help you get started with using pyRTE-RRTMGP and to demonstrate how to use the package.

## Setting up the Tutorial Notebooks

Follow these steps to run the tutorial notebooks:

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone https://github.com/earth-system-radiation/pyRTE-RRTMGP.git
    ```

2. **Go to the ``pyRTE-RRTMGP/examples`` directory**

    After cloning the repository, enter the examples directory in the repository:

    ```bash
    cd pyRTE-RRTMGP/examples
    ```

3. **Install the pyRTE-RRTMGP package** in your current conda environment (if you haven't already):

    ```bash
    conda install -c conda-forge pyrte_rrtmgp
    ```

    ```{note}
    This will install the package in your current conda environment. If you want to install the package in a different environment, activate your environment before running the `conda install` command above.
    ```

    See {ref}`installation` for more information on how to install the package.

4. **Install the Jupyter notebook package**:

    ```bash
    conda install jupyter
    ```

5. **Start the Jupyter notebook server**:

    ```bash
    jupyter notebook
    ```

    This will open a new tab in your web browser with the Jupyter notebook interface. You can navigate to the `examples` folder and open the tutorial notebooks from there.

```{note}
Some of the notebooks might require you to install additional dependencies into your environment. For example, the Dask notebook requires `dask` to be installed on your system.
```

## Using the Tutorials

Once you have opened a tutorial notebook in Jupyter, you can run the cells in the notebook by pressing `Shift + Enter`. You can also use the "Run" button in the toolbar at the top of the notebook.

See the [Jupyter documentation](https://docs.jupyter.org) for more information on how to use Jupyter notebooks.

<!--
Note about including interactive notebooks in the documentation:
- We can't include interactive notebooks in the documentation directly because they require a running Jupyter server to work. readthedocs doesn't support running Jupyter notebooks interactively.
- We could include links to run the notebooks on Google Colab (``[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/your-notebook.ipynb)``). However, this requires setting up the environments with the required packages (potentially with something like https://pypi.org/project/condacolab/).
 -->
