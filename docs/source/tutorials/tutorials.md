(tutorials)=
# pyRTE-RRTMGP Tutorials

The pyRTE-RRTMGP repository contains a number of tutorials in the form of  [Jupyter notebooks](https://docs.jupyter.org). These notebooks are located in the `examples` folder of the repository.

Currently, the following tutorials are available in directory `examples/`:

* `sw_example.ipynb`: How to use pyRTE-RRTMGP to compute clear-sky shortwave fluxes.
* `lw_example.ipynb`: How to use pyRTE-RRTMGP to compute clear-sky longwave fluxes.
* `dask_example.ipynb`: How to use pyRTE-RRTMGP with [Dask](https://docs.dask.org/en/stable/) to compute clear-sky fluxes with parallel computing (the example is trivally small).
* `all_sky_example.ipynb`: How to use pyRTE-RRTMGP to compute both longwave and shortwave fluxes for an idealized all-sky (gases and clouds) scenario.
* `dyamond_clouds/example.ipynb`: Using a very large (global 3-km) snapshot from the [NASA/GMAO contribution to the DYAMOND 2 project](https://gmao.gsfc.nasa.gov/global_mesoscale/dyamond_phaseII/data_access/) together with Dask to compute cloud optical properties.

The tutorials are designed to help you get started with using pyRTE-RRTMGP and to demonstrate how to use the package.

## Setting up the Tutorial Notebooks

Follow these steps to run the tutorial notebooks:

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone https://github.com/earth-system-radiation/pyRTE-RRTMGP.git
    ```

2. **Go to the ``pyRTE-RRTMGP`` directory**

    After cloning the repository, enter the root directory: 
    ```bash
    cd pyRTE-RRTMGP
    ```

3. **Create a virtual environment** with the pyRTE-RRTMGP package and notebook dependencies:

    ```bash
    conda env create -f example-notebooks-interactive.yml
    conda activate pyrte_notebooks 
    ```

    ```{note}
    This will create an environment named `pyrte_notebooks` that you need to activate. 
    ```

    See {ref}`installation` for more information on how to install the package.


4. **Start the Jupyter notebook server**:

    ```bash
    jupyter notebook
    ```

    This will open a new tab in your web browser with the Jupyter notebook interface. You can navigate to the `examples` folder and open the tutorial notebooks from there.


## Using the Tutorials

Once you have opened a tutorial notebook in Jupyter, you can run the cells in the notebook by pressing `Shift + Enter`. You can also use the "Run" button in the toolbar at the top of the notebook.

See the [Jupyter documentation](https://docs.jupyter.org) for more information on how to use Jupyter notebooks.

<!--
Note about including interactive notebooks in the documentation:
- We can't include interactive notebooks in the documentation directly because they require a running Jupyter server to work. readthedocs doesn't support running Jupyter notebooks interactively.
- We could include links to run the notebooks on Google Colab (``[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/your-notebook.ipynb)``). However, this requires setting up the environments with the required packages (potentially with something like https://pypi.org/project/condacolab/).
 -->
