(tutorials)=
# pyRTE-RRTMGP Tutorials

The pyRTE-RRTMGP repository contains a number of tutorials and examples in the form of  [Jupyter notebooks](https://docs.jupyter.org). These notebooks are located in the `examples` folder of the repository.

The tutorials are designed to help you get started with using pyRTE-RRTMGP and to demonstrate how to use the package.

## Setting up the Tutorial Notebooks

Follow these steps to run the tutorial notebooks:

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone https://github.com/earth-system-radiation/pyRTE-RRTMGP.git
    ```

2. **Create a virtual environment** with the pyRTE-RRTMGP package and notebook dependencies:

    After cloning the repository, enter the root directory to install the software needed to run the notebooks:
    ```bash
    cd pyRTE-RRTMGP
    mamba env create -f example-notebooks-interactive.yml
    mamba activate pyrte_notebooks
    ```

3. **Start the Jupyter notebook server**:

    ```bash
    jupyter notebook
    ```

    This will open a new tab in your web browser with the Jupyter notebook interface. You can navigate to the `examples` folder and open the tutorial notebooks from there.

<!--
Note about including interactive notebooks in the documentation:
- We can't include interactive notebooks in the documentation directly because they require a running Jupyter server to work. readthedocs doesn't support running Jupyter notebooks interactively.
- We could include links to run the notebooks on Google Colab (``[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/your-notebook.ipynb)``). However, this requires setting up the environments with the required packages (potentially with something like https://pypi.org/project/condacolab/).
 -->
