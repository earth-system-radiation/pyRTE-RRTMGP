.. pyRTE-RRTMGP documentation master file, created by
   sphinx-quickstart on Thu Feb 29 10:39:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyRTE-RRTMGP's documentation!
========================================

pyRTE-RRTMGP provides a **Python interface** to the
`RTE+RRTMGP <https://earth-system-radiation.github.io/rte-rrtmgp/>`_ Fortran software package.
for computing radiative fluxes
in planetary atmospheres. RTE+RRTMGP is described in a
`paper <https://doi.org/10.1029/2019MS001621>`_ in
`Journal of Advances in Modeling Earth Systems <http://james.agu.org/>`_.


This package uses `pybind11 <https://github.com/pybind/pybind11>`_ to create a Python interface to a subset of the RTE+RRTMGP functions available in Fortran.

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   how_to/installation
   explanation/usage
   tutorials/tutorials

.. toctree::
   :maxdepth: 2
   :caption:  Reference:

   reference/pyrte_rrtmgp_python_modules
   reference/low_level_kernel_modules

.. toctree::
   :maxdepth: 2
   :caption: Contributing:

   contributor_guide/contribute
   how_to/installation-local-dev
   contributor_guide/fortran-compatibility



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
