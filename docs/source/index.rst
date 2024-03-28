.. pyRTE-RRTMGP documentation master file, created by
   sphinx-quickstart on Thu Feb 29 10:39:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyRTE-RRTMGP's documentation!
========================================

pyRTE-RRTMGP provides a **Python interface** to the
`RTE+RRTMGP <https://earth-system-radiation.github.io/rte-rrtmgp/>`_ Fortran software package.

This package uses [pybind11](https://github.com/pybind/pybind11) to create a Python interface to a subset of the RTE+RRTMGP functions available in Fortran.

The RTE+RRTMGP package is a set of libraries for for computing radiative fluxes
in planetary atmospheres. RTE+RRTMGP is described in a
`paper <https://doi.org/10.1029/2019MS001621>`_ in
`Journal of Advances in Modeling Earth Systems <http://james.agu.org/>`_.


.. note::
   This project is currently in an early stage of development.

.. toctree::
   :maxdepth: 2
   :caption: Using pyRTE-RRTMGP:

   user_guide/installation
   user_guide/usage
   reference_guide/pyrte_rrtmgp

.. toctree::
   :maxdepth: 2
   :caption: Contributing to pyRTE-RRTMGP:

   contributor_guide/contribute
   contributor_guide/roadmap



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


