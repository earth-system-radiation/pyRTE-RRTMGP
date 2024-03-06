from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'pybind_interface',
        ['pybind_interface.cpp'],
        include_dirs=[pybind11.get_include()],
        libraries=['test'],  # This should match the name of your compiled C++ library
        library_dirs=['./build'],  # This should point to the directory where libtest.so is located
        extra_link_args=['-Wl,-rpath=./build'],  # This ensures that the shared library can be found during runtime
    ),
]

setup(
    name='pybind_interface',
    ext_modules=ext_modules,
    script_args=['build_ext', '--build-lib', './build'],  # Specify the build directory here
)