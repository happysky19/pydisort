---
title: 'Pydisort: A Modern Python Interface for Discrete Ordinates Radiative Transfer (DISORT)'
tags:
  - Python
  - C/C++
  - atmospheric science
  - radiative transfer
  - planetary atmosphere
authors:
  - name: Zoey Hu
    orcid: 0009-0007-1511-3269
    affiliation: 1
  - name: Cheng Li
    orcid: 0000-0002-8280-3119
    affiliation: 1
affiliations:
 - name: University of Michigan, Ann Arbor, MI, USA
   index: 1
date: 1 May 2025
bibliography: paper.bib
---

# Summary

Radiative transfer in layered media is central to atmospheric and planetary science. DISORT (Discrete Ordinate Radiative Transfer) is a widely-used algorithm that calculates the scattering and absorption of radiation in a plane-parallel medium.
The original DISORT algorithm was implemented in `Fortran` [@stamnes1988numerically] with static memory allocation. The dimension of the internal arrays, such as the number of atmospheric layers and the number of radiation streams must be specified at compile time.
Despite of this limitation, the original `Fortran` program has been widely used in Earth and planetary atmospheres (e.g. [@clough2005atmospheric; @li2018high; @tan2021atmospheric; @komacek2022patchy; @lee2024testing; @zhang2015aerosol]).

With the increasing adoption of `Python` in scientific computing, over the years, many efforts have been made to wrap the original `Fortran` code in `Python` to provide a more user-friendly interface.
Notable examples include (1) [chanGimeno's](https://github.com/chanGimeno/pyDISORT),
(2) [SeregaOsipov's](https://github.com/SeregaOsipov/pyDISORT),
(3) [danielkoll's](https://github.com/danielkoll/PyDISORT3) [@koll2019hot], and
(4) [mjwolf's](https://github.com/mjwolff/pyDISORT).
All of them are named `pydisort` and are based on the original `Fortran` code.
Their `Python` interfaces are enabled by `f2py` [@van2011numpy] functionality, which is a part of the `NumPy` package and provides a convenient way to call `Fortran` subroutines from `Python`.
A specific implementation of `pydisort` for near-IR radiative transfer in Titan's atmosphere is provided through the [atmosphere](https://github.com/adamkovics/atmosphere) package by @adamkovics2016meridional.

Independently, @ho2024pythonicdisort developed a pure `Python` implementation of the DISORT algorithm, which is less efficient than the `Fortran` implementation but is easier to use for pedagogical and exploratory purposes.

Here, we provide a `pip`-installable modern `Python` interface to the original DISORT implementation.
Instead of starting from the original `Fortran` code, we use the `C` implementation of DISORT by @buras2011new as the computing backend.
The `cdisort` library improves upon the original `Fortran` code by using dynamic memory allocation, allowing for a more flexible implementation and easier adaptation to `Python`.
`cdisort` has been an integral component of the `libRadtran` radiative transfer software package [@emde2016libradtran],
but is lesser known in the atmospheric science community than the original `Fortran` implementation because the previous generation of atmospheric models were mostly written in `Fortran`.


Our `Pydisort` package differs from the previous `pydisort` packages in the following ways:

1. We use the `C` implementation of DISORT as the backend, which allows for dynamic memory allocation during runtime.
2. We use `PyTorch` [@paszke2019pytorch]'s ``tensor`` data structure for memory management, paving the way for heterogeneous computing and automatic parallelization.
3. We create an intermediate `C++` layer to the backend `C`-library. The intermediate layer handles pre- and post-processing of the raw `cdisort` data structures.
4. We leverage `Pybind11` [@jakob2024pybind11] to interface between `Python` API calls and our intermediate `C++` interface.
`Pybind11` is a header-only modern alternative to `f2py` with graceful type-handling, casting and memory management.
5. We design software architecture to support building the C/C++ backend libraries as shared libraries, linking them to `libtorch` and `Python`, and distributing them as `pip`-installable packages for various platforms, including `Linux` and `MacOS`.
6. We automate the building and distribution process using `cibuildwheel` on Mac images and Linux images with `glibc 2.28+`, which is the minimum version of `glibc` required by `PyTorch v2.7+`.
7. We dynamically determine the `CXX11_ABI` version from the upstream `libtorch`
library. Currently with `libtorch v2.7`, the `CXX11_ABI` version is `1` for Linux distributions and `0` for MacOS distributions.
8. We provide two frontends: (1) a `Python` interface and (2) a `C++` interface.
The former is useful for users who want to use the package in `Python` and take advantage of the machine learning capabilities enabled by `PyTorch`, while the latter is useful for users who want to integrate the package into their own `C/C++` packages.
9. We automate the Continuous Integration (CI) and Continuous Distribution (CD) processes using `GitHub Actions` to ensure that minimal human effort is required for maintaining the package.
10. We adhere to the `PEP 8` [@van2001pep] style guide for `Python` code, making the program a `Python-first` experience.
The function calls make frequent use of `Python` features such as keyword arguments and named arguments, which are idiomatic to `Python` users.


# Statement of need

Tools for calculating atmospheric radiative transfer are essential for a wide range of applications in atmospheric modeling and remote sensing.
As demonstrated in the [Summary](#Summary) section, the original `Fortran` implementation of DISORT has been widely used in research.
To accommodate the growing popularity of `Python` in the scientific community, many groups have developed their own Python wrappers
for the original Fortran implementation of DISORT. However, few have reached the same level of maturity, efficiency, and usability as our package.

Our package manages to distribute `pip`-installable pre-built library that wraps the DISORT algorithm
implemented in a lower-level compiled language such as `C` or `Fortran`.
Our package

1. saves users from the hassle of setting up the build environment and the tool chains.
2. skips the compilation process because we ship the compiled shared libraries with the package on PyPI.
3. achieves the same level of performance as the original `C/Fortran` implementation.
4. provides a modern `Python` interface that is easy to use and understand.

The previous effort of educational purpose has been made by @ho2024pythonicdisort, which is a pure `Python` implementation of the DISORT algorithm and @ricchiazzi1998sbdart, which is a `Fortran77` implementation.
Since our backend subroutines are implemented in `C`, our code runs as fast as the original `Fortran` implementation, while the `Python` interface lowers the barrier to entry for users who are not familiar with `Fortran` or `C`.

Additionally, `Pydisort` improves upon the `C/Fortran` implementations by enabling parallelization over wavelengths and columns.
For plane-parallel atmospheres, the radiative transfer equation is separable by columns and by wavelengths/wavenumbers.
However, looping over columns and wavelengths in `Python` will significantly slow down the code.
Therefore, we leverage the `PyTorch` tensor data structure and its `TensorIterator` functionality to enable concurrent execution over columns and wavelengths.
Future improvements of the package will seamlessly allow GPU acceleration of the
radiative transfer calculation when the original `C` backend is adapted to execute on GPUs.

# Acknowledgements

We acknowledge Dr. Timothy E. Dowling for his work on migrating the original FORTRAN version of DISORT to C, which is the basis for our implementation.
We acknowledge Dr. Xi Zhang and Dr. Tianhao Le for initiating the project and testing the code.

# References
