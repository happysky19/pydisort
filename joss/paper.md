---
title: 'Pydisort: A Modern Python Interface for Discrete Ordinates Radiative Transfer (DISORT)'
tags:
  - Python
  - C/C++
  - atmospheric science
  - radiative transfer
  - planetary atmosphere
authors:
  - name: Zhiying Hu
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Cheng Li
    orcid: 0000-0002-1954-098X
    affiliation: 1
affiliations:
 - name: University of Michigan, Ann Arbor, MI, USA
   index: 1
date: 1 May 2025
bibliography: paper.bib
---

# Summary

DISORT (Discrete Ordinate Radiative Transfer) is a widely-used algorithm that calculates the scattering and absorption of radiation in a plane-parallel medium.
The original DISORT algorithm was implemented in `Fortran` [@stamnes1988numerically] with static memory allocation. The dimension of the internal arrays, such as the number of atmospheric layers and the number of radiation streams must be specified at compile time.
Despite of this limitation, the original `Fortran` program has been widely used in atmospheric and remote sensing applications.

Since `Python` has become the main stream programming language for scientific computing, over the years, many attempts have been made to wrap the original `Fortran` code in `Python` to provide a more user-friendly interface.
Notable examples include (1) [chanGimeno's](https://github.com/chanGimeno/pyDISORT),
(2) [SeregaOsipov's](https://github.com/SeregaOsipov/pyDISORT),
(3) [danielkoll's](https://github.com/danielkoll/PyDISORT3), and
(4) [mjwolf's](https://github.com/mjwolff/pyDISORT).
All of them are named ``pydisort`` and are based on the original `Fortran` code.
Their `Python` interfaces are enabled by `f2py` [@van2011numpy] functionality, which is a part of the `NumPy` package and provides a convenient way to call `Fortran` subroutines from `Python`.

Independently, [@ho2024pythonicdisort] developed a pure `Python` implementation of the DISORT algorithm, which is less efficient than the `Fortran` implementation but is easier to use for pedagogical and exploratory purposes.

Here, we provide a `pip`-installable modern `Python` interface to the original DISORT implementation.
Instead of starting from the original `Fortran` code, we use the `C` implementation of DISORT by [@buras2011new] as the computing backend.
The `cdisort` library improves upon the original `Fortran` code by using dynamic memory allocation, allowing for a more flexible implementation and easier adaptation to `Python`.
`cdisort` has been an integral component of the `libRadtran` radiative transfer software package [@emde2016libradtran],
but is lesser known in the atmospheric science community than the original `Fortran` implementation because the previous generation of atmospheric models were mostly written in `Fortran`.


Our `Pydisort` package differs from the previous `pydisort` packages in the following ways:

1. We use the `C` implementation of DISORT as the backend, which allows for dynamic memory allocation during runtime.
2. We use `Pytorch` [@paszke2019pytorch]'s ``tensor`` data structure for memory management, which paves the road for heterogeneous computing and automatic parallelization.
3. We create an intermediate `C++` interface to the backend `C`-library, which handles pre- and post-processing of the raw `cdisort` data structures.
4. We leverage `Pybind11` [@jakob2024pybind11] to interface between `Python` and our intermediate `C++` interface.
`Pybind11` is a header-only modern alternative to `f2py` with graceful type-handling and memory management.
5. We design software architecture to support building the C/C++ backend libraries as shared libraries, linking them to `libtorch` and the `Python` interface, and distributing them as `pip`-installable packages for various platforms, including `Linux` and `MacOS`.
6. We provide two frontends: (1) a `Python` interface and (2) a `C++` interface.
The former is useful for users who want to use the package in `Python` and take advantage of the machine learning capabilities enabled by `Pytorch`, while the latter is useful for users who want to integrate the package into their own `C/C++` packages.
7. We automate the Continuous Integration (CI) and Continuous Distribution (CD) processes using `GitHub Actions` to ensure that minimal human effort is required for maintaining the package.
8. We adhere to the `PEP 8` [@van2001pep] style guide for `Python` code, making the program a `Python-first` experience.
The function calls make frequent use of `Python` features such as keyword arguments and named arguments, which are idiomatic to `Python` users.

``Pydisort`` is designed to be used by both planetary and earth science researchers and by educators or students in radiative transfer courses.
The previous effort of educational purpose has been made by [@ho2024pythonicdisort], which is a pure `Python` implementation of the DISORT algorithm and [@richardson2023radiative] which is a `Fortran77` implementation.
Since our backend subroutines are implemented in `C`, our code runs as fast as the original `Fortran` implementation, while the `Python` interface lowers the barrier to entry for users who are not familiar with `Fortran` or `C`.


Additionally, ``Pydisort`` improves upon the previous implementations by enabling parallelization over wavelengths and columns. For plane-parallel atmospheres, the radiative transfer equation is separable by columns and by wavelengths/wavenumbers.
However, looping over columns and wavelengths in python will significantly slow down the code.
Therefore, we leverage the `Pytorch` tensor data structure and its ``TensorIterator`` functionality to enable parallelization over columns and wavelengths.
Future improvements of the package will allow GPU acceleration of the
radiative transfer calculation when the original `C` backend is adapted to execute on GPUs.

# Acknowledgements

We acknowledge Dr. Timothy E. Dowling for his work on migrating the original FORTRAN version of DISORT to C, which is the basis for our implementation.
We acknowledge Dr. Xi Zhang and Dr. Tianhao Le for initiating the project and testing the code.

# References
