---
title: "Pydisort: A Modern Python Interface for Discrete Ordinates Radiative Transfer (DISORT)"
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

Radiative transfer is the study of how electromagnetic radiation travels through and interacts with a medium—such as Earth’s atmosphere, ocean waters, or a planetary atmosphere. It is a fundamental process that governs how light and thermal radiation are absorbed, scattered, and emitted by gases, aerosols, clouds, and surfaces. Understanding radiative transfer is crucial for interpreting remote sensing data from satellites and spacecrafts, modeling climate systems, and studying planetary atmospheres.

DISORT (Discrete Ordinates Radiative Transfer) is a widely-used algorithm that abstracts the medium as a plane-parallel structure, with the horizontal properties uniform but possibly varying vertically. It solves the radiative transfer equation using the discrete ordinates method, which discretizes the angular domain into a finite number of directions and solves the resulting system of equations. Among the three main components of radiative transfer — absorption, scattering, and emission — scattering is often the most complex and computationally intensive to model, as it involves integral from all incoming angles to all outgoing angles. In a realistic atmosphere with variable scattering, absorption, multiple layers, and possibly multiple wavelengths, in cases where the radiance is sensitive to small-scale details, or when it's necessary to obtain high accuracy, the scalability and efficiency of the radiative transfer model become critical.

The original DISORT algorithm was implemented in Fortran [@stamnes1988numerically]. It has been widely used in Earth and planetary atmospheres [@clough2005atmospheric; @li2018high; @tan2021atmospheric; @komacek2022patchy; @lee2024testing; @zhang2015aerosol]. However, the original Fortran implementation is limited by static memory allocation, which requires the user to specify the number of atmospheric layers and radiation streams at compile time. This limitation persists even with efforts to wrap the original Fortran code via `f2py` for more user-friendly access (e.g. [chanGimeno's](https://github.com/chanGimeno/pyDISORT), [SeregaOsipov's](https://github.com/SeregaOsipov/pyDISORT), [danielkoll's](https://github.com/danielkoll/PyDISORT3), [mjwolff's](https://github.com/mjwolff/pyDISORT)). Another approach is to implement the DISORT algorithm in pure Python, which solves the problem of static memory allocation but is less efficient than the Fortran implementation [@ho2024pythonicdisort].

An endeavor to overcome the limitations of the original Fortran implementation with even better efficiency is the `cdisort` library, which is a C reimplementation of DISORT[@buras2011new], and it has been an integral component of the libRadtran radiative transfer software package [@emde2016libradtran]. It preserves its stable, well-tested numerical core while adding portability and ease of embedding in modern workflows. A key advancement over the Fortran version is the adoption of dynamic memory allocation, eliminating the need for compile-time parameters. It's a 100% double precision package that steers away from the occasional numerical instabilities caused by the mixed-precision calculations in the original Fortran code. Additionally, for large problem sizes, `cdisort` achieves over a 20× speedup through an improved intensity correction method and by leveraging the operating system’s background memory-zeroing instead of the explicit nested loop structure used in Fortran DISORT for initializing multidimensional arrays.

The `cdisort` library, while powerful, remains a single-threaded, low-level implementation that lacks a modern interface to exploit today’s parallel-computing advances — GPU acceleration, heterogeneous architectures, or seamless integration with machine-learning frameworks. More critically, `cdisort` is not a GitHub-hosted easy-to-contribute library and is not easily accessible to the broader scientific community, particularly those more comfortable with Python than C and eager to utilize modern tools like `PyTorch`.

To provide the radiative transfer community with a truly modern, high-performance, and user-friendly solution, we developed `Pydisort`, a `pip`-installable, compile-free Python package that wraps the `cdisort` library with support for parallel processing and compatibility with machine learning frameworks via `PyTorch`. `Pydisort` is designed for ease of use, efficiency, and flexibility, enabling users to perform radiative transfer calculations in a plane-parallel atmosphere with minimal effort. It supports batch processing, allowing efficient computation of radiative properties across multiple atmospheric layers and wavelengths in a single invocation.

Specifically, `Pydisort` provides the following features:

1. We use the `cdisort` library as the backend, which allows for dynamic memory allocation during runtime.
2. We use `PyTorch` [@paszke2019pytorch]'s `tensor` data structure for memory management, paving the way for heterogeneous computing and automatic parallelization.
3. We create an intermediate `C++` layer to the backend `C`-library. The intermediate layer handles pre- and post-processing of the raw `cdisort` data structures, eliminating the need to feed long parameter lists to the backend library.
4. We leverage `Pybind11` [@jakob2024pybind11] to interface between `Python` API calls and our intermediate `C++` interface. `Pybind11` is a header-only modern alternative to `f2py` with graceful type-handling, casting and memory management.
5. We design software architecture to support building the C/C++ backend libraries as shared libraries, linking them to `libtorch` and `Python`, and distributing them as a `pip`-installable PyPI package `[pydisort](https://pypi.org/project/pydisort/)` for various platforms, including `Linux` and `MacOS`.
6. We automate the building and distribution process using `cibuildwheel` on Mac images and Linux images with `glibc 2.28+`, which is the minimum version of `glibc` required by `PyTorch v2.7+`.
7. We dynamically determine the `CXX11_ABI` version from the upstream `libtorch` library. Currently with `libtorch v2.7`, the `CXX11_ABI` version is `1` for Linux distributions and `0` for MacOS distributions.
8. We provide two frontends: (1) a `Python` interface and (2) a `C++` interface. The former is useful for users who want to use the package in `Python` and take advantage of the machine learning capabilities enabled by `PyTorch`, while the latter is useful for users who want to integrate the package into their own `C/C++` packages.
9. We automate the Continuous Integration (CI) and Continuous Distribution (CD) processes using `GitHub Actions` to ensure that minimal human effort is required for developers contributing to and maintaining the package.
10. We adhere to the `PEP 8` [@van2001pep] style guide for `Python` code, making the program a `Python-first` experience. The function calls make frequent use of `Python` features such as keyword arguments and named arguments, which are idiomatic to `Python` users.

# Statement of need

Tools for calculating atmospheric radiative transfer are essential for a wide range of applications in atmospheric modeling and remote sensing. As demonstrated in the [Summary](#Summary) section, the original `Fortran` implementation of DISORT has been widely used in research. To accommodate the growing popularity of `Python` in the scientific community, many groups have developed their own Python wrappers for the original Fortran implementation of DISORT. However, few have reached the same level of maturity, efficiency, and usability as our package.

Our package manages to distribute `pip`-installable pre-built library that wraps the DISORT algorithm implemented in a lower-level compiled language such as `C` or `Fortran`. Our package

1. saves users from the hassle of setting up the build environment and the tool chains;
2. skips the compilation process because we ship the compiled shared libraries with the package on PyPI;
3. achieves at least the same level of performance as the original `C/Fortran` implementation on the same hardware, with the extended capability of GPU acceleration in the future;
4. provides a modern `Python` interface that is easy to use and understand;
5. offers comprehensive documentation and user support to facilitate adoption and integration into existing workflows.

The previous effort of educational purpose has been made by @ho2024pythonicdisort, which is a pure `Python` implementation of the DISORT algorithm and @ricchiazzi1998sbdart, which is a `Fortran77` implementation. Since our backend subroutines are implemented in `C`, our code runs at least as fast as the original `Fortran` implementation, while the `Python` interface lowers the barrier to entry for users who are not familiar with `Fortran` or `C`.

Additionally, `Pydisort` improves upon the `C/Fortran` implementations by enabling parallelization over wavelengths and columns. For plane-parallel atmospheres, the radiative transfer equation is separable by columns and by wavelengths/wavenumbers. However, looping over columns and wavelengths in `Python` will significantly slow down the code. Therefore, we leverage the `PyTorch` tensor data structure and its `TensorIterator` functionality to enable concurrent execution over columns and wavelengths. Future improvements of the package will seamlessly allow GPU acceleration of the radiative transfer calculation when the original `C` backend is adapted to execute on GPUs.

# Acknowledgements

We acknowledge Dr. Timothy E. Dowling for his work on migrating the original FORTRAN version of DISORT to C, which is the basis for our implementation. We acknowledge Dr. Xi Zhang and Dr. Tianhao Le for initiating the project and testing the code. We also thank Andrew Ryan for early testing and feedback on the package.

# References
