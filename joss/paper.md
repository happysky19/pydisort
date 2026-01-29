---
title: "Pydisort: A Modern Python Package for Parallelized Discrete Ordinates Radiative Transfer (DISORT)"
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

An endeavor to overcome the limitations of the original Fortran implementation with even better efficiency is the `cdisort` library, which is a C reimplementation of DISORT[@buras2011new], and it has been an integral component of the `libRadtran` radiative transfer software package [@emde2016libradtran]. It preserves its stable, well-tested numerical core while adding portability and ease of embedding in modern workflows. A key advancement over the Fortran version is the adoption of dynamic memory allocation, eliminating the need for compile-time parameters. It's a 100% double-precision package that steers away from the occasional numerical instabilities caused by the mixed-precision calculations in the original Fortran code. Additionally, for large problem sizes, `cdisort` achieves over a 20× speedup through an improved intensity correction method and by leveraging the operating system’s background memory-zeroing instead of the explicit nested loop structure used in Fortran DISORT for initializing multidimensional arrays.

The `cdisort` library, while powerful, remains a single-threaded, low-level implementation that lacks a modern interface to exploit today’s parallel-computing advances — GPU acceleration, heterogeneous architectures, or seamless integration with machine-learning frameworks. More critically, `cdisort` is not a GitHub-hosted easy-to-contribute library and is not easily accessible to the broader scientific community, particularly those more comfortable with Python than C and eager to utilize modern tools like `PyTorch`.

To provide the radiative transfer community with a truly modern, high-performance, and user-friendly solution, we developed `Pydisort`, a `pip`-installable, compile-free Python package that wraps the `cdisort` library with support for parallel processing and compatibility with machine learning frameworks via `PyTorch`. `Pydisort` is designed for ease of use, efficiency, and flexibility, enabling users to perform radiative transfer calculations in a plane-parallel atmosphere with minimal effort. It supports batch processing, allowing efficient computation of radiative properties across multiple atmospheric layers and wavelengths in a single invocation.

# Software Design

`Pydisort` is designed as a modular, layered software system that bridges a high-performance radiative transfer backend with modern, Python-centric scientific workflows. The core numerical solver is provided by the `cdisort` library, which serves as the computational backend and supports dynamic memory allocation at runtime. This choice enables flexible problem sizes while retaining the numerical robustness of the established DISORT implementation.

To facilitate efficient memory management and future-proof the codebase for heterogeneous computing, `Pydisort` adopts `PyTorch` [@paszke2019pytorch] tensors as its primary data structure at the user interface level. Using tensors allows seamless integration with `PyTorch`’s automatic parallelization, GPU acceleration, and machine-learning ecosystem, while maintaining compatibility with CPU-only environments.

An intermediate C++ layer is introduced between the Python interface and the C backend. This layer encapsulates the raw `cdisort` data structures and is responsible for pre-processing inputs and post-processing outputs. By centralizing this logic, the design avoids exposing users to long, error-prone parameter lists and isolates backend-specific details from the public API. This intermediate layer also enables reuse of the backend functionality in non-Python contexts.

The Python bindings are implemented using `pybind11` [@jakob2024pybind11], a modern, header-only C++ library that provides robust type conversion, memory ownership semantics, and exception handling. Compared to traditional approaches such as `f2py`, `pybind11` offers greater flexibility and maintainability for mixed C++/Python codebases.

The build system is designed to produce shared C and C++ libraries that link against both `libtorch` and Python, allowing the package to be distributed as a standard `pip`-installable PyPI package ([`pydisort`](https://pypi.org/project/pydisort/)). Prebuilt binary wheels are provided for Linux and macOS platforms. The build and distribution process is fully automated using `cibuildwheel`, targeting Linux systems with `glibc` version 2.28 or newer, which is the minimum required by PyTorch v2.7 and later.

To ensure ABI compatibility with upstream PyTorch binaries, the build system dynamically determines the appropriate `CXX11_ABI` setting from the linked `libtorch` distribution. For `libtorch` v2.7, this corresponds to `CXX11_ABI=1` on Linux and `CXX11_ABI=0` on macOS.

`Pydisort` provides two user-facing interfaces: a Python API and a C++ API. The Python interface is intended for interactive use, scripting, and integration with machine-learning workflows, while the C++ interface supports embedding `Pydisort` directly into larger C or C++ simulation frameworks.

Continuous integration (CI) and continuous distribution (CD) are handled through GitHub Actions, enabling automated testing, building, and release of binary wheels with minimal manual intervention. The Python codebase follows the PEP 8 style guide [@van2001pep] and adopts a Python-first design philosophy, making extensive use of keyword arguments and named parameters to provide a clear, idiomatic user experience.

# Statement of need

Tools for calculating atmospheric radiative transfer are essential for a wide range of applications in atmospheric modeling and remote sensing. As demonstrated in the [Summary](#Summary) section, the original `Fortran` implementation of DISORT has been widely used in research. To accommodate the growing popularity of `Python` in the scientific community, many groups have developed their own Python wrappers for the original Fortran implementation of DISORT. However, few have reached the same level of maturity, efficiency, and usability as our package.

Our package manages to distribute `pip`-installable pre-built library that wraps the DISORT algorithm implemented in a lower-level compiled language. Our package

1. saves users from the hassle of setting up the build environment and the tool chains;
2. skips the compilation process because we ship the compiled shared libraries with the package on PyPI;
3. achieves at least the same level of performance as the original `C/Fortran` implementation on the same hardware, with the extended capability of GPU acceleration in the future;
4. provides a modern `Python` interface that is easy to use and understand;
5. offers comprehensive documentation and user support to facilitate adoption and integration into existing workflows.

The previous effort of educational purpose has been made by @ho2024pythonicdisort, which is a pure `Python` implementation of the DISORT algorithm and @ricchiazzi1998sbdart, which is a `Fortran77` implementation. Since our backend subroutines are implemented in `C`, our code runs at least as fast as the original `Fortran` implementation in single-threaded mode, while the `Python` interface lowers the barrier to entry for users who are not familiar with `Fortran` or `C`.

Additionally, `Pydisort` improves upon the `C/Fortran` implementations by enabling parallelization over wavelengths and columns. For plane-parallel atmospheres, the radiative transfer equation is separable by columns and by wavelengths/wavenumbers. However, looping over columns and wavelengths in `Python` will significantly slow down the code. Therefore, we leverage the `PyTorch` tensor data structure and its `TensorIterator` functionality to enable concurrent execution over columns and wavelengths. This design achieves strong scaling performance on modern multi-core CPUs. Future improvements of the package will seamlessly allow GPU acceleration of the radiative transfer calculation when the original `C` backend is adapted to execute on GPUs.

# Performance Evaluation

To quantify the efficiency of `Pydisort` relative to existing implementations, we benchmarked runtimes using a test driver based on Test Problem 9 (“General Emitting/Absorbing/Scattering”) from the DISORT suite (`diotest`), with increased computational workload. Specifically, the benchmark configuration includes 32 streams and 100 atmospheric layers, typical for atmospheric radiation calculations; all other parameters are kept consistent with the original test case. We increase the workload by solving the radiative transfer equations repetitively for multiple wavelengths or columns, representing a realistic application case in remote sensing and atmospheric modeling such that multiple spectral bands or atmospheric columns are processed simultaneously.

We evaluated `Pydisort` performance against two alternative implementations:

- **`cdisort`**: the original C reimplementation of DISORT, which serves as our baseline performance reference,
- **`PythonicDISORT`**: a pure-Python translation of DISORT, used to highlight the performance gap between interpreted and optimized implementations.

To assess parallel performance, we ran `Pydisort` in both single-threaded and multi-threaded modes using `PyTorch`'s internal parallelism configuration (`torch.set_num_threads`). Multi-threaded tests were conducted with 10 threads to illustrate scalability on modern CPU architectures. All tests were conducted on a MacBook Pro equipped with an Apple M1 Max chip (10-core CPU, peak clock speed of 3.22 GHz) and 64 GB of RAM.

![Runtime comparison of radiative transfer implementations as a function of the number of wavenumbers, evaluated on a fixed test problem (DISORT Test 09 with 32 streams and 100 layers). `cdisort` (gray) serves as the baseline C implementation. `PythonicDisort` (red) is a pure-Python reimplementation. `Pydisort` is shown in both single-threaded (blue, 1 core) and multi-threaded (green, 10 cores) modes, using `PyTorch`-based parallelism. Both axes use logarithmic scaling. The results demonstrate that `Pydisort` outperforms `cdisort` in runtime efficiency and exhibits strong scaling performance with increasing spectral resolution.](perf.png)

We did not benchmark the original Fortran DISORT implementation, as prior work by @buras2011new has demonstrated that the C version (`cdisort`) provides better runtime performance and is commonly used in research settings as part of the `libRadtran` package [@emde2016libradtran].

Figure 1 summarizes the results, showing runtime as a function of the number of wavenumbers. Both axes use logarithmic scaling to highlight relative performance trends across a wide range of spectral resolutions. As shown, `Pydisort` is at least as efficient as `cdisort` when using a single core, and outperforms `cdisort` in multi-threaded mode by an order of magnitude when the workload increases, demonstrating the effectiveness of `PyTorch`'s parallelism for this problem. The pure-Python implementation (`PythonicDISORT`) is significantly slower, highlighting the performance benefits of using a compiled backend. Additionally, `Pydisort` shows strong scaling performance, demonstrating its suitability for high-throughput radiative transfer workloads.

# Research Impact Statement

`Pydisort` solves the radiative transfer equation in an efficient and user-friendly manner, addressing a long-standing need in atmospheric science, climate modeling, and planetary science for a modern, high-performance DISORT-based tool. It has been adopted as a core computation component within a growing ecosystem of research projects, supporting studies that span terrestrial atmospheres, giant planets, exoplanets, finite volum models for compressible fluids, coupled chemistry-thermodynamics frameworks, and physically based rendering in vision and imaging applications. It's currently used in dozens of GitHub repositories across multiple institutions, for both research and educational purposes.

By integrating with `PyTorch`’s tensor and parallel computing infrastructure, `Pydisort` can be deployed in modern high-performance computing environments, enabling large-scale simulations and high-throughput data analysis that were difficult to achieve with traditional DISORT implementations. Its native compatibility with machine-learning frameworks like `PyTorch` enables researchers to integrate radiative transfer calculations into data-driven models. Adoption metrics further demonstrates `Pydisort`'s impact. The PyPI public dataset shows a marked increase in downloads following the transition to `PyTorch`-based infrastructure in March 2025.

![The PyPI download statistics show an increasing trend in adoption following the introduction of `PyTorch` acceleration, indicating growing interest from the community. Charts generated via ClickPy.](pydisort_stats.png)

As of January 2026, `Pydisort` has been downloaded over 192,000 times since its initial release in 2023, placing it within the top 10% of all Python packages by download counts. The user base spans over 140 countries, reflecting its global reach and impact.

![Geographical distribution of `Pydisort` downloads, indicating a global user base with significant adoption in North America, Europe, and Asia. Chart generated via ClickPy.](pypi_by_country.png)

Together, these trends reflect the growing recognition of `Pydisort` as a reliable, scalable, and versatile tool for radiative transfer calculations that lowers technical barriers while enabling new classes of scientific inquiry.

# Acknowledgements

We acknowledge Dr. Timothy E. Dowling for his work on migrating the original FORTRAN version of DISORT to C, which is the basis for our implementation. We acknowledge Dr. Xi Zhang and Dr. Tianhao Le for initiating the project and testing the code. We also thank Andrew Ryan for early testing and feedback on the package.

# AI Usage Disclosure

No generative AI tools were used in the development of this software, the writing of this manuscript, or the preparation of supporting materials.

# References
