# 1. Overview of the Project

## DISORT in Fortran, C, C++, Python

You can find the detailed overview of the DISORT project, including information about disort, cdisort, cppdisort, and pydisort in the [README - Overview of the DISORT project in Fortran, C, C++, Python.md](./README%20-%20Overview%20of%20the%20DISORT%20project%20in%20Fortran%2C%20C%2C%20C%2B%2B%2C%20Python.md) file.


## Major Branches

There are three formal branches in this repo: 
- `cdisort_patches`
- `cppdisort`
- `pydisort`

The `cdisort_patches` branch contains the original `cdisort` library (v2.1.3) by Timothy E. Dowling, plus some patch files containing modifications made by [Cheng Li](https://chengcli.io/). Different from `cdisort`, which uses `Makefile` to build the library, we modified the configuration and adapted a `CMake`-built approach. For detailed changes, please see the `README.md` file in the `cdisort_patches` branch.

The `cppdisort` branch provides a C++ wrapper for the "cdisort" library, allowing easy access to its functionality from C++ code. We use toml++ for configuration management, allowing users to specify various parameters in the TOML configuration file. The updated implementation ensures compatibility with modern C++ standards and incorporates bug fixes and enhancements compared to the original cdisort library. For detailed changes, please see the `README.md` file in the `cppdisort` branch.

The `pydisort` branch builds a Python library that provides a Pythonic interface to the cppdisort library. It serves as a bridge between the C++ implementation of cppdisort and the Python programming language, enabling users to leverage the power of cppdisort within their Python applications. For detailed changes, please see the `README.md` file in the `pydisort` branch.



# 2. This Branch: a CMake-built version of `cdisort`

This branch is a CMake-built version of the popular `cdisort` program for solving the radiative transfer equation by Timothy E. Dowling.

The source code of `cdisort` locate in the `src` folder, and the test code in `tests`. You could also find a helper cmake file `setup_compiler_flags.cmake` in the `cmake` folder, together with the original makefile `Makefile_cdisort` for reference and comparison purposes. 


## Why CMake?

CMake uses a script to generate a set of makefiles, which can then be used to compile the code. CMake is platform independent and can generate makefiles for different build systems. As we try to build up more functionalities on the `cdisort` project, a more scalable tool like CMake is more suitable compared to Makefiles. In addition, CMake is easier to maintain and can work across multiple platforms, be used with different compilers and build systems, and it can automatically detect the correct settings to use for each platform. 


## Differences: CMake vs Makefile

The main differences between the CMake-built version and the Makefile-built version of the cdisort project are:

1. Build system: CMake is a cross-platform build system generator, while Make is a build automation tool for Unix-like systems.

2. Project structure: The directory structure of the CMake-built version is more complex than that of the Makefile-built version. A CMake-built version requires a CMakeLists.txt file in the root directory, a cmake directory with configuration files, and separate CMakeLists.txt files for each subdirectory that contains source files.

3. Building process: With Make, the building process is initiated by running the make command, while with CMake, the building process is initiated by running the cmake command to generate a Makefile, and then running the make command.

4. Compiler flags: The Makefile-built version specifies compiler flags directly in the Makefile, while the CMake-built version uses a separate setup_compiler_flags.cmake file to specify compiler flags.

5. Library creation: In the Makefile-built version, the library is created using the ar and ranlib commands. In the CMake-built version, the add_library command is used.

6. Unit testing: The CMake-built version includes a separate CMakeLists.txt file for unit testing, while the Makefile-built version does not.


## Build

```bash
mkdir build
cd build
cmake ..
make
```
