# Snopt.jl

[Snopt.jl](https://github.com/EllissoideRotondo/Snopt.jl) is a wrapper for
the [SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) sparse nonlinear
optimizer.

## License

`Snopt.jl` is licensed under the [MIT License](LICENSE).

The underlying library is a closed-source commercial product for which
you must [purchase a license](https://ccom.ucsd.edu/~optimizers/solvers/snopt/).

## Installation

`Snopt.jl` requires a SNOPT shared library (`libsnopt7.so` on Linux,
`libsnopt7.dylib` on macOS, `libsnopt7.dll` on Windows). SNOPT binaries are
not distributed with this package.

Set the `SNOPTDIR` environment variable to the directory containing the SNOPT
shared library, then install the package:

```julia
import Pkg
Pkg.add(url="https://github.com/EllissoideRotondo/Snopt.jl")
using Snopt
Snopt.has_snopt()  # true if the library was found
```

On Linux, `LD_LIBRARY_PATH` can be used in place of `SNOPTDIR`:

```
export LD_LIBRARY_PATH=/path/to/snopt:$LD_LIBRARY_PATH
```

## Usage

This package is intended to be used as a solver backend through OptimizationSnopt,
an [Optimization.jl](https://github.com/SciML/Optimization.jl) interface
that is currently under development. Direct use of the low-level C wrapper is
not recommended.

## Platform support

**Linux** and **macOS** should work out of the box with the precompiled SNOPT library.

**Windows** requires recompiling the SNOPT source with
[MinGW](https://www.mingw-w64.org/) to produce a compatible DLL. If
recompiling the library is not an option, [WSL](https://learn.microsoft.com/en-us/windows/wsl/)
is a viable alternative.

## Acknowledgements

This package draws from the following projects:
- [snopt/SNOPT7.jl](https://github.com/snopt/SNOPT7.jl)
- [byuflowlab/Snopt.jl](https://github.com/byuflowlab/Snopt.jl)
- [Yuricst/joptimise](https://github.com/Yuricst/joptimise)
