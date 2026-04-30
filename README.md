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
Pkg.add("Snopt")
using Snopt
Snopt.has_snopt()  # true if the library was found
```

On Linux, `LD_LIBRARY_PATH` can be used in place of `SNOPTDIR`:

```
export LD_LIBRARY_PATH=/path/to/snopt:$LD_LIBRARY_PATH
```

## Usage

For most SciML workflows, use
[OptimizationSnopt.jl](https://github.com/EllissoideRotondo/OptimizationSnopt.jl)
through [Optimization.jl](https://github.com/SciML/Optimization.jl). `Snopt.jl`
also provides a small low-level API for direct use with Julia callbacks.

```julia
using Snopt

result = snopt(
    x -> (x[1] - 1)^2 + (x[2] - 2)^2,
    (g, x) -> begin
        g[1] = 2(x[1] - 1)
        g[2] = 2(x[2] - 2)
    end,
    [0.0, 0.0];
    lb = -10.0,
    ub = 10.0,
    options = [
        "Major print level" => 0,
        "Minor print level" => 0,
    ],
)

result.status          # SNOPT inform code
result.status_symbol   # symbolic status
result.objective
result.x
```

Constrained problems can provide nonlinear constraint callbacks and a sparse
Jacobian sparsity pattern `J`. If `J` is omitted, a dense pattern is assumed.
The `snopt` API calls SNOPT's own `snMemB` memory estimator through the exported
`f_snmem` wrapper before allocating the solve workspace.
Pass solver options as a vector of pairs. Keys may be strings or symbols; symbol
underscores are converted to spaces, so `:major_print_level => 0` is equivalent
to `"Major print level" => 0`.

For manual workspace sizing, call `snmemb` directly:

```julia
memory = snmemb(m, n, neJ, negCon, nnCon, nnJac, nnObj)
memory.miniw, memory.minrw
```

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
