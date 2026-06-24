# SNOPT.jl

[![CI](https://github.com/EllissoideRotondo/SNOPT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EllissoideRotondo/SNOPT.jl/actions/workflows/CI.yml)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://EllissoideRotondo.github.io/SNOPT.jl/stable)

[SNOPT.jl](https://github.com/EllissoideRotondo/SNOPT.jl) is a Julia wrapper for
[SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/), the sparse nonlinear
optimizer for large-scale constrained problems. It exposes SNOPT's `snOptA`,
`snOptB`, and `snOptC` interfaces through Julia callbacks, plus a single
high-level `snopt` entry point for the common case.

## License

`SNOPT.jl` is licensed under the [MIT License](LICENSE). The underlying solver is
a closed-source commercial product for which you must
[purchase a license](https://ccom.ucsd.edu/~optimizers/solvers/snopt/); its
binaries are **not** distributed with this package.

## Installation

`SNOPT.jl` needs a SNOPT shared library (`libsnopt7.so` on Linux,
`libsnopt7.dylib` on macOS, `libsnopt7.dll` on Windows). Set the `SNOPTDIR`
environment variable to the directory containing it, then add the package:

```julia
import Pkg
Pkg.add("SNOPT")

using SNOPT
SNOPT.has_snopt()   # true once the library is found
```

On Linux and macOS the platform library-path variables work in place of `SNOPTDIR`:

```bash
export LD_LIBRARY_PATH=/path/to/snopt:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/path/to/snopt:$DYLD_LIBRARY_PATH   # macOS
```

If the library is not found, the package still loads; `has_snopt()` returns
`false` and solves raise an informative error.

## Usage

For most modeling workflows, use SNOPT through
[Optimization.jl](https://github.com/SciML/Optimization.jl) via the
[OptimizationSNOPT.jl](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl)
adapter (**work in progress**). `SNOPT.jl` itself provides a compact low-level API
for driving SNOPT directly with Julia callbacks.

The main entry point is `snopt`, which solves a problem through SNOPT's `snOptB`
interface. You supply an objective `f(x)`, its gradient `g!(g, x)`, and a starting
point:

```julia
using SNOPT

result = snopt(
    x -> (x[1] - 1)^2 + (x[2] - 2)^2,                          # objective
    (g, x) -> (g[1] = 2(x[1]-1); g[2] = 2(x[2]-2); nothing),   # gradient
    [0.0, 0.0];
    lb = -10.0, ub = 10.0,
    options = ["Major print level" => 0, :minor_print_level => 0],
)

result.status          # SNOPT inform code
result.status_symbol   # e.g. :Solve_Succeeded
result.objective       # final objective value
result.x               # solution vector
```

Key points of the low-level interface:

- **Constraints.** Pass `eval_con`, `eval_jac`, `lcon`, `ucon`, and an optional
  sparse Jacobian sparsity pattern `J` (a `SparseMatrixCSC`). `eval_jac(jnz, x)`
  fills the Jacobian nonzeros in `J`'s column-major order; if `J` is omitted, a
  dense pattern is assumed. The solve workspace is sized automatically from SNOPT's
  own `snMemB` estimator, exposed as `snmemb`.
- **Options.** A vector of pairs whose keys are strings or symbols (symbol
  underscores become spaces, so `:major_print_level => 0` equals
  `"Major print level" => 0`). Options can also be read from a specs file with
  `read_options`.
- **Monitoring.** `snlog` receives a `SnoptMajorLog` per major iteration (counters,
  objective, infeasibilities, the current point); the lower-level `callback` keyword
  fires on each objective/constraint evaluation. Returning `false` from either
  requests early termination.

```julia
result = snopt(f, g!, x0;
    options = ["Major print level" => 1],
    snlog = event -> (println("major $(event.major_iter): f = $(event.objective)"); true),
)
```

Beyond `snopt`, the package exports the `snOptA`/`snOptB`/`snOptC` problem types
(`SnoptA`, `SnoptB`/`SnoptProblem`, `SnoptC`), their in-place solvers (`snopta!`,
`snoptb!`, `snoptc!`, `snopt!`), workspace management (`initialize`, `set_option!`,
`snmemb`), and callback builders (`make_objfun`, `make_confun`, `make_usrfun_a`,
`make_usrfun_c`, `make_snlog`). See the
[documentation](https://EllissoideRotondo.github.io/SNOPT.jl/stable) and the
[`examples/`](examples) directory (`hs71.jl`, `unconstrained.jl`) for full worked
problems.

## Platform support

**Linux** and **macOS** work out of the box with a compatible `libsnopt7`.

**Windows** requires a `libsnopt7.dll` built from the SNOPT source with
[MinGW](https://www.mingw-w64.org/) (the Intel-compiled distribution is not
ABI-compatible). If recompiling is not an option,
[WSL](https://learn.microsoft.com/en-us/windows/wsl/) is a working alternative.

## Acknowledgements

This package draws on prior Julia SNOPT wrappers:

- [snopt/SNOPT7.jl](https://github.com/snopt/SNOPT7.jl)
- [byuflowlab/Snopt.jl](https://github.com/byuflowlab/Snopt.jl)
- [Yuricst/joptimise](https://github.com/Yuricst/joptimise)
</content>
