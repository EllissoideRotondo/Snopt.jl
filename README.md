# SNOPT.jl

[![CI](https://github.com/EllissoideRotondo/SNOPT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EllissoideRotondo/SNOPT.jl/actions/workflows/CI.yml)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://EllissoideRotondo.github.io/SNOPT.jl/stable/)

SNOPT.jl is an unofficial Julia interface to
[SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/). SNOPT solves large,
constrained nonlinear optimization problems.

Use this package for direct access to SNOPT's `snOptA`, `snOptB`, and `snOptC`
interfaces. Use
[OptimizationSNOPT.jl](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl)
for Optimization.jl problems and automatic differentiation.

## Requirements

- Julia 1.10 or later.
- A licensed SNOPT 7 shared library.
- SNOPT's `snopt-interface` C functions in that library.

SNOPT.jl does not include SNOPT or a SNOPT license.

## Install

Install the registered package when it is available in your registry:

```julia
import Pkg
Pkg.add("SNOPT")
```

Install the current source version with:

```julia
import Pkg
Pkg.develop(url = "https://github.com/EllissoideRotondo/SNOPT.jl")
```

Set `SNOPTDIR` to the directory that contains `libsnopt7`:

```bash
export SNOPTDIR=/path/to/snopt/lib
```

Windows PowerShell uses this command:

```powershell
$env:SNOPTDIR = "C:\path\to\snopt\lib"
```

Verify the complete setup:

```bash
julia -e 'using SNOPT; @assert SNOPT.has_snopt(); println(SNOPT.libsnopt7)'
```

See the [installation guide](https://EllissoideRotondo.github.io/SNOPT.jl/stable/installation/)
for library names, search paths, licensing, and platform limits.

## Quick start

The high-level [`snopt`](https://EllissoideRotondo.github.io/SNOPT.jl/stable/interface/)
function manages the workspace and returns a `SnoptResult`.

```julia
using SNOPT

objective(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2

function gradient!(gradient, x)
    gradient[1] = 2.0 * (x[1] - 1.0)
    gradient[2] = 2.0 * (x[2] - 2.0)
    return nothing
end

result = snopt(
    objective,
    gradient!,
    [0.0, 0.0];
    lb = -10.0,
    ub = 10.0,
    options = ["Major print level" => 0],
)

result.status_symbol
result.x
result.objective
```

The gradient callback must fill every entry of `gradient`. It may return any
value because SNOPT uses the mutated array.

## Run the examples

Run these commands from the repository root:

```bash
julia --project=. examples/unconstrained.jl
julia --project=. examples/hs71.jl
```

The second example includes bounds, constraints, and a constraint Jacobian.
A Jacobian is the matrix of constraint derivatives.

## Main interfaces

| Need | Interface |
| --- | --- |
| Managed workspace and split callbacks | `snopt` |
| Separate objective and constraint callbacks | `SnoptB` |
| One combined callback | `SnoptC` |
| Stacked rows and separate derivative structure | `SnoptA` |

The [documentation](https://EllissoideRotondo.github.io/SNOPT.jl/stable/)
contains callback contracts, warm starts, monitoring, and low-level examples.

## Concurrency

SNOPT owns one global Fortran workspace per process. SNOPT.jl serializes
workspace creation and solves. Threaded solves are safe, but run sequentially.

Use separate Julia processes for parallel solves.

## Test

Run the test suite from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Solver tests run when Julia finds `libsnopt7`. Library discovery tests remain
available without it.

## Platform support

- Linux is tested with a compatible `libsnopt7.so`.
- Intel macOS is expected to work, but is not tested by the maintainers.
- Apple Silicon is not currently supported.
- Windows requires a MinGW-built `libsnopt7.dll`.

The vendor's Intel-built Windows library is not compatible with this package.
Use the MinGW build or Windows Subsystem for Linux.

## License

SNOPT.jl uses the MIT License. SNOPT is a separate commercial product.
See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md).

## Acknowledgements

This package draws on prior Julia wrappers:

- [snopt/SNOPT7.jl](https://github.com/snopt/SNOPT7.jl)
- [byuflowlab/Snopt.jl](https://github.com/byuflowlab/Snopt.jl)
- [Yuricst/joptimise](https://github.com/Yuricst/joptimise)
