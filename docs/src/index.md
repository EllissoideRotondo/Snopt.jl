```@meta
CurrentModule = SNOPT
```

# SNOPT.jl

SNOPT.jl is an unofficial Julia interface to
[SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/). SNOPT solves smooth,
constrained nonlinear optimization problems.

The package provides three low-level interfaces. It also provides [`snopt`](@ref)
as the main Julia entry point.

!!! note "Commercial solver required"
    Obtain a SNOPT license and a compatible `libsnopt7` shared library.
    The Julia package does not include either item.

## Choose an interface

| Need | Interface |
| --- | --- |
| Managed workspace and split callbacks | [`snopt`](@ref) |
| Separate objective and constraint callbacks | [`SnoptB`](@ref) |
| One combined callback | [`SnoptC`](@ref) |
| Stacked rows and separate derivative structure | [`SnoptA`](@ref) |

Use
[OptimizationSNOPT.jl](https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/stable/)
for Optimization.jl problems and automatic differentiation.

## Quick start

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

## Continue

1. Follow [Installation](@ref).
2. Read the [High-level interface](@ref).
3. Run the [Examples](@ref).
4. Use the [Low-level interface](@ref) only when needed.
5. Check the [API reference](@ref) for exact signatures.

## Concurrency

SNOPT owns one global Fortran workspace per process. SNOPT.jl serializes
workspace creation and solves. Threaded solves are safe, but run sequentially.

Use separate Julia processes for parallel solves.
