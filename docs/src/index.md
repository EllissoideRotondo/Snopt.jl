```@meta
CurrentModule = SNOPT
```

# SNOPT.jl

[SNOPT.jl](https://github.com/EllissoideRotondo/SNOPT.jl) is a Julia wrapper for
[SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) (Sparse Nonlinear
OPTimizer), a sequential quadratic programming (SQP) solver for smooth,
large-scale, sparsely constrained nonlinear optimization problems of the form

```math
\min_{x \in \mathbb{R}^n} \; f(x)
\quad \text{subject to} \quad
l \le \begin{pmatrix} x \\ c(x) \end{pmatrix} \le u,
```

where ``f`` and the constraint functions ``c`` are smooth and may be nonlinear.

The package exposes all three SNOPT Fortran entry points — `snOptA`, `snOptB`,
and `snOptC` — through Julia callbacks, and adds a single high-level [`snopt`](@ref)
function that covers the common case (objective + gradient, optional sparse
nonlinear constraints).

!!! note "Commercial solver required"
    SNOPT itself is closed-source commercial software. You must obtain a
    [SNOPT license](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) and a built
    `libsnopt7` shared library separately; it is **not** bundled with this package.
    See [Installation](@ref).

## Quick start

```julia
using SNOPT

result = snopt(
    x -> (x[1] - 1)^2 + (x[2] - 2)^2,                          # objective f(x)
    (g, x) -> (g[1] = 2(x[1]-1); g[2] = 2(x[2]-2); nothing),   # gradient g!(g, x)
    [0.0, 0.0];                                                # starting point
    lb = -10.0, ub = 10.0,
    options = ["Major print level" => 0],
)

result.status_symbol   # :Solve_Succeeded
result.x               # ≈ [1.0, 2.0]
result.objective       # ≈ 0.0
```

## Where to go next

- [Installation](@ref) — providing `libsnopt7` and verifying the setup.
- [High-level interface](@ref) — the `snopt` entry point in detail.
- [Low-level interface](@ref) — driving `snOptA`/`snOptB`/`snOptC` directly.
- [Examples](@ref) — fully worked constrained and unconstrained problems.
- [Optimization.jl integration](@ref) — using SNOPT through the SciML stack.
- [API reference](@ref) — every exported symbol.

## Relationship to Optimization.jl

For modeling-first workflows, SNOPT is intended to be used through
[Optimization.jl](https://github.com/SciML/Optimization.jl) via the
[OptimizationSNOPT.jl](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl)
adapter, which handles automatic differentiation and problem assembly. That
adapter is a **work in progress**; see [Optimization.jl integration](@ref).
</content>
