```@meta
CurrentModule = SNOPT
```

# Optimization.jl integration

For modeling-first workflows, `SNOPT.jl` is intended to be used through
[Optimization.jl](https://github.com/SciML/Optimization.jl) (the SciML unified
optimization interface) via the
[OptimizationSNOPT.jl](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl)
adapter. The adapter lets SNOPT consume an `OptimizationProblem` directly, with
gradients and constraint Jacobians supplied by Optimization.jl's automatic
differentiation backends instead of hand-written callbacks.

!!! warning "Work in progress"
    OptimizationSNOPT.jl is under active development and not yet registered. The
    interface described here may change. For stable, fully supported usage today,
    call [`snopt`](@ref) and the [low-level interface](@ref Low-level-interface)
    directly.

## Intended usage

The adapter exposes a `SnoptOptimizer` algorithm that plugs into the standard
`solve(prob, alg; ...)` workflow:

```julia
using OptimizationBase, OptimizationSNOPT

# objective and (optional) constraints assembled the usual Optimization.jl way,
# with an AD backend providing derivatives:
optf = OptimizationFunction(f, AutoForwardDiff(); cons = cons)
prob = OptimizationProblem(optf, x0; lb = lb, ub = ub, lcons = lcons, ucons = ucons)

opt = SnoptOptimizer(
    major_iterations_limit = 2000,
    major_optimality_tolerance = 1e-8,
    additional_options = Dict("Linesearch tolerance" => 0.9),
)

sol = solve(prob, opt; maxiters = 500, abstol = 1e-8, verbose = Val(true))
```

`SnoptOptimizer` surfaces the most common SNOPT controls as keyword arguments
(print levels, iteration limits, optimality/feasibility tolerances, derivative and
Hessian options) and accepts any other SNOPT option through `additional_options`.
The SciML common solver arguments (`maxiters`, `abstol`, `reltol`, `verbose`,
`store_trace`) are mapped onto the corresponding SNOPT options.

## How it relates to this package

`OptimizationSNOPT.jl` depends on `SNOPT.jl`: it translates an
`OptimizationProblem` into the callbacks and sparse structures that the
[low-level interface](@ref Low-level-interface) expects, then drives a solve. Anything
expressible through the adapter is also expressible by calling [`snopt`](@ref)
directly — the adapter mainly removes the bookkeeping of writing gradient and
Jacobian callbacks by hand.

Refer to the OptimizationSNOPT.jl repository for its current status, installation,
and the authoritative description of `SnoptOptimizer`.
</content>
