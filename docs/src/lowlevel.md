```@meta
CurrentModule = SNOPT
```

# Low-level interface

Use the low-level interface only when [`snopt`](@ref) cannot represent the
problem. Examples include linear objective rows and hot starts.

The low-level types mirror SNOPT's three Fortran entry points.

## Workspace lifecycle

[`initialize`](@ref) creates a [`SnoptWorkspace`](@ref SNOPT.SnoptWorkspace).
The workspace owns SNOPT's work arrays and active Fortran session.

Always close a manually managed workspace:

```julia
workspace = initialize("", "")
try
    set_option!(workspace, "Major print level", 0)
    # Build and solve a low-level problem here.
finally
    close(workspace)
end
```

Prefer the `do` form because it closes the workspace after errors:

```julia
initialize("", "") do workspace
    set_option!(workspace, "Major print level", 0)
    # Build and solve a low-level problem here.
end
```

SNOPT owns one active workspace per process. Calling `initialize` closes the
previous active workspace. All workspace operations and solves are serialized.

Do not create several workspaces for parallel solves. Use separate Julia
processes instead.

## Workspace size

Each work array must contain at least 500 elements. `initialize` enforces this
SNOPT requirement.

For a known problem, use [`snmemb`](@ref) to ask SNOPT for minimum sizes:

```julia
memory = snmemb(m, n, neJ, negCon, nnCon, nnObj, nnJac)
workspace = initialize("", "", memory.miniw, memory.minrw)
```

| Name | Meaning |
| --- | --- |
| `m` | Total rows passed to SNOPT. |
| `n` | Design variables. |
| `neJ` | Stored Jacobian entries. |
| `negCon` | Nonlinear constraint derivatives. |
| `nnCon` | Nonlinear constraints. |
| `nnObj` | Variables in the nonlinear objective. |
| `nnJac` | Variables in the nonlinear constraint Jacobian. |

The high-level [`snopt`](@ref) function performs this estimate automatically.

## Set options

[`set_option!`](@ref) calls SNOPT's native option functions.

```julia
set_option!(workspace, "Major iterations limit", 250)
set_option!(workspace, "Major optimality tolerance", 1.0e-8)
set_option!(workspace, "Hessian limited memory")
```

The function requires an open workspace. Invalid options raise an
`ArgumentError`. Use [`read_options`](@ref) to read a specs file.

## Problem types

| Type | SNOPT entry | Callback shape |
| --- | --- | --- |
| [`SnoptB`](@ref) | `snOptB` | Separate objective and constraints. |
| [`SnoptC`](@ref) | `snOptC` | Combined objective and constraints. |
| [`SnoptA`](@ref) | `snOptA` | Stacked row vector and derivative structure. |

All three types extend [`AbstractSnoptProblem`](@ref). [`snopt!`](@ref)
dispatches to the matching in-place solver.

After solving, the problem contains its status, multipliers, and final point.
`SnoptB` and `SnoptC` also store `obj_val`.

## Callback builders

The builders adapt Julia functions to SNOPT's C callback signatures.

| Builder | Julia contract |
| --- | --- |
| [`make_objfun`](@ref) | `eval_obj(x)` and `eval_grad(gradient, x)` |
| [`make_confun`](@ref) | `eval_con(values, x)` and `eval_jac(nonzeros, x)` |
| [`make_dummy_confun`](@ref) | No constraints. |
| [`make_usrfun_c`](@ref) | Combined objective and constraints. |
| [`make_usrfun_a`](@ref) | `eval_F(F, x)` and optional `eval_G(G, x)`. |
| [`make_snlog`](@ref) | Major-iteration progress events. |
| [`make_snstop`](@ref) | Major-iteration stop events. |

Mutation callbacks must fill every requested entry. Array lengths and storage
order must match the problem definition.

Problem-evaluation builders accept an optional `callback` keyword. It receives
an event after each evaluation. Return `false` to request a stop.

## Minimal `SnoptB` construction

This example shows the required extended arrays. SNOPT appends one slack
variable per row. A slack converts a constraint row into a bounded variable.

```julia
using SNOPT
using SparseArrays

initialize("", "") do workspace
    set_option!(workspace, "Major print level", 0)

    n = 2
    rows = 1

    objective = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
    gradient! = (gradient, x) -> begin
        gradient[1] = 2.0 * (x[1] - 1.0)
        gradient[2] = 2.0 * (x[2] - 2.0)
        return nothing
    end

    objfun = make_objfun(objective, gradient!, workspace.iw)
    confun = make_dummy_confun()

    x = [0.0, 0.0, 0.0]
    lower = [-10.0, -10.0, -1.0e20]
    upper = [10.0, 10.0, 1.0e20]
    states = zeros(Int32, n + rows)
    J = SparseMatrixCSC{Float64, Int32}(
        1,
        n,
        Int32[1, 2, 2],
        Int32[1],
        [0.0],
    )

    problem = SnoptB(
        workspace,
        n,
        0,
        rows,
        n,
        x,
        lower,
        upper,
        states,
        J,
        0.0,
        0,
        Float64[],
        objfun,
        confun,
    )

    snoptb!(problem)
    problem.status, problem.obj_val, problem.x[1:n]
end
```

Prefer [`snopt`](@ref) when it supports the problem. It provides validation,
automatic sizing, and simpler result handling.
