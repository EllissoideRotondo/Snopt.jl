```@meta
CurrentModule = SNOPT
```

# Low-level interface

Below [`snopt`](@ref) sits a thin layer that mirrors SNOPT's three Fortran entry
points. Use it when you need a problem shape the high-level function does not cover
(for example a linear objective row, warm/hot starts reusing a workspace, or
SNOPT's combined `snOptC` user function), or when you want fine control over the
workspace and options.

## Workspace lifecycle

Every solve runs against a [`SnoptWorkspace`](@ref SNOPT.SnoptWorkspace) created by
[`initialize`](@ref). The workspace owns SNOPT's integer/real work arrays and the
solver session; it must be closed to release the Fortran-side state.

```julia
ws = initialize("", "")          # default size (small/medium problems)
try
    set_option!(ws, "Major print level", 0)
    # ... build and solve a SnoptA/SnoptB/SnoptC problem against ws ...
finally
    close(ws)                    # calls SNOPT's snend; also runs as a finalizer
end
```

A `do`-block form handles the cleanup for you, including on error:

```julia
initialize("", "") do ws
    set_option!(ws, "Major print level", 0)
    # ... use ws ...
end
```

!!! warning "One active solve per process"
    SNOPT.jl supports one active SNOPT solve at a time per Julia process.
    Sequential solves are supported, but concurrent solves from multiple Julia
    threads are not. Creating multiple [`SnoptWorkspace`](@ref) objects does not
    make independent solver sessions. For parallel independent solves, use
    separate Julia processes.

    When [`initialize`](@ref) is called again, SNOPT.jl closes the previous active
    workspace before creating the new one. Use the high-level [`snopt`](@ref)
    entry point or an `initialize do` block unless you specifically need to
    manage a workspace yourself.

For larger problems, size the work arrays explicitly. A reasonable rule of thumb is

```julia
n  = 500    # design variables
nc = 200    # nonlinear constraints
leniw = 500 + 100 * (n + nc)
lenrw = 500 + 200 * (n + nc)
ws = initialize("", "", leniw, lenrw)
```

Each array must hold at least 500 elements (SNOPT's `sninit` minimum), which
`initialize` enforces.

## Setting options

[`set_option!`](@ref) wraps SNOPT's `snSet`/`snSeti`/`snSetr`:

```julia
set_option!(ws, "Major iterations limit", 250)   # integer value
set_option!(ws, "Major optimality tolerance", 1e-8)  # real value
set_option!(ws, "Hessian limited memory")        # keyword-only string
```

Alternatively, read an entire SNOPT specs file with [`read_options`](@ref); the
returned inform code is explained by [`specs_status_message`](@ref).

## Estimating workspace memory

The high-level path calls SNOPT's `snMemB` estimator automatically. To do it
yourself, call [`snmemb`](@ref) with the problem dimensions; it returns a
[`SnoptMemory`](@ref) carrying the minimum integer/real work-array lengths:

```julia
mem = snmemb(m, n, neJ, negCon, nnCon, nnObj, nnJac)
mem.miniw, mem.minrw
```

The dimensions are: `m` total constraints, `n` variables, `neJ` Jacobian nonzeros,
`negCon` nonlinear Jacobian nonzeros, `nnCon` nonlinear constraints, `nnObj`
nonlinear objective variables, and `nnJac` nonlinear Jacobian variables.

## Problem types

| Type | SNOPT entry | User function |
|------|-------------|---------------|
| [`SnoptB`](@ref) (= [`SnoptProblem`](@ref)) | `snOptB` | split objective + constraint callbacks |
| [`SnoptC`](@ref) | `snOptC` | one combined callback for objective + constraints |
| [`SnoptA`](@ref) | `snOptA` | one function returning a stacked `F` vector, separate linear/nonlinear derivative pattern |

All three are subtypes of [`AbstractSnoptProblem`](@ref) and are solved in place by
[`snopt!`](@ref), which dispatches to [`snopta!`](@ref), [`snoptb!`](@ref), or
[`snoptc!`](@ref). After a solve the problem's `status`, objective, multipliers, and
final point are populated.

## Building callbacks

The package provides builders that adapt ordinary Julia functions to the C
signatures SNOPT expects:

| Builder | For | Wraps |
|---------|-----|-------|
| [`make_objfun`](@ref) | `snOptB` | `eval_obj(x)`, `eval_grad(g, x)` |
| [`make_confun`](@ref) | `snOptB` | `eval_con(c, x)`, `eval_jac(jnz, x)` |
| [`make_dummy_confun`](@ref) | `snOptB` | no-op constraints for unconstrained problems |
| [`make_usrfun_c`](@ref) | `snOptC` | combined objective + constraint evaluation |
| [`make_usrfun_a`](@ref) | `snOptA` | `eval_F(F, x)` and optional `eval_G(G, x)` |
| [`make_snlog`](@ref) | `snOptB`/`snOptC` | a `snLog` hook delivering [`SnoptMajorLog`](@ref) events |

The problem-evaluating builders ([`make_objfun`](@ref), [`make_confun`](@ref),
[`make_usrfun_a`](@ref), [`make_usrfun_c`](@ref)) take a `callback` keyword for
per-evaluation monitoring. Leave it at the default `nothing` to skip monitoring with
no per-evaluation overhead, or pass [`snopt_no_progress`](@ref) as an explicit
accept-everything callback — note this still builds an event object on each
evaluation, so only `nothing` truly avoids the overhead. [`make_snlog`](@ref) instead
takes its callback as a positional argument, and [`make_dummy_confun`](@ref) takes
none.

A minimal `snOptB` solve assembled by hand:

```julia
using SparseArrays

initialize("", "") do ws
    set_option!(ws, "Major print level", 0)

    n, m_eff = 2, 1
    objfun = make_objfun((x) -> (x[1]-1)^2 + (x[2]-2)^2,
                         (g, x) -> (g[1] = 2(x[1]-1); g[2] = 2(x[2]-2); nothing),
                         ws.iw)
    confun = make_dummy_confun()

    x  = [0.0, 0.0, 0.0]                 # n design vars + m_eff slack
    bl = [-10.0, -10.0, -1e20]
    bu = [ 10.0,  10.0,  1e20]
    hs = zeros(Int32, n + m_eff)
    J  = SparseMatrixCSC{Float64,Int32}(1, n, Int32.(vcat(1, fill(2, n))),
                                        Int32[1], Float64[0.0])

    prob = SnoptB(ws, n, 0, m_eff, n, x, bl, bu, hs, J, 0.0, 0, Float64[],
                  objfun, confun)
    snoptb!(prob)
    prob.status, prob.obj_val, prob.x[1:n]
end
```

In practice, prefer [`snopt`](@ref) unless you specifically need this level of
control — it performs exactly this assembly, with validation and automatic
workspace sizing.
