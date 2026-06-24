```@meta
CurrentModule = SNOPT
```

# High-level interface

The [`snopt`](@ref) function is the recommended entry point. It assembles a
[`SnoptB`](@ref) problem, sizes the workspace with SNOPT's own estimator, runs the
solve, and returns a [`SnoptResult`](@ref). It manages the SNOPT workspace for you,
freeing it even if an error or callback exception occurs.

```julia
result = snopt(eval_obj, eval_grad, x0; kwargs...)
```

## Required arguments

| Argument    | Meaning |
|-------------|---------|
| `eval_obj`  | `eval_obj(x) -> Real`, the scalar objective. |
| `eval_grad` | `eval_grad(g, x)`, fills the objective gradient `g` in place. |
| `x0`        | starting point; its length sets the number of variables `n`. |

## Bounds

`lb` and `ub` set the variable lower/upper bounds. Each may be a vector of length
`n` or a scalar that is broadcast to every variable. Omitted bounds default to
``\pm`` infinity. Infinite values are clamped to SNOPT's "infinite bound" sentinel
(`1e20`) automatically.

```julia
snopt(f, g!, x0; lb = [0.0, -1.0], ub = 5.0)   # vector lower, scalar upper
```

## Nonlinear constraints

To add `m` nonlinear constraints ``l_c \le c(x) \le u_c``, supply all of:

| Argument   | Meaning |
|------------|---------|
| `eval_con` | `eval_con(c, x)`, fills the `m` constraint values in place. |
| `eval_jac` | `eval_jac(jnz, x)`, fills the Jacobian nonzeros (see below). |
| `lcon`     | constraint lower bounds (length `m`). |
| `ucon`     | constraint upper bounds (length `m`). |
| `J`        | *optional* sparse `SparseMatrixCSC` giving the Jacobian sparsity. |

`eval_jac` writes the ``\partial c / \partial x`` nonzeros into `jnz` in the
**column-major order of `J`** (i.e. the order of `J.nzval`). If `J` is omitted, a
dense `m × n` pattern is assumed and you must fill every entry column by column.
Only the sparsity *structure* of `J` matters; its numeric values are ignored.

```julia
using SparseArrays

# two constraints in four variables, all four partials nonzero in each row
J = sparse(Int32[1,2,1,2,1,2,1,2], Int32[1,1,2,2,3,3,4,4], ones(8), 2, 4)

snopt(f, g!, x0;
    eval_con = c!, eval_jac = jac!,
    lcon = [25.0, 40.0], ucon = [1e20, 40.0], J = J)
```

Equality constraints are expressed by setting the matching entries of `lcon` and
`ucon` equal (as with the second constraint above).

## Options

`options` is a vector of pairs. Keys may be strings or symbols; in a symbol,
underscores become spaces, so these are equivalent:

```julia
options = ["Major print level" => 0, "Minor print level" => 0]
options = [:major_print_level => 0, :minor_print_level => 0]
```

Values may be integers, finite floats, strings, or symbols (`Bool` is rejected, to
avoid silently coercing `true`/`false` to `1`/`0`). String/symbol values are
appended to the keyword, so `"Hessian" => :limited` sends `"Hessian limited"`.
Options can also be loaded from a SNOPT specs file with [`read_options`](@ref).

A few commonly used keywords:

| Keyword | Effect |
|---------|--------|
| `"Major print level"` / `"Minor print level"` | verbosity of the print file |
| `"Major iterations limit"` | SQP iteration cap |
| `"Major optimality tolerance"` | convergence tolerance |
| `"Major feasibility tolerance"` | constraint tolerance |
| `"Derivative option"` | `1` if you supply all gradients, `0` for finite differences |

## Monitoring and early termination

Two independent hooks are available:

- **`snlog`** receives a [`SnoptMajorLog`](@ref) once per *major* iteration, with
  meaningful iteration counters and the current point, objective, infeasibilities,
  and multipliers. Use it for trace/progress output.
- **`callback`** fires on each objective/constraint *evaluation* (which may happen
  several times per major iteration). It receives a `NamedTuple` event with fields
  such as `kind` (`:objective` or `:constraint`), `mode`, `major_iter`,
  `minor_iter`, `x`, and `f` or `c`.

Returning `false` from either hook requests SNOPT to stop; the resulting
[`SnoptResult`](@ref) then carries a `:User_Requested_Stop` status.

```julia
snopt(f, g!, x0;
    snlog = ev -> (println("major $(ev.major_iter): f = $(ev.objective)"); true),
    callback = ev -> ev.f < 1e6,   # bail out if the objective blows up
)
```

## Output files and start mode

| Keyword     | Default  | Meaning |
|-------------|----------|---------|
| `printfile` | `""`     | path for SNOPT's detailed print output (empty = suppressed) |
| `summfile`  | `""`     | path for SNOPT's summary output |
| `start`     | `"Cold"` | `"Cold"`, `"Warm"`, or `"Hot"` start |
| `name`      | `"Julia"`| ≤8-character problem name shown in SNOPT output |

## The result

[`snopt`](@ref) returns a [`SnoptResult`](@ref):

```julia
result.status          # SNOPT inform code (Int)
result.status_symbol   # Symbol, e.g. :Solve_Succeeded (see SNOPT_STATUS)
result.objective       # final objective value
result.x               # final variable values (length n)
result.lambda          # Lagrange multipliers (variables and constraints)
result.num_inf         # number of remaining infeasibilities
result.sum_inf         # sum of infeasibilities
result.iterations      # total minor iterations
result.major_itns      # total major iterations
result.run_time        # SNOPT-reported solve time (s)
result.memory          # the SnoptMemory estimate used to size the workspace
```

Map an inform code to its symbolic meaning through [`SNOPT_STATUS`](@ref).
</content>
