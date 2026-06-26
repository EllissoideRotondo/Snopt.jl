```@meta
CurrentModule = SNOPT
```

# Examples

Both examples below are included as runnable scripts in the
[`examples/`](https://github.com/EllissoideRotondo/SNOPT.jl/tree/main/examples)
directory of the repository.

## Unconstrained quadratic

Minimize ``(x_1 - 1)^2 + (x_2 - 2)^2``, whose minimizer is ``x^\star = (1, 2)`` with
``f^\star = 0``. This example also shows the `snlog` major-iteration hook.

```julia
using SNOPT

objective(x) = (x[1] - 1)^2 + (x[2] - 2)^2

function gradient!(g, x)
    g[1] = 2(x[1] - 1)
    g[2] = 2(x[2] - 2)
    return nothing
end

function progress(event::SnoptMajorLog)
    println("  major $(event.major_iter)  minor $(event.minor_iter)  f = $(event.objective)")
    return true
end

result = snopt(
    objective, gradient!, [0.0, 0.0];
    lb = -10.0, ub = 10.0,
    options = ["Major print level" => 1, "Minor print level" => 0],
    snlog = progress,
)

println("Status : ", result.status, " (", result.status_symbol, ")")
println("Obj    : ", result.objective)
println("x*     : ", result.x)
```

## Constrained: Hock–Schittkowski 71

The classic HS71 test problem:

```math
\begin{aligned}
\min_{x}\quad & x_1 x_4 (x_1 + x_2 + x_3) + x_3 \\
\text{s.t.}\quad & x_1 x_2 x_3 x_4 \ge 25 \\
& x_1^2 + x_2^2 + x_3^2 + x_4^2 = 40 \\
& 1 \le x_i \le 5, \quad i = 1,\dots,4
\end{aligned}
```

with known solution ``x^\star \approx (1,\ 4.743,\ 3.821,\ 1.379)`` and
``f^\star \approx 17.014``.

The two constraints depend on all four variables, so the Jacobian is dense; we
still declare its sparsity explicitly to fix the ordering that `eval_jac` must
follow (column-major: ``\partial c_1/\partial x_1, \partial c_2/\partial x_1,
\partial c_1/\partial x_2, \dots``).

```julia
using SNOPT
using SparseArrays

objective(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]

function gradient!(g, x)
    g[1] = x[4] * (2x[1] + x[2] + x[3])
    g[2] = x[1] * x[4]
    g[3] = x[1] * x[4] + 1
    g[4] = x[1] * (x[1] + x[2] + x[3])
    return nothing
end

function constraints!(c, x)
    c[1] = x[1] * x[2] * x[3] * x[4]
    c[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    return nothing
end

function jacobian!(jac, x)
    jac[1] = x[2] * x[3] * x[4];  jac[2] = 2x[1]    # ∂c/∂x1
    jac[3] = x[1] * x[3] * x[4];  jac[4] = 2x[2]    # ∂c/∂x2
    jac[5] = x[1] * x[2] * x[4];  jac[6] = 2x[3]    # ∂c/∂x3
    jac[7] = x[1] * x[2] * x[3];  jac[8] = 2x[4]    # ∂c/∂x4
    return nothing
end

J = sparse(Int32[1,2,1,2,1,2,1,2], Int32[1,1,2,2,3,3,4,4], ones(8), 2, 4)

result = snopt(
    objective, gradient!, [1.0, 5.0, 5.0, 1.0];
    lb = ones(4), ub = 5 * ones(4),
    eval_con = constraints!, eval_jac = jacobian!,
    lcon = [25.0, 40.0], ucon = [Inf, 40.0],   # c1 >= 25; c2 = 40
    J = J,
    options = ["Major print level" => 1, "Minor print level" => 0],
)

println("Status : ", result.status, " (", result.status_symbol, ")")
println("Obj    : ", result.objective)
println("x*     : ", result.x)
```

The equality constraint ``c_2 = 40`` is encoded by setting `lcon[2] == ucon[2] ==
40`, and the inequality ``c_1 \ge 25`` by an upper bound at SNOPT's infinity
sentinel, which the high-level wrapper accepts as `Inf`.
