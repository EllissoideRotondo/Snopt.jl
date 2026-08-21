```@meta
CurrentModule = SNOPT
```

# Examples

Run both scripts from the SNOPT.jl repository root:

```bash
julia --project=. examples/unconstrained.jl
julia --project=. examples/hs71.jl
```

Both commands require a working licensed SNOPT library.

## Unconstrained quadratic

This problem minimizes ``(x_1 - 1)^2 + (x_2 - 2)^2``. Its solution is
``x = (1, 2)`` with objective zero.

```julia
using SNOPT

objective(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2

function gradient!(gradient, x)
    gradient[1] = 2.0 * (x[1] - 1.0)
    gradient[2] = 2.0 * (x[2] - 2.0)
    return nothing
end

function progress(event::SnoptMajorLog)
    println("major $(event.major_iter): objective = $(event.objective)")
    return true
end

result = snopt(
    objective,
    gradient!,
    [0.0, 0.0];
    lb = -10.0,
    ub = 10.0,
    options = ["Major print level" => 0],
    snlog = progress,
)

println("status = ", result.status_symbol)
println("objective = ", result.objective)
println("x = ", result.x)
```

`gradient!` fills both gradient entries. `progress` returns `true` so SNOPT
continues after each major iteration.

## Constrained problem

Hock-Schittkowski problem 71 has four variables and two constraints.

```text
minimize    x1*x4*(x1 + x2 + x3) + x3
subject to  x1*x2*x3*x4 >= 25
            x1^2 + x2^2 + x3^2 + x4^2 = 40
            1 <= xi <= 5
```

The known objective is approximately `17.014017`.

```julia
using SNOPT
using SparseArrays

objective(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]

function gradient!(gradient, x)
    gradient[1] = x[4] * (2.0 * x[1] + x[2] + x[3])
    gradient[2] = x[1] * x[4]
    gradient[3] = x[1] * x[4] + 1.0
    gradient[4] = x[1] * (x[1] + x[2] + x[3])
    return nothing
end

function constraints!(values, x)
    values[1] = x[1] * x[2] * x[3] * x[4]
    values[2] = sum(abs2, x)
    return nothing
end

function jacobian!(nonzeros, x)
    nonzeros[1] = x[2] * x[3] * x[4]
    nonzeros[2] = 2.0 * x[1]
    nonzeros[3] = x[1] * x[3] * x[4]
    nonzeros[4] = 2.0 * x[2]
    nonzeros[5] = x[1] * x[2] * x[4]
    nonzeros[6] = 2.0 * x[3]
    nonzeros[7] = x[1] * x[2] * x[3]
    nonzeros[8] = 2.0 * x[4]
    return nothing
end

J = sparse(
    Int32[1, 2, 1, 2, 1, 2, 1, 2],
    Int32[1, 1, 2, 2, 3, 3, 4, 4],
    ones(8),
    2,
    4,
)

result = snopt(
    objective,
    gradient!,
    [1.0, 5.0, 5.0, 1.0];
    lb = ones(4),
    ub = fill(5.0, 4),
    eval_con = constraints!,
    eval_jac = jacobian!,
    lcon = [25.0, 40.0],
    ucon = [Inf, 40.0],
    J,
    options = ["Major print level" => 0],
)

println("status = ", result.status_symbol)
println("objective = ", result.objective)
println("x = ", result.x)
```

The Jacobian is dense, but `J` makes the storage order explicit.
`jacobian!` fills entries by column, in `J.nzval` order.
