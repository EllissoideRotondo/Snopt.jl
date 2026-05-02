# Unconstrained quadratic  min (x - 1)^2 + (y - 2)^2
# Analytical solution: x* = (1, 2), f* = 0

using Pkg
Pkg.activate(@__DIR__)

using Snopt

function eval_obj(x)
    return (x[1] - 1)^2 + (x[2] - 2)^2
end

function eval_grad!(g, x)
    g[1] = 2(x[1] - 1)
    g[2] = 2(x[2] - 2)
    return nothing
end

function progress(event::SnoptMajorLog)
    println("  major $(event.major_iter)  minor $(event.minor_iter)  f = $(event.objective)")
    return true
end

result = snopt(
    eval_obj,
    eval_grad!,
    [0.0, 0.0];
    lb = -10.0,
    ub = 10.0,
    options = [
        "Major print level" => 1,
        "Minor print level" => 0,
    ],
    snlog = progress,
    printfile = joinpath(@__DIR__, "unconstrained.out"),
)

println("\n=== Unconstrained Quadratic Result ===")
println("Status : ", result.status, " (", result.status_symbol, ")")
println("Obj    : ", result.objective)
println("x*     : ", result.x)
