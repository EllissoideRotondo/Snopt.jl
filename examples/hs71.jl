# Hock-Schittkowski Problem 71
#
#   min  x1*x4*(x1+x2+x3) + x3
#   s.t. x1*x2*x3*x4 >= 25
#        x1^2+x2^2+x3^2+x4^2 = 40
#        1 <= xi <= 5,  i = 1..4
#
#   Known solution: x* = (1, 4.7430, 3.8211, 1.3791), f* ≈ 17.0140

using Pkg
Pkg.activate(@__DIR__)

using Snopt
using SparseArrays

function eval_obj(x)
    return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
end

function eval_grad!(g, x)
    g[1] = x[4] * (2x[1] + x[2] + x[3])
    g[2] = x[1] * x[4]
    g[3] = x[1] * x[4] + 1
    g[4] = x[1] * (x[1] + x[2] + x[3])
    return nothing
end

function eval_con!(c, x)
    c[1] = x[1] * x[2] * x[3] * x[4]
    c[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    return nothing
end

function eval_jac!(jnz, x)
    jnz[1] = x[2] * x[3] * x[4]
    jnz[2] = 2x[1]
    jnz[3] = x[1] * x[3] * x[4]
    jnz[4] = 2x[2]
    jnz[5] = x[1] * x[2] * x[4]
    jnz[6] = 2x[3]
    jnz[7] = x[1] * x[2] * x[3]
    jnz[8] = 2x[4]
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
    eval_obj,
    eval_grad!,
    [1.0, 5.0, 5.0, 1.0];
    lb = ones(4),
    ub = 5 * ones(4),
    eval_con = eval_con!,
    eval_jac = eval_jac!,
    lcon = [25.0, 40.0],
    ucon = [1.0e20, 40.0],
    J,
    options = [
        "Major print level" => 1,
        "Minor print level" => 0,
    ],
    printfile = joinpath(@__DIR__, "hs71.out"),
)

println("\n=== HS71 Result ===")
println("Status : ", result.status, " (", result.status_symbol, ")")
println("Obj    : ", result.objective)
println("x*     : ", result.x)
