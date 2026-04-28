# Hock-Schittkowski Problem 71
#
#   min  x1*x4*(x1+x2+x3) + x3
#   s.t. x1*x2*x3*x4 >= 25
#        x1^2+x2^2+x3^2+x4^2 = 40
#        1 <= xi <= 5,  i = 1..4
#
#   Known solution: x* = (1, 4.7430, 3.8211, 1.3791),  f* ≈ 17.0140
using Pkg
Pkg.activate(@__DIR__)

using Snopt
using SparseArrays

ws = initialize("hs71.out", "")
set_option!(ws, "Major print level", 1)

function eval_obj(x)
    x[1]*x[4]*(x[1]+x[2]+x[3]) + x[3]
end

function eval_grad!(g, x)
    g[1] = x[4]*(2x[1]+x[2]+x[3])
    g[2] = x[1]*x[4]
    g[3] = x[1]*x[4] + 1
    g[4] = x[1]*(x[1]+x[2]+x[3])
end

function eval_con!(c, x)
    c[1] = x[1]*x[2]*x[3]*x[4]
    c[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
end

function eval_jac!(jnz, x)
    jnz[1] = x[2]*x[3]*x[4];  jnz[2] = 2x[1]
    jnz[3] = x[1]*x[3]*x[4];  jnz[4] = 2x[2]
    jnz[5] = x[1]*x[2]*x[4];  jnz[6] = 2x[3]
    jnz[7] = x[1]*x[2]*x[3];  jnz[8] = 2x[4]
end

J = sparse(Int32[1,2,1,2,1,2,1,2], Int32[1,1,2,2,3,3,4,4], ones(8), 2, 4)

objfun = make_objfun(eval_obj, eval_grad!, ws.iw)
confun = make_confun(eval_con!, eval_jac!, J)

n    = 4
x0   = [1.0, 5.0, 5.0, 1.0]
bl_x = ones(4);   bu_x = 5 * ones(4)
bl_c = [25.0, 40.0];  bu_c = [1e20, 40.0]

m  = 2
x  = [x0; zeros(m)];  bl = [bl_x; bl_c];  bu = [bu_x; bu_c]
hs = zeros(Int32, n + m)

prob = SnoptProblem(ws, n, 2, m, x, bl, bu, hs, J, 0.0, 0, Float64[], objfun, confun)

status = snopt!(prob)

println("\n=== HS71 Result ===")
println("Status : ", SNOPT_STATUS[status])
println("Obj    : ", prob.obj_val)
println("x*     : ", prob.x[1:4])
