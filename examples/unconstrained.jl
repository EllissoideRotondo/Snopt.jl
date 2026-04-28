# Unconstrained quadratic  min (x-1)^2 + (y-2)^2
# Analytical solution: x*=(1,2), f*=0

using Snopt
using SparseArrays

ws = initialize("unconstrained.out", "")
set_option!(ws, "Major print level", 1)

objfun = make_objfun(
    x -> (x[1]-1)^2 + (x[2]-2)^2,
    (g, x) -> begin g[1] = 2(x[1]-1); g[2] = 2(x[2]-2) end,
    ws.iw;
    callback = event -> (println("  iter $(event.major_iter)  f=$(event.f)"); true)
)
confun = make_dummy_confun()

n     = 2
m_eff = 1
x0    = [0.0, 0.0]
x     = [x0; zeros(m_eff)]
bl    = [-10.0, -10.0, -1e20]
bu    = [ 10.0,  10.0,  1e20]
hs    = zeros(Int32, n + m_eff)
J     = SparseMatrixCSC{Float64,Int32}(1, n,
    Int32.(vcat(1, fill(2, n))), Int32[1], Float64[0.0])

prob = SnoptProblem(ws, n, 0, m_eff, x, bl, bu, hs, J, 0.0, 0, Float64[], objfun, confun)

status = snopt!(prob)

println("\n=== Unconstrained Quadratic Result ===")
println("Status : ", status, " (", SNOPT_STATUS[status], ")")
println("Obj    : ", prob.obj_val)
println("x*     : ", prob.x[1:2])
