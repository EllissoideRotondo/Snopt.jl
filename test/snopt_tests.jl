using SparseArrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function make_ws(; silent=true)
    ws = initialize("", "")
    if silent
        set_option!(ws, "Major print level", 0)
        set_option!(ws, "Minor print level", 0)
    end
    return ws
end

# Build a SnoptB/SnoptProblem for an unconstrained problem (nc=0).
# objfun and confun must already be closures wrapping ws.iw.
function make_unconstrained_prob(ws, x0, bl_x, bu_x, objfun, confun)
    n = length(x0)
    m_eff = 1
    x    = [x0; zeros(m_eff)]
    bl   = [bl_x; -1e20 * ones(m_eff)]
    bu   = [bu_x;  1e20 * ones(m_eff)]
    hs   = zeros(Int32, n + m_eff)
    J = SparseMatrixCSC{Float64,Int32}(1, n,
        Int32.(vcat(1, fill(2, n))), Int32[1], Float64[0.0])
    return SnoptB(ws, n, 0, m_eff, x, bl, bu, hs, J, 0.0, 0, Float64[], objfun, confun)
end

function make_constrained_prob(ws, x0, bl_x, bu_x, bl_c, bu_c, objfun, confun, J)
    n    = length(x0)
    nc   = size(J, 1)
    m    = nc
    x    = [x0; zeros(m)]
    bl   = [bl_x; bl_c]
    bu   = [bu_x; bu_c]
    hs   = zeros(Int32, n + m)
    return SnoptB(ws, n, nc, m, x, bl, bu, hs, J, 0.0, 0, Float64[], objfun, confun)
end

function make_constrained_prob_c(ws, x0, bl_x, bu_x, bl_c, bu_c, usrfun, J)
    n    = length(x0)
    nc   = size(J, 1)
    m    = nc
    x    = [x0; zeros(m)]
    bl   = [bl_x; bl_c]
    bu   = [bu_x; bu_c]
    hs   = zeros(Int32, n + m)
    return SnoptC(ws, n, nc, m, x, bl, bu, hs, J, 0.0, 0, Float64[], usrfun)
end

mutable struct SnoptLogCollector
    logs::Vector{SnoptMajorLog}
end

function (collector::SnoptLogCollector)(event)
    push!(collector.logs, event)
    return true
end

@testset "Workspace initialization" begin
    ws = initialize("", "")
    @test ws isa Snopt.SnoptWorkspace
    @test ws.leniw > 0
    @test ws.lenrw > 0
    @test_throws ArgumentError Snopt.SnoptWorkspace(0, 10)
    @test_throws ArgumentError Snopt.SnoptWorkspace(10, 0)

    ws2 = initialize("", "", 40000, 5000)
    @test ws2.leniw == 40000
    @test ws2.lenrw == 5000
    @test SnoptProblem === SnoptB
    @test SnoptA <: AbstractSnoptProblem
    @test SnoptB <: AbstractSnoptProblem
    @test SnoptC <: AbstractSnoptProblem
end

@testset "Library discovery" begin
    libdir = dirname(Snopt.libsnopt7)
    original_path = get(ENV, "PATH", "")
    withenv("SNOPTDIR" => libdir, "SNOPT_GFORTRAN_BINDIR" => "not-used") do
        @test normpath(Snopt.find_snopt_lib()) == normpath(Snopt.libsnopt7)
        @test get(ENV, "PATH", "") == original_path
    end
    mktempdir() do dir
        withenv("SNOPTDIR" => dir) do
            @test Snopt.find_snopt_lib() == ""
        end
    end
end

@testset "SNOPTB memory estimation" begin
    ws = make_ws()
    @test Snopt.SNOPT_MEMORY_WORKSPACE >= 1000
    memory = snmemb(ws, 1, 2, 1, 0, 0, 0, 2)
    @test memory isa SnoptMemory
    @test memory.info == 104
    @test memory.miniw >= 500
    @test memory.minrw >= 500

    memory2 = snmemb(2, 4, 8, 8, 2, 4, 4;
                     options = [
                         "Major print level" => 0,
                         "Minor print level" => 0,
                     ])
    @test memory2.info == 104
    @test memory2.miniw >= memory.miniw
    @test memory2.minrw >= memory.minrw
end

@testset "set_option! variants" begin
    ws = make_ws()

    # String form
    @test set_option!(ws, "Major print level 0") == 0

    # Integer form
    @test set_option!(ws, "Major iterations limit", 500) == 0
    @test set_option!(ws, "Minor iterations limit", 200) == 0

    # Float64 form
    @test set_option!(ws, "Major feasibility tolerance", 1e-8) == 0
    @test set_option!(ws, "Major optimality tolerance",  1e-8) == 0

    # Empty and whitespace-only strings are rejected before touching SNOPT.
    @test_throws ArgumentError set_option!(ws, "")
    @test_throws ArgumentError set_option!(ws, "   ")
    @test_throws ArgumentError set_option!(ws, "", 0)
    @test_throws ArgumentError set_option!(ws, "  ", 0)
    @test_throws ArgumentError set_option!(ws, "", 0.0)
    @test_throws ArgumentError set_option!(ws, "  ", 0.0)

    # Non-ASCII and misspelled keywords are rejected by SNOPT and surfaced
    # as Julia errors.
    stdout_file = tempname()
    open(stdout_file, "w") do io
        redirect_stdout(io) do
            @test_throws ArgumentError set_option!(ws, "Mäjor print level 0")
            @test_throws ArgumentError set_option!(ws, "Definitely unknown option 0")
        end
    end
    if !Sys.iswindows()
        @test !occursin("Keyword not recognized", read(stdout_file, String))
    end
end

@testset "options vector of pairs" begin
    ws = make_ws(silent=false)

    @test Snopt.apply_options!(ws, [
        "Major print level" => 0,
        :minor_print_level => 0,
        "Major feasibility tolerance" => 1.0e-8,
        :hessian => :full_memory,
    ]) === ws

    @test_throws ArgumentError Snopt.apply_options!(ws, "Major print level 0")
    @test_throws ArgumentError Snopt.apply_options!(ws, "Major print level" => 0)
    @test_throws ArgumentError Snopt.apply_options!(ws, Dict("Major print level" => 0))
    @test_throws ArgumentError Snopt.apply_options!(ws, ("Major print level" => 0,))
    @test_throws ArgumentError Snopt.apply_options!(ws, ["Major print level 0"])
    @test_throws ArgumentError Snopt.apply_options!(ws, ["Major print level" => false])
    @test_throws ArgumentError Snopt.apply_options!(ws, ["Major feasibility tolerance" => Inf])
    @test_throws ArgumentError Snopt.apply_options!(ws, ["Major print level" => ""])
    @test_throws ArgumentError Snopt.apply_options!(ws, [1 => 0])

    @test_throws ArgumentError snmemb(2, 4, 8, 8, 2, 4, 4;
                                     options = ["Major print level 0"])
end

@testset "public validation and callback exceptions" begin
    ws = make_ws()
    objfun = make_objfun(
        x -> (x[1] - 1)^2,
        (g, x) -> begin g[1] = 2(x[1] - 1) end,
        ws.iw
    )
    @test objfun isa Function
    @test Snopt.callback_state(objfun) isa Snopt.SnoptCallbackState
    prob = make_unconstrained_prob(
        ws, [0.0], [-10.0], [10.0], objfun, make_dummy_confun()
    )
    prob.hs = Int32[]
    @test_throws DimensionMismatch snopt!(prob)

    silent_options = [
        "Major print level" => 0,
        "Minor print level" => 0,
    ]
    @test_throws ErrorException snopt(
        x -> error("objective boom"),
        (g, x) -> begin g[1] = 2x[1] end,
        [0.0];
        options = silent_options
    )
    @test_throws ErrorException snopt(
        x -> x[1]^2,
        (g, x) -> begin g[1] = 2x[1] end,
        [1.0];
        options = silent_options,
        callback = _ -> error("progress boom")
    )
end

@testset "SnoptA toy problem" begin
    ws = make_ws()
    set_option!(ws, "Derivative option", 1)

    events = NamedTuple[]
    usrfun = make_usrfun_a(
        (F, x) -> begin
            F[1] = x[2]
            F[2] = x[1]^2 + 4x[2]^2
            F[3] = (x[1] - 2)^2 + x[2]^2
        end;
        eval_G = (G, x) -> begin
            G[1] = 2x[1]
            G[2] = 8x[2]
            G[3] = 2(x[1] - 2)
            G[4] = 2x[2]
        end,
        callback = event -> begin push!(events, event); true end
    )

    prob = SnoptA(
        ws,
        3, 2,
        0.0, 1,
        Int32[1], Int32[2], [1.0],
        Int32[2, 2, 3, 3], Int32[1, 2, 1, 2],
        [0.0, -1.0e20],
        [1.0e20, 1.0e20],
        [-1.0e20, -1.0e20, -1.0e20],
        [1.0e20, 4.0, 5.0],
        [1.0, 1.0],
        zeros(Int32, 2),
        zeros(2),
        zeros(3),
        zeros(Int32, 3),
        zeros(3),
        0, 0, 0, 0.0,
        usrfun
    )

    status = snopta!(prob)
    @test status == 1
    @test prob isa SnoptA
    @test prob.x[1] ≈ 0.0 atol=1.0e-4
    @test prob.x[2] ≈ -1.0 atol=1.0e-4
    @test any(event -> event.kind === :function, events)
end

@testset "SNOPT_STATUS dictionary" begin
    @test Snopt.SNOPT_STATUS[1]   == :Solve_Succeeded
    @test Snopt.SNOPT_STATUS[2]   == :Feasible_Point_Found
    @test Snopt.SNOPT_STATUS[11]  == :Infeasible_Problem_Detected
    @test Snopt.SNOPT_STATUS[21]  == :Unbounded_Problem_Detected
    @test Snopt.SNOPT_STATUS[31]  == :Maximum_Iterations_Exceeded
    @test Snopt.SNOPT_STATUS[33]  == :Maximum_Iterations_Exceeded
    @test Snopt.SNOPT_STATUS[34]  == :Maximum_CpuTime_Exceeded
    @test Snopt.SNOPT_STATUS[41]  == :Numerical_Difficulties
    @test Snopt.SNOPT_STATUS[71]  == :User_Requested_Stop
    @test Snopt.SNOPT_STATUS[81]  == :Insufficient_Memory
    @test Snopt.SNOPT_STATUS[84]  == :Insufficient_Memory
    @test Snopt.SNOPT_STATUS[141] == :Internal_Error
    @test Snopt.SNOPT_STATUS[142] == :Internal_Error
    @test Snopt.SNOPT_STATUS[999] == :Internal_Error
end

@testset "read_options" begin
    ws = make_ws()
    file = joinpath(@__DIR__, "specsfile")
    ret = read_options(ws, file)
    @test ret == 101
    @test ws.status == 101
    objfun = make_objfun(
        x -> x[1]^2,
        (g, x) -> begin g[1] = 2x[1] end,
        ws.iw
    )
    prob = make_unconstrained_prob(
        ws, [0.0], [-10.0], [10.0], objfun, make_dummy_confun()
    )
    @test read_options(prob, file) == 101
end

@testset "Low-level snopt API (unconstrained)" begin
    result = snopt(
        x -> (x[1] - 1)^2 + (x[2] - 2)^2,
        (g, x) -> begin
            g[1] = 2(x[1] - 1)
            g[2] = 2(x[2] - 2)
        end,
        [0.0, 0.0];
        lb = -10.0,
        ub = 10.0,
        options = [
            "Major print level" => 0,
            "Minor print level" => 0,
        ]
    )

    @test result isa SnoptResult
    @test result.status == 1
    @test result.status_symbol === :Solve_Succeeded
    @test result.objective ≈ 0.0 atol=1e-6
    @test result.x[1] ≈ 1.0 atol=1e-5
    @test result.x[2] ≈ 2.0 atol=1e-5
    @test result.memory.miniw > 0
    @test result.memory.minrw > 0
    @test !hasproperty(result, :problem)
end

# ---------------------------------------------------------------------------
# 5. Unconstrained quadratic  min (x-1)^2 + (y-2)^2
# ---------------------------------------------------------------------------
@testset "Unconstrained quadratic" begin
    ws = make_ws()

    events = NamedTuple[]
    progress = event -> begin push!(events, event); true end

    objfun = make_objfun(
        x -> (x[1]-1)^2 + (x[2]-2)^2,
        (g, x) -> begin g[1] = 2(x[1]-1); g[2] = 2(x[2]-2) end,
        ws.iw;
        callback=progress
    )
    confun = make_dummy_confun()

    prob = make_unconstrained_prob(
        ws,
        [0.0, 0.0],          # x0
        fill(-10.0, 2),      # lower bounds
        fill( 10.0, 2),      # upper bounds
        objfun, confun
    )

    status = snopt!(prob)

    @test status == 1
    @test prob.obj_val ≈ 0.0  atol=1e-6
    @test prob.ws.x[1] ≈ 1.0  atol=1e-5
    @test prob.ws.x[2] ≈ 2.0  atol=1e-5
    @test prob.x[1] ≈ 1.0  atol=1e-5
    @test prob.x[2] ≈ 2.0  atol=1e-5
    @test any(event -> event.kind === :objective, events)
    @test all(event -> event.x isa Vector{Float64}, events)
    @test all(event -> event.major_iter >= 0 && event.minor_iter >= 0, events)
end

@testset "snLog major iteration callback" begin
    ws = make_ws()
    collector = SnoptLogCollector(SnoptMajorLog[])

    objfun = make_objfun(
        x -> (x[1]-1)^2 + (x[2]-2)^2,
        (g, x) -> begin g[1] = 2(x[1]-1); g[2] = 2(x[2]-2) end,
        ws.iw
    )
    confun = make_dummy_confun()

    prob = make_unconstrained_prob(
        ws,
        [0.0, 0.0],
        fill(-10.0, 2),
        fill(10.0, 2),
        objfun, confun
    )

    status = snopt!(prob; snlog = collector)

    @test status == 1
    logs = collector.logs
    @test !isempty(logs)
    @test logs[end] isa SnoptMajorLog
    @test all(log -> length(log.x) == prob.n + prob.m_eff, logs)
    @test all(log -> length(log.hs) == prob.n + prob.m_eff, logs)
    @test any(log -> log.major_iter >= 0 && log.minor_iter >= 0, logs)
    @test any(log -> log.major_iter > 0, logs)
    @test logs[end].major_iter == prob.ws.major_itns
    @test minimum(abs(log.objective - prob.obj_val) for log in logs) <= 1e-6
end

@testset "Rosenbrock (unconstrained)" begin
    ws = make_ws()

    objfun = make_objfun(
        x -> (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2,
        (g, x) -> begin
            g[1] = -2*(1-x[1]) - 400*x[1]*(x[2]-x[1]^2)
            g[2] = 200*(x[2]-x[1]^2)
        end,
        ws.iw
    )
    confun = make_dummy_confun()

    prob = make_unconstrained_prob(
        ws,
        [-1.2, 1.0],
        fill(-10.0, 2),
        fill( 10.0, 2),
        objfun, confun
    )

    status = snopt!(prob)

    @test status ∈ [1, 2, 3]
    @test prob.ws.x[1] ≈ 1.0  atol=1e-4
    @test prob.ws.x[2] ≈ 1.0  atol=1e-4
    @test prob.x[1] ≈ 1.0  atol=1e-4
    @test prob.x[2] ≈ 1.0  atol=1e-4
end

@testset "Low-level snopt API (constrained)" begin
    function eval_obj_ll(x)
        x[1]*x[4]*(x[1]+x[2]+x[3]) + x[3]
    end
    function eval_grad_ll!(g, x)
        g[1] = x[4]*(2x[1]+x[2]+x[3])
        g[2] = x[1]*x[4]
        g[3] = x[1]*x[4] + 1
        g[4] = x[1]*(x[1]+x[2]+x[3])
    end
    function eval_con_ll!(c, x)
        c[1] = x[1]*x[2]*x[3]*x[4]
        c[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    end
    function eval_jac_ll!(jnz, x)
        jnz[1] = x[2]*x[3]*x[4]
        jnz[2] = 2x[1]
        jnz[3] = x[1]*x[3]*x[4]
        jnz[4] = 2x[2]
        jnz[5] = x[1]*x[2]*x[4]
        jnz[6] = 2x[3]
        jnz[7] = x[1]*x[2]*x[3]
        jnz[8] = 2x[4]
    end

    J = sparse(
        Int32[1,2,1,2,1,2,1,2],
        Int32[1,1,2,2,3,3,4,4],
        ones(8), 2, 4
    )

    result = snopt(
        eval_obj_ll, eval_grad_ll!, [1.0, 5.0, 5.0, 1.0];
        lb = ones(4),
        ub = 5 * ones(4),
        eval_con = eval_con_ll!,
        eval_jac = eval_jac_ll!,
        lcon = [25.0, 40.0],
        ucon = [1e20, 40.0],
        J,
        options = [
            "Major print level" => 0,
            "Minor print level" => 0,
        ]
    )

    @test result.status == 1
    @test result.objective ≈ 17.0140 atol=1e-3
    @test result.x[1] ≈ 1.0 atol=1e-3
    @test result.x[2] ≈ 4.7430 atol=1e-3
    @test result.x[3] ≈ 3.8211 atol=1e-3
    @test result.x[4] ≈ 1.3791 atol=1e-3
end

# ---------------------------------------------------------------------------
# 7. HS71 — Hock-Schittkowski problem 71 (canonical nonlinear NLP)
#
#   min  x1*x4*(x1+x2+x3) + x3
#   s.t. x1*x2*x3*x4 >= 25      (g1, inequality)
#        x1^2+x2^2+x3^2+x4^2 = 40  (g2, equality)
#        1 <= xi <= 5
#
#   Known solution: x* = (1, 4.7430, 3.8211, 1.3791), f* ≈ 17.0140
# ---------------------------------------------------------------------------
@testset "HS71 (nonlinearly constrained)" begin
    ws = make_ws()
    events = NamedTuple[]
    progress = event -> begin push!(events, event); true end

    function eval_obj(x)
        x[1]*x[4]*(x[1]+x[2]+x[3]) + x[3]
    end
    function eval_grad!(g, x)
        g[1] = x[4]*(2x[1]+x[2]+x[3])
        g[2] = x[1]*x[4]
        g[3] = x[1]*x[4] + 1
        g[4] = x[1]*(x[1]+x[2]+x[3])
    end

    # g1 = x1*x2*x3*x4 - 25 >= 0  → bl=0, bu=Inf
    # g2 = x1^2+x2^2+x3^2+x4^2 - 40 = 0
    function eval_con!(c, x)
        c[1] = x[1]*x[2]*x[3]*x[4]
        c[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    end
    function eval_jac!(jnz, x)
        # Column-major order matching J sparsity below
        # col 1: rows 1,2
        jnz[1] = x[2]*x[3]*x[4]   # ∂g1/∂x1
        jnz[2] = 2x[1]             # ∂g2/∂x1
        # col 2: rows 1,2
        jnz[3] = x[1]*x[3]*x[4]
        jnz[4] = 2x[2]
        # col 3: rows 1,2
        jnz[5] = x[1]*x[2]*x[4]
        jnz[6] = 2x[3]
        # col 4: rows 1,2
        jnz[7] = x[1]*x[2]*x[3]
        jnz[8] = 2x[4]
    end

    # Dense 2×4 Jacobian sparsity
    J = sparse(
        Int32[1,2,1,2,1,2,1,2],   # row indices
        Int32[1,1,2,2,3,3,4,4],   # col indices
        ones(8), 2, 4
    )

    objfun = make_objfun(eval_obj, eval_grad!, ws.iw; callback=progress)
    confun = make_confun(eval_con!, eval_jac!, J, ws.iw; callback=progress)

    x0    = [1.0, 5.0, 5.0, 1.0]
    bl_x  = ones(4)
    bu_x  = 5 * ones(4)
    bl_c  = [25.0, 40.0]
    bu_c  = [1e20, 40.0]

    prob = make_constrained_prob(ws, x0, bl_x, bu_x, bl_c, bu_c, objfun, confun, J)

    status = snopt!(prob)

    @test status == 1
    @test prob.obj_val    ≈ 17.0140  atol=1e-3
    @test prob.ws.x[1]   ≈ 1.0      atol=1e-3
    @test prob.ws.x[2]   ≈ 4.7430   atol=1e-3
    @test prob.ws.x[3]   ≈ 3.8211   atol=1e-3
    @test prob.ws.x[4]   ≈ 1.3791   atol=1e-3
    @test prob.x[1]      ≈ 1.0      atol=1e-3
    @test prob.x[2]      ≈ 4.7430   atol=1e-3
    @test prob.x[3]      ≈ 3.8211   atol=1e-3
    @test prob.x[4]      ≈ 1.3791   atol=1e-3
    @test any(event -> event.kind === :objective, events)
    @test any(event -> event.kind === :constraint && length(event.c) == 2, events)
end

@testset "HS71 with SnoptC (combined callback)" begin
    ws = make_ws()
    events = NamedTuple[]
    progress = event -> begin push!(events, event); true end

    function eval_obj_c(x)
        x[1]*x[4]*(x[1]+x[2]+x[3]) + x[3]
    end
    function eval_grad_c!(g, x)
        g[1] = x[4]*(2x[1]+x[2]+x[3])
        g[2] = x[1]*x[4]
        g[3] = x[1]*x[4] + 1
        g[4] = x[1]*(x[1]+x[2]+x[3])
    end
    function eval_con_c!(c, x)
        c[1] = x[1]*x[2]*x[3]*x[4]
        c[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    end
    function eval_jac_c!(jnz, x)
        jnz[1] = x[2]*x[3]*x[4]
        jnz[2] = 2x[1]
        jnz[3] = x[1]*x[3]*x[4]
        jnz[4] = 2x[2]
        jnz[5] = x[1]*x[2]*x[4]
        jnz[6] = 2x[3]
        jnz[7] = x[1]*x[2]*x[3]
        jnz[8] = 2x[4]
    end

    J = sparse(
        Int32[1,2,1,2,1,2,1,2],
        Int32[1,1,2,2,3,3,4,4],
        ones(8), 2, 4
    )

    usrfun = make_usrfun_c(eval_obj_c, eval_grad_c!, eval_con_c!, eval_jac_c!, J,
                           ws.iw; callback=progress)
    prob = make_constrained_prob_c(
        ws,
        [1.0, 5.0, 5.0, 1.0],
        ones(4),
        5 * ones(4),
        [25.0, 40.0],
        [1e20, 40.0],
        usrfun,
        J
    )

    status = snoptc!(prob)
    @test status == 1
    @test prob.obj_val ≈ 17.0140 atol=1e-3
    @test prob.x[1] ≈ 1.0 atol=1e-3
    @test prob.x[2] ≈ 4.7430 atol=1e-3
    @test prob.x[3] ≈ 3.8211 atol=1e-3
    @test prob.x[4] ≈ 1.3791 atol=1e-3
    @test any(event -> event.kind === :combined && length(event.c) == 2, events)
end

# ---------------------------------------------------------------------------
# 8. Bound-constrained  min (x+3)^2, x ∈ [-1, 5]  → solution x=-1 (active lb)
# ---------------------------------------------------------------------------
@testset "Bound-constrained (active lower bound)" begin
    ws = make_ws()

    objfun = make_objfun(
        x -> (x[1]+3)^2,
        (g, x) -> begin g[1] = 2*(x[1]+3) end,
        ws.iw
    )
    confun = make_dummy_confun()

    prob = make_unconstrained_prob(
        ws,
        [2.0],         # x0 (interior)
        [-1.0],        # lower bound
        [ 5.0],        # upper bound
        objfun, confun
    )

    status = snopt!(prob)

    @test status == 1
    @test prob.ws.x[1] ≈ -1.0  atol=1e-6
    @test prob.x[1] ≈ -1.0  atol=1e-6
    # Multiplier for active lower bound is positive (SNOPT sign convention)
    @test prob.lambda[1] >= 0
end

@testset "Legacy log_fn compatibility" begin
    ws = make_ws()
    called = Ref(false)

    objfun = make_objfun(
        x -> (x[1]-1)^2,
        (g, x) -> begin g[1] = 2*(x[1]-1) end,
        (iter, x, f) -> begin called[] = true; true end,
        ws.iw
    )
    confun = make_dummy_confun()

    prob = make_unconstrained_prob(
        ws, [5.0], [-10.0], [10.0], objfun, confun
    )

    status = snopt!(prob)
    @test status == 1
    @test called[]
end

@testset "User-requested stop via progress callback" begin
    ws = make_ws()
    called = Ref(false)

    objfun = make_objfun(
        x -> (x[1]-1)^2,
        (g, x) -> begin g[1] = 2*(x[1]-1) end,
        ws.iw;
        callback=event -> begin called[] = true; false end
    )
    confun = make_dummy_confun()

    prob = make_unconstrained_prob(
        ws, [5.0], [-10.0], [10.0], objfun, confun
    )

    status = snopt!(prob)
    @test called[]
    @test status ∈ keys(Snopt.SNOPT_STATUS)   # some valid inform code
    @test Snopt.SNOPT_STATUS[status] === :User_Requested_Stop
end
