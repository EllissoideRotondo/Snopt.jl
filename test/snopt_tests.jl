using Libdl
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
    return SnoptB(ws, n, 0, m_eff, n, x, bl, bu, hs, J, 0.0, 0, Float64[], objfun, confun)
end

function make_constrained_prob(ws, x0, bl_x, bu_x, bl_c, bu_c, objfun, confun, J)
    n    = length(x0)
    nc   = size(J, 1)
    m    = nc
    x    = [x0; zeros(m)]
    bl   = [bl_x; bl_c]
    bu   = [bu_x; bu_c]
    hs   = zeros(Int32, n + m)
    return SnoptB(ws, n, nc, m, n, x, bl, bu, hs, J, 0.0, 0, Float64[], objfun, confun)
end

function make_constrained_prob_c(ws, x0, bl_x, bu_x, bl_c, bu_c, usrfun, J)
    n    = length(x0)
    nc   = size(J, 1)
    m    = nc
    x    = [x0; zeros(m)]
    bl   = [bl_x; bl_c]
    bu   = [bu_x; bu_c]
    hs   = zeros(Int32, n + m)
    return SnoptC(ws, n, nc, m, n, x, bl, bu, hs, J, 0.0, 0, Float64[], usrfun)
end

mutable struct SnoptLogCollector
    logs::Vector{SnoptMajorLog}
end

function (collector::SnoptLogCollector)(event)
    push!(collector.logs, event)
    return true
end

# Collects snSTOP events and optionally asks SNOPT to stop once `stop_after`
# major iterations have been seen.
mutable struct SnoptStopCollector
    events::Vector{SnoptStopEvent}
    stop_after::Int
end

SnoptStopCollector() = SnoptStopCollector(SnoptStopEvent[], typemax(Int))

function (collector::SnoptStopCollector)(event)
    push!(collector.events, event)
    return event.major_iter < collector.stop_after
end

# HS71: min x1 x4 (x1 + x2 + x3) + x3  s.t.  x1 x2 x3 x4 >= 25, sum(x.^2) == 40.
hs71_obj(x) = x[1]*x[4]*(x[1]+x[2]+x[3]) + x[3]

function hs71_grad!(g, x)
    g[1] = x[4]*(2x[1]+x[2]+x[3])
    g[2] = x[1]*x[4]
    g[3] = x[1]*x[4] + 1
    g[4] = x[1]*(x[1]+x[2]+x[3])
    return nothing
end

function hs71_con!(c, x)
    c[1] = x[1]*x[2]*x[3]*x[4]
    c[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    return nothing
end

function hs71_jac!(jnz, x)
    jnz[1] = x[2]*x[3]*x[4]; jnz[2] = 2x[1]
    jnz[3] = x[1]*x[3]*x[4]; jnz[4] = 2x[2]
    jnz[5] = x[1]*x[2]*x[4]; jnz[6] = 2x[3]
    jnz[7] = x[1]*x[2]*x[3]; jnz[8] = 2x[4]
    return nothing
end

hs71_sparsity() = sparse(Int32[1,2,1,2,1,2,1,2], Int32[1,1,2,2,3,3,4,4],
                         ones(8), 2, 4)

solve_hs71(; kwargs...) = snopt(
    hs71_obj, hs71_grad!, [1.0, 5.0, 5.0, 1.0];
    lb = ones(4), ub = 5 * ones(4),
    eval_con = hs71_con!, eval_jac = hs71_jac!,
    lcon = [25.0, 40.0], ucon = [1e20, 40.0],
    J = hs71_sparsity(),
    options = ["Major print level" => 0, "Minor print level" => 0],
    kwargs...)

@testset "Workspace initialization" begin
    ws = initialize("", "")
    @test ws isa SNOPT.SnoptWorkspace
    @test ws.leniw > 0
    @test ws.lenrw == 60000
    @test_throws ArgumentError SNOPT.SnoptWorkspace(0, 10)
    @test_throws ArgumentError SNOPT.SnoptWorkspace(10, 0)
    # Workspaces below SNOPT's 500-element minimum let f_sninitx write out of
    # bounds (heap corruption / segfault); they must be rejected up front.
    @test_throws ArgumentError SNOPT.SnoptWorkspace(499, 1000)
    @test_throws ArgumentError SNOPT.SnoptWorkspace(1000, 499)
    @test_throws ArgumentError initialize("", "", 100, 100)

    ws2 = initialize("", "", 40000, 5000)
    @test ws2.leniw == 40000
    @test ws2.lenrw == 5000
    @test isopen(ws2)
    close(ws2)
    @test !isopen(ws2)
    close(ws2)
    @test !isopen(ws2)

    block_ws = Ref{Any}(nothing)
    block_result = initialize("", "", 1000, 1000) do ws
        block_ws[] = ws
        @test isopen(ws)
        :ok
    end
    @test block_result === :ok
    @test block_ws[] isa SNOPT.SnoptWorkspace
    @test !isopen(block_ws[])

    thrown_ws = Ref{Any}(nothing)
    @test_throws ErrorException initialize("", "", 1000, 1000) do ws
        thrown_ws[] = ws
        @test isopen(ws)
        error("workspace block failed")
    end
    @test thrown_ws[] isa SNOPT.SnoptWorkspace
    @test !isopen(thrown_ws[])

    @test SnoptProblem === SnoptB
    @test SnoptA <: AbstractSnoptProblem
    @test SnoptB <: AbstractSnoptProblem
    @test SnoptC <: AbstractSnoptProblem
end

@testset "Closed workspaces reject public operations" begin
    ws = make_ws()
    objfun = make_objfun(
        x -> (x[1] - 1)^2,
        (g, x) -> begin g[1] = 2(x[1] - 1) end,
        ws.iw
    )
    prob = make_unconstrained_prob(
        ws, [0.0], [-10.0], [10.0], objfun, make_dummy_confun()
    )
    specs = tempname()
    write(specs, "Begin\n  Major print level 0\nEnd\n")

    close(ws)
    @test !isopen(ws)
    @test_throws ArgumentError set_option!(ws, "Major print level", 0)
    @test_throws ArgumentError SNOPT.apply_options!(ws, ["Major print level" => 0])
    @test_throws ArgumentError snmemb(ws, 1, 1, 1, 0, 0, 1, 0)
    @test_throws ArgumentError read_options(ws, specs)
    @test_throws ArgumentError snopt!(prob)
end

@testset "Library discovery" begin
    libdir = dirname(SNOPT.libsnopt7)
    original_path = get(ENV, "PATH", "")
    withenv("SNOPTDIR" => libdir, "SNOPT_GFORTRAN_BINDIR" => "not-used") do
        @test normpath(SNOPT.find_snopt_lib()) == normpath(SNOPT.libsnopt7)
        @test get(ENV, "PATH", "") == original_path
    end
    mktempdir() do dir
        withenv("SNOPTDIR" => dir) do
            # A bad SNOPTDIR now warns and falls back to the platform library
            # path and the system loader, so on a machine with a system-wide
            # libsnopt7 the search still succeeds.
            found = @test_logs (:warn, r"SNOPTDIR is set but no loadable") match_mode=:any begin
                SNOPT.find_snopt_lib()
            end
            syslib = Libdl.dlopen_e(string("lib", "snopt7", ".", Libdl.dlext))
            if syslib == C_NULL
                @test found == ""
            else
                Libdl.dlclose(syslib)
                @test !isempty(found)
            end
        end
    end
end

@testset "Library discovery rejects a library without the f_* interface" begin
    @test :f_sninitx in SNOPT.REQUIRED_SNOPT_SYMBOLS
    @test :f_snoptb in SNOPT.REQUIRED_SNOPT_SYMBOLS
    # The probe must cover every f_* symbol the package actually ccalls, or a
    # library missing one of them would pass has_snopt() and fail mid-solve.
    called = Set{Symbol}()
    for file in readdir(dirname(pathof(SNOPT)); join = true)
        endswith(file, ".jl") || continue
        for m in eachmatch(r":(f_[a-z]+)", read(file, String))
            push!(called, Symbol(m.captures[1]))
        end
    end
    @test issubset(called, Set(SNOPT.REQUIRED_SNOPT_SYMBOLS))
    # libm loads fine but exports none of SNOPT's C shims.
    fake = string("libm.", Libdl.dlext)
    @test SNOPT.loadable_library_path(fake) == ""
    # The real library still passes.
    @test SNOPT.loadable_library_path(SNOPT.libsnopt7) == SNOPT.libsnopt7
end

@testset "SNOPTB memory estimation" begin
    ws = make_ws()
    @test SNOPT.SNOPT_MEMORY_WORKSPACE >= 1000
    memory = snmemb(ws, 1, 2, 1, 0, 0, 2, 0)
    @test memory isa SnoptMemory
    @test memory.info == 104
    @test memory.miniw >= 500
    @test memory.minrw >= 500

    @test_throws ArgumentError snmemb(ws, 0, 2, 1, 0, 0, 2, 0)
    @test_throws ArgumentError snmemb(ws, 1, 0, 1, 0, 0, 0, 0)
    @test_throws ArgumentError snmemb(ws, 1, 2, -1, 0, 0, 2, 0)
    @test_throws ArgumentError snmemb(ws, 1, 2, 1, -1, 0, 2, 0)
    @test_throws ArgumentError snmemb(ws, 1, 2, 1, 0, -1, 2, 0)
    @test_throws ArgumentError snmemb(ws, 1, 2, 1, 0, 0, -1, 0)
    @test_throws ArgumentError snmemb(ws, 1, 2, 1, 0, 0, 2, -1)
    @test_throws ArgumentError snmemb(ws, 1, 2, 1, 0, 0, 3, 0)
    @test_throws ArgumentError snmemb(ws, 1, 2, 1, 0, 0, 2, 3)
    @test_throws ArgumentError snmemb(ws, 1, 2, 1, 0, 2, 2, 0)
    @test_throws ArgumentError snmemb(ws, 1, 2, 1, 2, 0, 2, 0)

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

    @test SNOPT.apply_options!(ws, [
        "Major print level" => 0,
        :minor_print_level => 0,
        "Major feasibility tolerance" => 1.0e-8,
        :hessian => :full_memory,
    ]) === ws

    @test_throws ArgumentError SNOPT.apply_options!(ws, "Major print level 0")
    @test_throws ArgumentError SNOPT.apply_options!(ws, "Major print level" => 0)
    @test_throws ArgumentError SNOPT.apply_options!(ws, Dict("Major print level" => 0))
    @test_throws ArgumentError SNOPT.apply_options!(ws, ("Major print level" => 0,))
    @test_throws ArgumentError SNOPT.apply_options!(ws, ["Major print level 0"])
    @test_throws ArgumentError SNOPT.apply_options!(ws, ["Major print level" => false])
    @test_throws ArgumentError SNOPT.apply_options!(ws, ["Major feasibility tolerance" => Inf])
    @test_throws ArgumentError SNOPT.apply_options!(ws, ["Major print level" => ""])
    @test_throws ArgumentError SNOPT.apply_options!(ws, [1 => 0])

    @test_throws ArgumentError snmemb(2, 4, 8, 8, 2, 4, 4;
                                     options = ["Major print level 0"])
end

@testset "initialize guards missing library" begin
    original_libsnopt7 = SNOPT.libsnopt7
    try
        @eval SNOPT libsnopt7 = ""
        @test_throws ErrorException initialize("", "", 1000, 1000)
    finally
        @eval SNOPT libsnopt7 = $original_libsnopt7
    end
end

@testset "public validation and callback exceptions" begin
    ws = make_ws()
    objfun = make_objfun(
        x -> (x[1] - 1)^2,
        (g, x) -> begin g[1] = 2(x[1] - 1) end,
        ws.iw
    )
    @test objfun isa Function
    @test SNOPT.callback_state(objfun) isa SNOPT.SnoptCallbackState
    @test SNOPT.active_snopt_callback_count() == 0
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
    @test SNOPT.active_snopt_callback_count() == 0
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
    @test prob.ws.iu == Int32[0]
    @test prob.ws.leniu == 0
    @test SNOPT.active_snopt_callback_count() == 0
end

@testset "snLog major iteration callback on SnoptA" begin
    ws = make_ws()
    set_option!(ws, "Derivative option", 1)
    collector = SnoptLogCollector(SnoptMajorLog[])
    usrfun = make_usrfun_a(
        (F, x) -> begin F[1] = (x[1] - 2)^2 + (x[2] - 3)^2 end;
        eval_G = (G, x) -> begin G[1] = 2(x[1] - 2); G[2] = 2(x[2] - 3) end
    )
    prob = SnoptA(
        ws, 1, 2, 0.0, 1,
        Int32[], Int32[], Float64[],
        Int32[1, 1], Int32[1, 2],
        [-10.0, -10.0], [10.0, 10.0],
        [-1.0e20], [1.0e20],
        [0.0, 0.0], zeros(Int32, 2), zeros(2),
        zeros(1), zeros(Int32, 1), zeros(1),
        0, 0, 0, 0.0,
        usrfun
    )
    status = snopta!(prob; snlog = collector)
    @test status == 1
    @test prob.x[1] ≈ 2.0 atol = 1.0e-4
    @test prob.x[2] ≈ 3.0 atol = 1.0e-4
    @test !isempty(collector.logs)
    @test collector.logs[end] isa SnoptMajorLog
    @test SNOPT.active_snopt_callback_count() == 0
end

@testset "SnoptA finite-difference gradients (eval_G=nothing)" begin
    ws = make_ws()
    set_option!(ws, "Derivative option", 0)

    # No eval_G: SNOPT must estimate the gradient by finite differences even
    # though it still issues a needG>0 probe call.
    usrfun = make_usrfun_a((F, x) -> begin F[1] = (x[1] - 2)^2 end)

    prob = SnoptA(
        ws,
        1, 1,
        0.0, 1,
        Int32[], Int32[], Float64[],
        Int32[1], Int32[1],
        [-10.0], [10.0],
        [-1.0e20], [1.0e20],
        [0.0],
        zeros(Int32, 1),
        zeros(1),
        zeros(1),
        zeros(Int32, 1),
        zeros(1),
        0, 0, 0, 0.0,
        usrfun
    )

    status = snopta!(prob)
    @test status == 1
    @test prob.x[1] ≈ 2.0 atol=1.0e-4
    @test SNOPT.active_snopt_callback_count() == 0
end

@testset "SNOPT_STATUS dictionary" begin
    @test SNOPT.SNOPT_STATUS[1]   == :Solve_Succeeded
    @test SNOPT.SNOPT_STATUS[2]   == :Feasible_Point_Found
    @test SNOPT.SNOPT_STATUS[11]  == :Infeasible_Problem_Detected
    @test SNOPT.SNOPT_STATUS[21]  == :Unbounded_Problem_Detected
    @test SNOPT.SNOPT_STATUS[31]  == :Maximum_Iterations_Exceeded
    @test SNOPT.SNOPT_STATUS[32]  == :Maximum_Iterations_Exceeded
    # 33 is "the superbasics limit is too small", not an iteration limit.
    @test SNOPT.SNOPT_STATUS[33]  == :Superbasics_Limit_Too_Small
    @test SNOPT.SNOPT_STATUS[34]  == :Maximum_CpuTime_Exceeded
    @test SNOPT.SNOPT_STATUS[41]  == :Numerical_Difficulties
    @test SNOPT.SNOPT_STATUS[71]  == :User_Requested_Stop
    @test SNOPT.SNOPT_STATUS[81]  == :Insufficient_Memory
    @test SNOPT.SNOPT_STATUS[84]  == :Insufficient_Memory
    @test SNOPT.SNOPT_STATUS[141] == :Internal_Error
    @test SNOPT.SNOPT_STATUS[142] == :Internal_Error
    @test SNOPT.SNOPT_STATUS[999] == :Internal_Error
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

    # A missing specs file is reported clearly instead of as a SNOPT parse error.
    @test_throws ArgumentError read_options(ws, joinpath(@__DIR__, "no_such_specsfile"))
end

@testset "Low-level snopt API (unconstrained)" begin
    calls = Ref(0)
    grad_calls = Ref(0)
    stopped = snopt(
        x -> (x[1] - 1)^2,
        (g, x) -> begin
            grad_calls[] += 1
            g[1] = 2(x[1] - 1)
        end,
        [0.0];
        options = [
            "Major print level" => 0,
            "Minor print level" => 0,
        ],
        callback = event -> begin
            calls[] += 1
            false
        end
    )
    @test stopped.status == 71
    @test stopped.status_symbol === :User_Requested_Stop
    @test stopped.x == [0.0]
    @test stopped.objective ≈ 1.0
    @test calls[] == 1
    @test grad_calls[] == 0

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

@testset "Default workspace handles medium unconstrained problem" begin
    ws = make_ws()
    n = 50
    target = collect(1.0:n)
    objfun = make_objfun(
        x -> sum(abs2, x .- target),
        (g, x) -> begin g .= 2 .* (x .- target) end,
        ws.iw
    )
    prob = make_unconstrained_prob(
        ws,
        zeros(n),
        fill(-100.0, n),
        fill(100.0, n),
        objfun,
        make_dummy_confun()
    )
    status = snopt!(prob)
    @test status == 1
    @test prob.obj_val ≈ 0.0 atol=1.0e-6
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
    @test prob.ws.iu == Int32[0]
    @test prob.ws.leniu == 0
    @test SNOPT.active_snopt_callback_count() == 0
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
    @test prob.ws.iu == Int32[0]
    @test prob.ws.leniu == 0
    @test SNOPT.active_snopt_callback_count() == 0
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
    original_jnz = copy(J.nzval)

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
    @test J.nzval == original_jnz
end

@testset "HS71 with SnoptC (combined callback)" begin
    ws = make_ws()
    events = NamedTuple[]
    progress = event -> begin push!(events, event); true end
    collector = SnoptLogCollector(SnoptMajorLog[])

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

    status = snoptc!(prob; snlog = collector)
    @test status == 1
    @test prob.obj_val ≈ 17.0140 atol=1e-3
    @test prob.x[1] ≈ 1.0 atol=1e-3
    @test prob.x[2] ≈ 4.7430 atol=1e-3
    @test prob.x[3] ≈ 3.8211 atol=1e-3
    @test prob.x[4] ≈ 1.3791 atol=1e-3
    @test any(event -> event.kind === :combined && length(event.c) == 2, events)
    @test !isempty(collector.logs)
    @test collector.logs[end] isa SnoptMajorLog
    @test prob.ws.iu == Int32[0]
    @test prob.ws.leniu == 0
    @test SNOPT.active_snopt_callback_count() == 0
end

@testset "Suppressed output survives an unwritable temp directory" begin
    silent = ["Major print level" => 0, "Minor print level" => 0]
    solve_once() = snopt(
        x -> (x[1] + 3)^2,
        (g, x) -> begin g[1] = 2(x[1] + 3) end,
        [2.0]; lb = [-1.0], ub = [5.0], options = silent
    )
    @test solve_once().status == 1

    mktempdir() do dir
        readonly = joinpath(dir, "readonly")
        mkdir(readonly)
        chmod(readonly, 0o500)   # r-x: cannot create files
        try
            withenv("TMPDIR" => readonly) do
                # The scratch file must be created somewhere writable: never an
                # uncreated path, and never silently downgraded to the null
                # device, which leaves SNOPT stateful across later solves.
                _, summpath, _ = SNOPT.snopt_output_files("", "")
                @test summpath != SNOPT.SNOPT_DEVNULL
                @test isfile(summpath)
                rm(summpath; force = true)

                ws = initialize("", "")
                @test isopen(ws)
                close(ws)
            end
        finally
            chmod(readonly, 0o700)   # let mktempdir clean up
        end
    end

    # The session must still be healthy afterwards. This is the assertion that
    # actually caught the bad null-device fallback.
    @test solve_once().status == 1
end

@testset "Jacobian shape validation" begin
    ws = make_ws()
    objfun = make_objfun(
        x -> (x[1] - 1)^2 + (x[2] - 2)^2,
        (g, x) -> begin g[1] = 2(x[1] - 1); g[2] = 2(x[2] - 2) end,
        ws.iw
    )
    n, m_eff = 2, 1
    x  = [0.0, 0.0, 0.0]
    bl = [-10.0, -10.0, -1.0e20]
    bu = [10.0, 10.0, 1.0e20]
    hs = zeros(Int32, n + m_eff)
    # J claims a single column while the problem has n = 2, so J.colptr is one
    # element short of the locJ(n+1) that SNOPT reads.
    J_bad = SparseMatrixCSC{Float64,Int32}(1, 1, Int32[1, 2], Int32[1], Float64[0.0])
    prob = SnoptB(ws, n, 0, m_eff, n, x, bl, bu, hs, J_bad,
                  0.0, 0, Float64[], objfun, make_dummy_confun())
    @test_throws DimensionMismatch snoptb!(prob)

    J_bad_c = SparseMatrixCSC{Float64,Int32}(1, 1, Int32[1, 2], Int32[1], Float64[0.0])
    usrfun = make_usrfun_c(
        x -> x[1]^2,
        (g, x) -> begin fill!(g, 0.0); g[1] = 2x[1] end,
        (c, x) -> begin c[1] = x[1] end,
        (jnz, x) -> fill!(jnz, 0.0),
        J_bad_c, ws.iw
    )
    probc = SnoptC(ws, n, 1, 1, n, x, bl, bu, hs, J_bad_c,
                   0.0, 0, Float64[], usrfun)
    @test_throws DimensionMismatch snoptc!(probc)
end

@testset "SnoptC rejects inconsistent Jacobian sparsity" begin
    ws = make_ws()
    set_option!(ws, "Derivative option", 3)
    J_actual = sparse(
        Int32[1,2,1,2,1,2,1,2],
        Int32[1,1,2,2,3,3,4,4],
        ones(8), 2, 4
    )
    J_mismatched = sparse(
        Int32[1],
        Int32[1],
        ones(1), 2, 4
    )
    usrfun = make_usrfun_c(
        x -> x[1]^2,
        (g, x) -> begin fill!(g, 0.0); g[1] = 2x[1] end,
        (c, x) -> begin c[1] = x[1]; c[2] = x[2] end,
        (jnz, x) -> fill!(jnz, 0.0),
        J_mismatched,
        ws.iw
    )
    prob = make_constrained_prob_c(
        ws,
        [1.0, 5.0, 5.0, 1.0],
        ones(4),
        5 * ones(4),
        [0.0, 0.0],
        [1e20, 1e20],
        usrfun,
        J_actual
    )
    @test_throws DimensionMismatch snoptc!(prob)
    @test prob.ws.iu == Int32[0]
    @test prob.ws.leniu == 0
    @test SNOPT.active_snopt_callback_count() == 0
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

@testset "Superbasics count is retained after a solve" begin
    ws = make_ws()
    objfun = make_objfun(
        x -> (x[1] - 1)^2 + (x[2] - 2)^2,
        (g, x) -> begin g[1] = 2(x[1] - 1); g[2] = 2(x[2] - 2) end,
        ws.iw
    )
    prob = make_unconstrained_prob(
        ws, [0.0, 0.0], fill(-10.0, 2), fill(10.0, 2), objfun, make_dummy_confun()
    )
    @test prob.nS == 0
    @test snoptb!(prob) == 1
    # The value SNOPT returned must reach both the problem and the workspace
    # rather than being discarded. Do not assert a particular count: how many
    # variables end up superbasic is a solver detail. The warm-start test is
    # what proves the retained value is useful.
    @test prob.nS == prob.ws.nS
    @test prob.nS >= 0
    @test prob.ws.nS >= 0
end

struct ShiftedQuadratic
    target::Vector{Float64}
end
(q::ShiftedQuadratic)(x) = sum(abs2, x .- q.target)

struct ShiftedGradient
    target::Vector{Float64}
end
(q::ShiftedGradient)(g, x) = (g .= 2 .* (x .- q.target); nothing)

@testset "Callable structs work as objective and gradient" begin
    target = [1.0, 2.0]
    result = snopt(
        ShiftedQuadratic(target), ShiftedGradient(target), [0.0, 0.0];
        lb = -10.0, ub = 10.0,
        options = ["Major print level" => 0, "Minor print level" => 0]
    )
    @test result.status == 1
    @test result.x ≈ target atol = 1.0e-5
end

@testset "Concurrent solves are serialized" begin
    if Threads.nthreads() > 1
        results = Vector{Any}(undef, 8)
        Threads.@threads for i in 1:8
            results[i] = snopt(
                x -> (x[1] - 1)^2 + (x[2] - 2)^2,
                (g, x) -> begin g[1] = 2(x[1] - 1); g[2] = 2(x[2] - 2) end,
                [0.0, 0.0];
                lb = -10.0, ub = 10.0,
                options = ["Major print level" => 0, "Minor print level" => 0]
            )
        end
        @test all(r -> r.status == 1, results)
        @test all(r -> isapprox(r.x, [1.0, 2.0]; atol = 1.0e-5), results)
    else
        @info "single-threaded session; skipping concurrency test"
        @test true
    end
end

@testset "NaN input validation" begin
    silent_options = ["Major print level" => 0, "Minor print level" => 0]
    f = x -> (x[1] - 1)^2
    g! = (g, x) -> begin g[1] = 2(x[1] - 1) end
    @test_throws ArgumentError snopt(f, g!, [NaN]; options = silent_options)
    @test_throws ArgumentError snopt(f, g!, [Inf]; options = silent_options)
    @test_throws ArgumentError snopt(f, g!, [0.0]; lb = [NaN], options = silent_options)
    @test_throws ArgumentError snopt(f, g!, [0.0]; ub = [NaN], options = silent_options)
    @test_throws ArgumentError snopt(f, g!, [0.0]; lb = NaN, options = silent_options)
    @test_throws ArgumentError snopt(
        f, g!, [0.0];
        eval_con = (c, x) -> begin c[1] = x[1] end,
        eval_jac = (jnz, x) -> begin jnz[1] = 1.0 end,
        lcon = [NaN], ucon = [1.0],
        options = silent_options
    )
end

@testset "set_option! rejects non-finite values" begin
    ws = make_ws()
    @test_throws ArgumentError set_option!(ws, "Major feasibility tolerance", NaN)
    @test_throws ArgumentError set_option!(ws, "Major feasibility tolerance", Inf)
    @test_throws ArgumentError set_option!(ws, "Major feasibility tolerance", -Inf)
    # A finite value still works after the rejections.
    @test set_option!(ws, "Major feasibility tolerance", 1e-7) == 0
end

@testset "Preflight clamps x0 into bounds" begin
    # sqrt is undefined below 0; SNOPT projects x0 into [lb, ub] before its
    # first evaluation, and the preflight check must do the same instead of
    # probing the raw out-of-bounds x0.
    result = snopt(
        x -> sqrt(x[1]),
        (g, x) -> begin g[1] = 0.5 / sqrt(x[1]) end,
        [-5.0];
        lb = [1.0], ub = [10.0],
        options = ["Major print level" => 0, "Minor print level" => 0]
    )
    @test result.status == 1
    @test result.x[1] ≈ 1.0 atol = 1e-6
end

@testset "Warm start reuses a basis and costs no more iterations" begin
    silent = ["Major print level" => 0, "Minor print level" => 0]
    f  = x -> (x[1] - 1)^2 + (x[2] - 2)^2 + 0.5 * (x[1] * x[2] - 2)^2
    g! = (g, x) -> begin
        g[1] = 2(x[1] - 1) + (x[1] * x[2] - 2) * x[2]
        g[2] = 2(x[2] - 2) + (x[1] * x[2] - 2) * x[1]
    end

    solved = snopt(f, g!, [0.0, 0.0]; lb = -10.0, ub = 10.0, options = silent)
    @test solved.status == 1
    @test solved.basis isa SnoptBasis
    @test length(solved.basis.hs) == 2 + 1
    @test solved.basis.n == 2
    @test solved.basis.m == 1

    perturbed = solved.x .+ 0.25
    cold = snopt(f, g!, perturbed; lb = -10.0, ub = 10.0, options = silent)
    warm = snopt(f, g!, perturbed; lb = -10.0, ub = 10.0, options = silent,
                 start = "Warm", basis = solved.basis)
    @test warm.status == 1
    @test warm.x ≈ cold.x atol = 1.0e-5
    @test warm.major_itns <= cold.major_itns

    # A basis whose dimensions do not match the problem is rejected.
    @test_throws ArgumentError snopt(f, g!, [0.0, 0.0, 0.0];
        lb = -10.0, ub = 10.0, options = silent,
        start = "Warm", basis = solved.basis)
    # A warm start without a basis is rejected rather than silently going cold.
    @test_throws ArgumentError snopt(f, g!, perturbed;
        lb = -10.0, ub = 10.0, options = silent, start = "Warm")
    # A basis passed to a cold start is a mistake worth reporting.
    @test_throws ArgumentError snopt(f, g!, perturbed;
        lb = -10.0, ub = 10.0, options = silent, basis = solved.basis)
    # A hot start is rejected up front: SNOPT's hot start reuses factorization
    # state from the previous solve's workspace, and the high-level entry point
    # builds a fresh workspace per call — attempting it segfaults inside SNOPT
    # (observed in lu6sol/lu6u on SNOPT 7.7.7).
    @test_throws ArgumentError snopt(f, g!, perturbed;
        lb = -10.0, ub = 10.0, options = silent,
        start = "Hot", basis = solved.basis)
end

@testset "set_option! accepts any Integer and Real width" begin
    ws = make_ws()
    @test set_option!(ws, "Major iterations limit", Int32(50)) == 0
    @test set_option!(ws, "Major iterations limit", UInt16(50)) == 0
    @test set_option!(ws, "Major optimality tolerance", Float32(1e-5)) == 0
    @test set_option!(ws, "Major optimality tolerance", 1 // 100000) == 0
    @test_throws ArgumentError set_option!(ws, "Major optimality tolerance", NaN32)
    close(ws)
end

@testset "Warm start (snOptA)" begin
    ws = make_ws()
    set_option!(ws, "Derivative option", 1)
    usrfun = make_usrfun_a(
        (F, x) -> begin F[1] = (x[1] - 2)^2 end;
        eval_G = (G, x) -> begin G[1] = 2(x[1] - 2) end
    )
    prob = SnoptA(
        ws, 1, 1, 0.0, 1,
        Int32[], Int32[], Float64[],
        Int32[1], Int32[1],
        [-10.0], [10.0],
        [-1.0e20], [1.0e20],
        [0.0], zeros(Int32, 1), zeros(1),
        zeros(1), zeros(Int32, 1), zeros(1),
        0, 0, 0, 0.0,
        usrfun
    )
    @test snopta!(prob) == 1
    @test prob.x[1] ≈ 2.0 atol = 1e-4
    # Re-solve from the solution with SNOPT's warm start (integer code 2 for
    # snOptA; code 1 would request a basis-file start).
    @test snopta!(prob; start = "Warm") == 1
    @test prob.x[1] ≈ 2.0 atol = 1e-4
    @test_throws ArgumentError snopta!(prob; start = "Tepid")
end

@testset "Direct SnoptWorkspace close without initialize" begin
    # A workspace that never went through f_sninitx (init_id == 0) must not
    # call f_snend on its zeroed work arrays during finalization.
    ws = SNOPT.SnoptWorkspace(500, 500)
    @test isopen(ws)
    @test ws.init_id == 0
    close(ws)
    @test !isopen(ws)
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
    @test status ∈ keys(SNOPT.SNOPT_STATUS)   # some valid inform code
    @test SNOPT.SNOPT_STATUS[status] === :User_Requested_Stop
end


@testset "snSTOP major iteration callback" begin
    stops = SnoptStopCollector()
    logs = SnoptLogCollector(SnoptMajorLog[])
    result = solve_hs71(snlog = logs, snstop = stops)

    @test result.status == 1
    @test result.objective ≈ 17.0140 atol = 1e-3
    @test !isempty(stops.events)
    @test all(e -> e isa SnoptStopEvent, stops.events)

    # SNOPT calls snSTOP once per major iteration, in order.
    @test [e.major_iter for e in stops.events] == sort(unique(e.major_iter for e in stops.events))
    @test stops.events[end].major_iter == result.major_itns

    # The dimensions SNOPT hands to snSTOP must describe this problem. These
    # assertions are what pins down the Fortran argument order: a shifted
    # argument list shows up here as garbage rather than as 4 variables and
    # 2 constraints.
    last = stops.events[end]
    @test last.n == 4
    @test last.m == 2
    @test last.nb == 6
    @test last.nncon == 2
    @test last.nnobj == 4
    @test last.negcon == 8
    @test last.minimize == 1
    @test last.max_superbasics >= last.n_superbasics

    # ...and so do the vectors, which must match values we can compute here.
    @test length(last.x) == last.nb
    @test last.x[1:4] ≈ result.x atol = 1e-8
    @test last.bl == [1.0, 1.0, 1.0, 1.0, 25.0, 40.0]
    @test last.bu[1:4] == fill(5.0, 4)
    expected_fcon = zeros(2); hs71_con!(expected_fcon, last.x[1:4])
    expected_gobj = zeros(4); hs71_grad!(expected_gobj, last.x[1:4])
    expected_gcon = zeros(8); hs71_jac!(expected_gcon, last.x[1:4])
    @test last.fcon ≈ expected_fcon
    @test last.fx ≈ expected_fcon
    @test last.gobj ≈ expected_gobj
    @test last.gcon ≈ expected_gcon
    @test length(last.ycon) == 2
    @test length(last.pi) == 2
    @test length(last.rc) == last.nb
    @test length(last.rg) == last.max_superbasics
    @test length(last.hs) == last.nb

    # snLog and snSTOP see the same major iteration, so their shared fields
    # must agree; this catches a drift in either argument list.
    @test length(logs.logs) == length(stops.events)
    for (log, stop) in zip(logs.logs, stops.events)
        @test log.major_iter == stop.major_iter
        @test log.minor_iter == stop.minor_iter
        @test log.iteration == stop.iteration
        @test log.n_superbasics == stop.n_superbasics
        @test log.objective == stop.objective
        @test log.primal_infeasibility == stop.primal_infeasibility
        @test log.dual_infeasibility == stop.dual_infeasibility
        @test log.step == stop.step
        @test log.x == stop.x
        @test log.hs == stop.hs
    end
    @test SNOPT.active_snopt_callback_count() == 0
end

@testset "snSTOP requests early termination" begin
    stops = SnoptStopCollector(SnoptStopEvent[], 2)
    result = solve_hs71(snstop = stops)

    @test SNOPT.SNOPT_STATUS[result.status] === :User_Requested_Stop
    @test stops.events[end].major_iter == 2
    @test result.major_itns <= 3
    @test SNOPT.active_snopt_callback_count() == 0
end

@testset "snSTOP propagates callback exceptions" begin
    thrown = ErrorException("snSTOP callback failed")
    @test_throws ErrorException solve_hs71(snstop = _ -> throw(thrown))
    @test SNOPT.active_snopt_callback_count() == 0
end

@testset "snSTOP on SnoptA and SnoptC" begin
    ws = make_ws()
    set_option!(ws, "Derivative option", 1)
    stops = SnoptStopCollector()
    usrfun = make_usrfun_a(
        (F, x) -> begin F[1] = (x[1] - 2)^2 + (x[2] - 3)^2 end;
        eval_G = (G, x) -> begin G[1] = 2(x[1] - 2); G[2] = 2(x[2] - 3) end
    )
    prob = SnoptA(
        ws, 1, 2, 0.0, 1,
        Int32[], Int32[], Float64[],
        Int32[1, 1], Int32[1, 2],
        [-10.0, -10.0], [10.0, 10.0],
        [-1.0e20], [1.0e20],
        [0.0, 0.0], zeros(Int32, 2), zeros(2),
        zeros(1), zeros(Int32, 1), zeros(1),
        0, 0, 0, 0.0,
        usrfun
    )
    @test snopta!(prob; snstop = stops) == 1
    @test prob.x[1] ≈ 2.0 atol = 1.0e-4
    @test !isempty(stops.events)
    @test stops.events[end].n == 2
    close(ws)

    ws_c = make_ws()
    stops_c = SnoptStopCollector()
    J = hs71_sparsity()
    usrfun_c = make_usrfun_c(hs71_obj, hs71_grad!, hs71_con!, hs71_jac!, J, ws_c.iw)
    prob_c = make_constrained_prob_c(
        ws_c, [1.0, 5.0, 5.0, 1.0], ones(4), 5 * ones(4),
        [25.0, 40.0], [1e20, 40.0], usrfun_c, J
    )
    @test snoptc!(prob_c; snstop = stops_c) == 1
    @test prob_c.obj_val ≈ 17.0140 atol = 1e-3
    @test !isempty(stops_c.events)
    @test stops_c.events[end].n == 4
    @test stops_c.events[end].m == 2
    @test SNOPT.active_snopt_callback_count() == 0
end
