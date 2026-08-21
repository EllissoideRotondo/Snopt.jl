const SNOPT_INF = 1.0e20

function snopt_bound_value(value)
    value = Float64(value)
    value == Inf && return SNOPT_INF
    value == -Inf && return -SNOPT_INF
    return value
end

function reject_nan(vector::Vector{Float64}, name::AbstractString)
    any(isnan, vector) &&
        throw(ArgumentError("$name must not contain NaN"))
    return vector
end

function float_vector(values, name::AbstractString)
    values === nothing && throw(ArgumentError("$name must be provided"))
    values isa Number && throw(ArgumentError("$name must be a vector, not a scalar"))
    return reject_nan(snopt_bound_value.(collect(values)), name)
end

function bound_vector(values, n::Int, default::Float64, name::AbstractString)
    values === nothing && return fill(default, n)
    values isa Number && return reject_nan(fill(snopt_bound_value(values), n), name)
    vector = reject_nan(snopt_bound_value.(collect(values)), name)
    length(vector) == n ||
        throw(ArgumentError("$name must have length $n; got $(length(vector))"))
    return vector
end

function dummy_jacobian_sparsity(n::Int)
    colptr = Int32.(vcat(1, fill(2, n)))
    return SparseMatrixCSC{Float64,Int32}(1, n, colptr, Int32[1], Float64[0.0])
end

function dense_jacobian_sparsity(nc::Int, n::Int)
    colptr = Vector{Int32}(undef, n + 1)
    rowval = Vector{Int32}(undef, nc * n)
    nzval = zeros(Float64, nc * n)
    next = 1
    colptr[1] = Int32(next)
    for j in 1:n
        for i in 1:nc
            rowval[next] = Int32(i)
            next += 1
        end
        colptr[j + 1] = Int32(next)
    end
    return SparseMatrixCSC{Float64,Int32}(nc, n, colptr, rowval, nzval)
end

function jacobian_sparsity32(J::SparseMatrixCSC, nc::Int, n::Int)
    size(J) == (nc, n) ||
        throw(ArgumentError("J must have size ($nc, $n); got $(size(J))"))
    return SparseMatrixCSC{Float64,Int32}(
        nc, n, Int32.(J.colptr), Int32.(J.rowval), Float64.(J.nzval))
end

function prepare_jacobian_sparsity(J, nc::Int, n::Int)
    nc == 0 && return dummy_jacobian_sparsity(n)
    J === nothing && return dense_jacobian_sparsity(nc, n)
    J isa SparseMatrixCSC ||
        throw(ArgumentError("J must be a SparseMatrixCSC Jacobian sparsity pattern"))
    return jacobian_sparsity32(J, nc, n)
end

function prepare_constraint_data(eval_con, eval_jac, lcon, ucon)
    no_bounds = lcon === nothing && ucon === nothing
    no_callbacks = eval_con === nothing && eval_jac === nothing
    no_bounds && no_callbacks && return (0, Float64[], Float64[])
    (lcon === nothing || ucon === nothing) &&
        throw(ArgumentError("both lcon and ucon must be provided for constrained problems"))
    lcon_vector = float_vector(lcon, "lcon")
    ucon_vector = float_vector(ucon, "ucon")
    length(lcon_vector) == length(ucon_vector) ||
        throw(ArgumentError("lcon and ucon must have the same length"))
    if isempty(lcon_vector)
        no_callbacks ||
            throw(ArgumentError("constraint callbacks were provided, but lcon/ucon are empty"))
        return (0, Float64[], Float64[])
    end
    eval_con === nothing &&
        throw(ArgumentError("eval_con must be provided for constrained problems"))
    eval_jac === nothing &&
        throw(ArgumentError("eval_jac must be provided for constrained problems"))
    return (length(lcon_vector), lcon_vector, ucon_vector)
end

function snopt_result(prob::SnoptB, memory::SnoptMemory)
    status = Int(prob.status)
    status_symbol = get(SNOPT_STATUS, status, :Unknown_Status)
    lambda_length = prob.n + prob.nc
    basis = SnoptBasis(copy(prob.hs), prob.nS, prob.n, prob.m_eff)
    return SnoptResult(status, status_symbol, prob.obj_val, copy(prob.x[1:prob.n]),
                       copy(prob.lambda[1:lambda_length]),
                       prob.ws.num_inf, prob.ws.sum_inf,
                       prob.ws.iterations, prob.ws.major_itns, prob.ws.run_time,
                       memory, basis)
end

function preflight_callbacks!(eval_obj, eval_grad,
                              eval_con, eval_jac, x::Vector{Float64},
                              xlow::Vector{Float64}, xupp::Vector{Float64},
                              nc::Int, J::SparseMatrixCSC, callback)
    # SNOPT projects the starting point into the variable bounds before its
    # first evaluation; do the same here so a domain-limited objective is not
    # probed at an out-of-bounds x0 that the solver itself would never visit.
    xcheck = clamp.(x, xlow, xupp)
    f = eval_obj(xcheck)
    if callback !== nothing
        event = (kind = :objective, mode = 0, major_iter = 0, minor_iter = 0,
                 x = copy(xcheck), f = f)
        call_progress(callback, event) ||
            return (status = 71, objective = Float64(f), x = copy(xcheck))
    end
    gcheck = zeros(length(xcheck))
    eval_grad(gcheck, xcheck)
    if nc > 0
        ccheck = zeros(nc)
        eval_con(ccheck, xcheck)
        if callback !== nothing
            event = (kind = :constraint, mode = 0, major_iter = 0, minor_iter = 0,
                     x = copy(xcheck), c = copy(ccheck))
            call_progress(callback, event) ||
                return (status = 71, objective = Float64(f), x = copy(xcheck))
        end
        jcheck = zeros(nnz(J))
        eval_jac(jcheck, xcheck)
    end
    return nothing
end

function preflight_stop_result(stop, n::Int, nc::Int, memory::SnoptMemory)
    status = Int(stop.status)
    return SnoptResult(
        status,
        get(SNOPT_STATUS, status, :Unknown_Status),
        stop.objective,
        stop.x,
        zeros(n + nc),
        0,
        0.0,
        0,
        0,
        0.0,
        memory,
        SnoptBasis(zeros(Int32, n + max(nc, 1)), 0, n, max(nc, 1))
    )
end

"""
    snopt(eval_obj, eval_grad, x0; kwargs...) -> SnoptResult

Solve through SNOPT's `snOptB` interface and return a [`SnoptResult`](@ref).

`eval_obj(x)` returns one real objective value. `eval_grad(gradient, x)` must
fill every objective-gradient entry. `x0` must be nonempty and finite.

Constrained problems require `eval_con`, `eval_jac`, `lcon`, and `ucon`.
`eval_con(values, x)` fills every constraint value. `eval_jac(nonzeros, x)`
fills derivatives in the column-major storage order of `J`.

# Keywords

- `lb`, `ub`: Scalar or vector variable bounds. Bounds may contain `Inf`.
- `eval_con`, `eval_jac`: In-place constraint and Jacobian callbacks.
- `lcon`, `ucon`: Constraint bounds with equal lengths.
- `J`: Optional sparse Jacobian structure. The default structure is dense.
- `options`: SNOPT settings as a vector of pairs.
- `callback`: Evaluation event callback. Return `false` to request a stop.
- `snlog`: [`SnoptMajorLog`](@ref) callback for major-iteration progress.
- `snstop`: [`SnoptStopEvent`](@ref) callback for custom stopping rules.
- `start`: `"Cold"` or `"Warm"`. A warm start requires `basis`.
- `basis`: [`SnoptBasis`](@ref) from a compatible previous result.
- `printfile`, `summfile`: SNOPT output paths. Empty strings suppress output.
- `name`: Problem name with at most eight characters.

The function rejects `start = "Hot"`. A hot start needs one reused low-level
workspace. See the [Low-level interface](@ref).

SNOPT owns one active workspace per process. This function holds the package
lock across workspace sizing, initialization, options, and the solve.

"""
function snopt(eval_obj, eval_grad,
               x0::AbstractVector{<:Real};
               lb=nothing, ub=nothing,
               eval_con=nothing, eval_jac=nothing,
               lcon=nothing, ucon=nothing,
               J=nothing,
               options=nothing,
               callback=nothing,
               snlog=nothing,
               snstop=nothing,
               printfile::String = "",
               summfile::String = "",
               start::String = "Cold",
               basis=nothing,
               name::String = "Julia")
    x0_vector = Float64.(collect(x0))
    n = length(x0_vector)
    n > 0 || throw(ArgumentError("x0 must contain at least one variable"))
    all(isfinite, x0_vector) ||
        throw(ArgumentError("x0 must contain only finite values"))
    xlow = bound_vector(lb, n, -SNOPT_INF, "lb")
    xupp = bound_vector(ub, n, SNOPT_INF, "ub")
    nc, lcon_vector, ucon_vector =
        prepare_constraint_data(eval_con, eval_jac, lcon, ucon)
    m_eff = nc > 0 ? nc : 1
    J32 = prepare_jacobian_sparsity(J, nc, n)
    neJ = nnz(J32)
    negCon = nc > 0 ? neJ : 0
    nnCon = nc
    nnJac = nc > 0 ? n : 0
    nnObj = n
    # Hold the SNOPT lock across the whole solve, not merely across the
    # individual calls that take it themselves. The workspace-sizing estimate,
    # the workspace, its options, and the solve are one transaction: releasing
    # the lock between them would let another task's `initialize` close this
    # workspace mid-flight.
    hs_start, nS_start = prepare_start_basis(basis, start, n, m_eff)
    return lock(SNOPT_LOCK) do
        snopt_locked(eval_obj, eval_grad, x0_vector, xlow, xupp, nc,
                     lcon_vector, ucon_vector, eval_con, eval_jac, J32, m_eff,
                     neJ, negCon, nnCon, nnObj, nnJac, options, callback, snlog,
                     snstop, printfile, summfile, start, name, n, hs_start,
                     nS_start)
    end
end

# A warm start is only meaningful with the basis SNOPT ended a previous solve
# with; without it SNOPT would restart from a zeroed basis, which is a cold
# start wearing a different name.
function prepare_start_basis(basis, start::AbstractString, n::Int, m_eff::Int)
    key = lowercase(strip(start))
    if key == "cold"
        basis === nothing ||
            throw(ArgumentError("basis is only meaningful with start = \"Warm\""))
        return zeros(Int32, n + m_eff), 0
    end
    # SNOPT's Hot start reuses the LU factors and reduced-Hessian state living
    # inside the workspace of the previous solve. The high-level entry point
    # builds a fresh workspace per call, so that state never exists here and
    # SNOPT reads uninitialized memory (observed as a segfault in lu6sol/lu6u).
    # Hot starts are only possible at the low-level interface, reusing the same
    # workspace across solves.
    key == "hot" &&
        throw(ArgumentError("start = \"Hot\" is not supported by the high-level " *
                            "snopt: each call uses a fresh workspace, and SNOPT's " *
                            "hot start needs factorization state from the previous " *
                            "solve's workspace. Use start = \"Warm\" with a basis, " *
                            "or drive the low-level interface with one workspace."))
    basis === nothing &&
        throw(ArgumentError("start = $(repr(start)) requires `basis` from a previous SnoptResult"))
    basis isa SnoptBasis ||
        throw(ArgumentError("basis must be a SnoptBasis; got $(typeof(basis))"))
    basis.n == n ||
        throw(ArgumentError("basis was built for n = $(basis.n); this problem has n = $n"))
    basis.m == m_eff ||
        throw(ArgumentError("basis was built for m = $(basis.m); this problem has m = $m_eff"))
    length(basis.hs) == n + m_eff ||
        throw(ArgumentError("basis hs must have length n + m = $(n + m_eff); " *
                            "got $(length(basis.hs))"))
    return copy(basis.hs), basis.nS
end

function snopt_locked(eval_obj, eval_grad, x0_vector, xlow, xupp, nc,
                      lcon_vector, ucon_vector, eval_con, eval_jac, J32, m_eff,
                      neJ, negCon, nnCon, nnObj, nnJac, options, callback, snlog,
                      snstop, printfile, summfile, start, name, n, hs_start,
                      nS_start)
    memory = check_memory_estimate(
        snmemb(m_eff, n, neJ, negCon, nnCon, nnObj, nnJac;
               options, printfile, summfile))
    preflight_stop = preflight_callbacks!(
        eval_obj, eval_grad, eval_con, eval_jac, x0_vector, xlow, xupp,
        nc, J32, callback)
    preflight_stop !== nothing &&
        return preflight_stop_result(preflight_stop, n, nc, memory)
    ws = initialize(printfile, summfile, memory.miniw, memory.minrw)
    try
        apply_options!(ws, options)
        objfun = make_objfun(eval_obj, eval_grad, ws.iw; callback)
        confun = nc > 0 ? make_confun(eval_con, eval_jac, J32, ws.iw; callback) :
                          make_dummy_confun()
        x = [x0_vector; zeros(m_eff)]
        bl = [xlow; nc > 0 ? lcon_vector : [-SNOPT_INF]]
        bu = [xupp; nc > 0 ? ucon_vector : [SNOPT_INF]]
        prob = SnoptB(ws, n, nc, m_eff, n, x, bl, bu, hs_start, J32,
                      0.0, 0, Float64[], objfun, confun, nS_start)
        snoptb!(prob; start, name, snlog, snstop)
        return snopt_result(prob, memory)
    finally
        free!(ws)
    end
end
