function float_vector(values, name::AbstractString)
    values === nothing && throw(ArgumentError("$name must be provided"))
    values isa Number && throw(ArgumentError("$name must be a vector, not a scalar"))
    return Float64.(collect(values))
end

function bound_vector(values, n::Int, default::Float64, name::AbstractString)
    values === nothing && return fill(default, n)
    values isa Number && return fill(Float64(values), n)
    vector = Float64.(collect(values))
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
    return SnoptResult(status, status_symbol, prob.obj_val, copy(prob.x[1:prob.n]),
                       copy(prob.lambda[1:lambda_length]),
                       prob.ws.num_inf, prob.ws.sum_inf,
                       prob.ws.iterations, prob.ws.major_itns, prob.ws.run_time,
                       memory)
end

"""
    snopt(eval_obj, eval_grad, x0; kwargs...) -> SnoptResult
Solve a nonlinear optimization problem with SNOPTB using Julia callbacks.
`eval_obj(x)` returns the scalar objective and `eval_grad(g, x)` fills the
objective gradient.
Keyword arguments:
  * `lb`, `ub`: variable lower/upper bounds; scalars are broadcast.
  * `eval_con`, `eval_jac`, `lcon`, `ucon`: optional nonlinear constraints.
    `eval_con(c, x)` fills constraint values and `eval_jac(jnz, x)` fills the
    Jacobian nonzeros in the column-major order of `J`.
  * `J`: optional sparse constraint-Jacobian sparsity. If omitted for a
    constrained problem, a dense sparsity pattern is used.
  * `options`: SNOPT options as a vector of pairs, for example
    `["Major print level" => 0, :minor_print_level => 0]`.
  * `callback`: optional callback receiving objective/constraint events.
  * `snlog`: optional callback receiving `SnoptMajorLog` major-iteration events.

"""

function snopt(eval_obj::Function, eval_grad::Function,
               x0::AbstractVector{<:Real};
               lb=nothing, ub=nothing,
               eval_con=nothing, eval_jac=nothing,
               lcon=nothing, ucon=nothing,
               J=nothing,
               options=nothing,
               callback=nothing,
               snlog=nothing,
               printfile::String = "",
               summfile::String = "",
               start::String = "Cold",
               name::String = "Julia")
    x0_vector = Float64.(collect(x0))
    n = length(x0_vector)
    n > 0 || throw(ArgumentError("x0 must contain at least one variable"))
    xlow = bound_vector(lb, n, -Inf, "lb")
    xupp = bound_vector(ub, n, Inf, "ub")
    nc, lcon_vector, ucon_vector =
        prepare_constraint_data(eval_con, eval_jac, lcon, ucon)
    m_eff = nc > 0 ? nc : 1
    J32 = prepare_jacobian_sparsity(J, nc, n)
    neJ = nnz(J32)
    negCon = nc > 0 ? neJ : 0
    nnCon = nc
    nnJac = nc > 0 ? n : 0
    nnObj = n
    memory = check_memory_estimate(
        snmemb(m_eff, n, neJ, negCon, nnCon, nnJac, nnObj;
               options, printfile, summfile))
    ws = initialize(printfile, summfile, memory.miniw, memory.minrw)
    try
        apply_options!(ws, options)
        objfun = make_objfun(eval_obj, eval_grad, ws.iw; callback)
        confun = nc > 0 ? make_confun(eval_con, eval_jac, J32, ws.iw; callback) :
                          make_dummy_confun()
        x = [x0_vector; zeros(m_eff)]
        bl = [xlow; nc > 0 ? lcon_vector : [-Inf]]
        bu = [xupp; nc > 0 ? ucon_vector : [Inf]]
        hs = zeros(Int32, n + m_eff)
        prob = SnoptB(ws, n, nc, m_eff, x, bl, bu, hs, J32,
                      0.0, 0, Float64[], objfun, confun)
        snoptb!(prob; start, name, snlog)
        return snopt_result(prob, memory)
    finally
        free!(ws)
    end
end
