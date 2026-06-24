"""
    specs_status_message(info::Int) -> String

Return a human-readable description of the inform code returned by [`read_options`](@ref)
when reading a SNOPT specs file (e.g. `101` is success; codes above `134` indicate that
`info - 134` errors were found while parsing).
"""
function specs_status_message(info::Int)
    info == 101 && return "Specs file read successfully."
    info == 131 && return "No Specs file specified (iSpecs ≤ 0 or iSpecs > 99)."  # unreachable via f_snspecf filename path
    info == 132 && return "End-of-file while looking for Specs file (Begin not found)."
    info == 133 && return "End-of-file before finding End."
    info == 134 && return "Endrun found before any valid options."
    info > 134  && return "$(info - 134) error(s) while reading Specs file."
    return "Unknown specs inform code: $info."
end

"""
    read_options(prob, specsfile::String) -> Int

Read a SNOPT specs (options) file into the workspace of `prob` (a
[`SnoptWorkspace`](@ref) or any [`AbstractSnoptProblem`](@ref)) via SNOPT's `snSpecf`,
and return the resulting inform code. A warning is emitted unless the file is read
cleanly (`101`); see [`specs_status_message`](@ref) for the code meanings. Use this as
an alternative to passing options programmatically through [`set_option!`](@ref).
"""
function read_options(prob::SnoptWorkspace, specsfile::String)
    require_open_workspace(prob, "read_options")
    isfile(specsfile) ||
        throw(ArgumentError("read_options: specs file not found: $(repr(specsfile))"))
    status = Int32[0]
    ccall((:f_snspecf, libsnopt7), Cvoid,
          (Cstring, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          specsfile, Cint(ncodeunits(specsfile)), status,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    info = Int(status[1])
    prob.status = info
    info != 101 && @warn "read_options: $(specs_status_message(info))"
    return info
end

read_options(prob::AbstractSnoptProblem, specsfile::String) =
    read_options(prob.ws, specsfile)

function require_dimension(condition::Bool, message::AbstractString)
    condition || throw(DimensionMismatch(String(message)))
    return nothing
end

"""
    snopta!(prob::SnoptA; start="Cold", name="Julia") -> Int

Solve a [`SnoptA`](@ref) problem in place through SNOPT's `snOptA` interface and
return the SNOPT inform code. The final point is written into `prob.x`, multipliers
into `prob.xmul`/`prob.Fmul`, and the row values into `prob.F`. If the problem was
built without a derivative function, SNOPT must be configured for finite-difference
gradients (`set_option!(ws, "Derivative option", 0)`).
"""
function snopta!(prob::SnoptA; start::String = "Cold", name::String = "Julia")
    require_open_workspace(prob.ws, "snopta!")
    require_dimension(
        prob.n == length(prob.x) == length(prob.xlow) == length(prob.xupp),
        "SnoptA variable arrays must all have length n=$(prob.n)")
    require_dimension(
        prob.n == length(prob.xstate) == length(prob.xmul),
        "SnoptA variable state and multiplier arrays must have length n=$(prob.n)")
    require_dimension(
        prob.nf == length(prob.F) == length(prob.flow) == length(prob.fupp),
        "SnoptA function arrays must all have length nf=$(prob.nf)")
    require_dimension(
        prob.nf == length(prob.Fstate) == length(prob.Fmul),
        "SnoptA function state and multiplier arrays must have length nf=$(prob.nf)")
    require_dimension(
        length(prob.iAfun) == length(prob.jAvar) == length(prob.A),
        "SnoptA linear Jacobian row, column, and value arrays must have equal lengths")
    require_dimension(
        length(prob.iGfun) == length(prob.jGvar),
        "SnoptA nonlinear Jacobian row and column arrays must have equal lengths")
    prob.ws.iu = Int32[0]
    prob.ws.ru = [0.0]
    usrfun = prob.usrfun
    usr_callback = snopt_callback_pointer(SNOPTA_CALLBACK_PTR)
    status = Int32[0]
    nS     = Int32[prob.nS]
    nInf   = Int32[0]
    sInf   = [0.0]
    miniw  = Int32[0]
    minrw  = Int32[0]
    active_callbacks = ActiveSnoptACallbacks(usrfun)
    with_active_snopt_callbacks(prob.ws, active_callbacks) do
        reset_callback_exception!(usrfun)
        GC.@preserve usrfun begin
            ccall((:f_snopta, libsnopt7), Cvoid,
                  (Cint, Cstring,
                   Cint, Cint, Cdouble, Cint,
                   Ptr{Cvoid},
                   Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cdouble},
                   Ptr{Cint}, Ptr{Cint}, Cint,
                   Ptr{Cdouble}, Ptr{Cdouble},
                   Ptr{Cdouble}, Ptr{Cdouble},
                   Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble},
                   Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble},
                   Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                   Ptr{Cint}, Ptr{Cint},
                   Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
                   Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
                  start_mode_code(start), name,
                  prob.nf, prob.n, prob.objadd, prob.objrow,
                  usr_callback,
                  prob.iAfun, prob.jAvar, length(prob.A), prob.A,
                  prob.iGfun, prob.jGvar, length(prob.iGfun),
                  prob.xlow, prob.xupp,
                  prob.flow, prob.fupp,
                  prob.x, prob.xstate, prob.xmul,
                  prob.F, prob.Fstate, prob.Fmul,
                  status, nS, nInf, sInf,
                  miniw, minrw,
                  prob.ws.iu, prob.ws.leniu, prob.ws.ru, prob.ws.lenru,
                  prob.ws.iw, prob.ws.leniw, prob.ws.rw, prob.ws.lenrw)
        end
        rethrow_callback_exception!(usrfun)
        rethrow_active_callback_exception!(active_callbacks)
    end
    prob.status = Int(status[1])
    prob.nS = Int(nS[1])
    prob.num_inf = Int(nInf[1])
    prob.sum_inf = sInf[1]
    prob.ws.status = prob.status
    prob.ws.x = copy(prob.x)
    prob.ws.lambda = copy(prob.xmul)
    prob.ws.obj_val = prob.objrow > 0 ? prob.F[prob.objrow] + prob.objadd : prob.objadd
    prob.ws.num_inf = prob.num_inf
    prob.ws.sum_inf = prob.sum_inf
    prob.ws.iterations = workspace_value(prob.ws.iw, IW_MINOR_ITNS)
    prob.ws.major_itns = workspace_value(prob.ws.iw, IW_MAJOR_ITNS)
    prob.ws.run_time = workspace_value(prob.ws.rw, RW_RUN_TIME)
    return prob.status
end

function snoptb!(prob::SnoptWorkspace, start::String, name::String,
                 m::Int, n::Int, nnCon::Int, nnObj::Int, nnJac::Int,
                 fObj::Float64, iObj::Int,
                 confun::Function, objfun::Function,
                 J::SparseMatrixCSC, bl::Vector{Float64}, bu::Vector{Float64},
                 hs::Vector{Int32}, x::Vector{Float64};
                 snlog=nothing)
    require_open_workspace(prob, "snoptb!")
    total = n + m
    require_dimension(
        total == length(x) == length(bl) == length(bu),
        "SNOPTB x, lower-bound, and upper-bound arrays must have length n + m = $total")
    require_dimension(
        total == length(hs),
        "SNOPTB basis-status array must have length n + m = $total")
    prob.iu = Int32[0]
    prob.ru = [0.0]
    prob.x      = copy(x)
    prob.lambda = zeros(Float64, n + m)
    pi_         = zeros(Float64, m)
    obj_callback = snopt_callback_pointer(SNOPTB_OBJ_CALLBACK_PTR)
    con_callback = snopt_callback_pointer(SNOPTB_CON_CALLBACK_PTR)
    valJ = copy(J.nzval)
    indJ = convert(Array{Cint}, J.rowval)
    locJ = convert(Array{Cint}, J.colptr)
    neJ  = length(valJ)
    status  = Int32[0]
    nS      = Int32[0]
    nInf    = Int32[0]
    sInf    = [0.0]
    obj_val = [0.0]
    miniw   = Int32[0]
    minrw   = Int32[0]
    start_code = start_mode_code(start)
    if snlog === nothing
        active_callbacks = ActiveSnoptBCallbacks(confun, objfun)
        with_active_snopt_callbacks(prob, active_callbacks) do
            reset_callback_exception!(confun, objfun)
            GC.@preserve confun objfun begin
                ccall((:f_snoptb, libsnopt7), Cvoid,
                      (Cint, Cstring,
                       Cint, Cint, Cint, Cint, Cint, Cint,
                       Cint, Cdouble,
                       Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint},
                       Ptr{Float64}, Ptr{Float64}, Ptr{Cint},
                       Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                       Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},
                       Ptr{Cint}, Ptr{Cint},
                       Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
                       Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
                      start_code, name, m, n, neJ, nnCon, nnObj, nnJac,
                      iObj, fObj,
                      con_callback, obj_callback,
                      valJ, indJ, locJ,
                      bl, bu, hs, prob.x, pi_, prob.lambda,
                      status, nS, nInf, sInf, obj_val,
                      miniw, minrw,
                      prob.iu, prob.leniu, prob.ru, prob.lenru,
                      prob.iw, prob.leniw, prob.rw, prob.lenrw)
            end
            rethrow_callback_exception!(confun, objfun)
            rethrow_active_callback_exception!(active_callbacks)
        end
    else
        snlog_fn = make_snlog(snlog)
        snlog_callback = snopt_callback_pointer(SNOPT_SNLOG_CALLBACK_PTR)
        null_callback = Ptr{Cvoid}(C_NULL)
        active_callbacks = ActiveSnoptBCallbacks(confun, objfun; snlog=snlog_fn)
        with_active_snopt_callbacks(prob, active_callbacks) do
            reset_callback_exception!(confun, objfun, snlog_fn)
            GC.@preserve confun objfun snlog_fn begin
                ccall((:f_snkerb, libsnopt7), Cvoid,
                      (Cint, Cstring,
                       Cint, Cint, Cint, Cint, Cint, Cint,
                       Cint, Cdouble,
                       Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint},
                       Ptr{Float64}, Ptr{Float64}, Ptr{Cint},
                       Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                       Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},
                       Ptr{Cint}, Ptr{Cint},
                       Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
                       Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
                      start_code, name, m, n, neJ, nnCon, nnObj, nnJac,
                      iObj, fObj,
                      con_callback, obj_callback,
                      snlog_callback, null_callback, null_callback, null_callback,
                      valJ, indJ, locJ,
                      bl, bu, hs, prob.x, pi_, prob.lambda,
                      status, nS, nInf, sInf, obj_val,
                      miniw, minrw,
                      prob.iu, prob.leniu, prob.ru, prob.lenru,
                      prob.iw, prob.leniw, prob.rw, prob.lenrw)
            end
            rethrow_callback_exception!(confun, objfun, snlog_fn)
            rethrow_active_callback_exception!(active_callbacks)
        end
    end
    prob.status  = status[1]
    prob.obj_val = obj_val[1]
    prob.num_inf = nInf[1]
    prob.sum_inf = sInf[1]
    prob.iterations = workspace_value(prob.iw, IW_MINOR_ITNS)
    prob.major_itns = workspace_value(prob.iw, IW_MAJOR_ITNS)
    prob.run_time   = workspace_value(prob.rw, RW_RUN_TIME)
    copyto!(x, prob.x)
    return Int(prob.status)
end

"""
    snoptc!(prob::SnoptC; start="Cold", name="Julia", snlog=nothing) -> Int

Solve a [`SnoptC`](@ref) problem in place through SNOPT's `snOptC` interface and
return the SNOPT inform code. The final point, objective, and multipliers are written
back into `prob`. Pass `snlog` to receive a [`SnoptMajorLog`](@ref) at each major
iteration (routing the solve through SNOPT's `snKerC` kernel).
"""
function snoptc!(prob::SnoptC; start::String = "Cold", name::String = "Julia",
                 snlog=nothing)
    require_open_workspace(prob.ws, "snoptc!")
    total = prob.n + prob.m_eff
    require_dimension(
        total == length(prob.x) == length(prob.bl) == length(prob.bu),
        "SnoptC x, lower-bound, and upper-bound arrays must have length n + m_eff = $total")
    require_dimension(
        total == length(prob.hs),
        "SnoptC basis-status array must have length n + m_eff = $total")
    prob.ws.iu = Int32[0]
    prob.ws.ru = [0.0]
    prob.ws.x      = copy(prob.x)
    prob.ws.lambda = zeros(Float64, prob.n + prob.m_eff)
    pi_            = zeros(Float64, prob.m_eff)
    usrfun = prob.usrfun
    usr_callback = snopt_callback_pointer(SNOPTC_CALLBACK_PTR)
    valJ = copy(prob.J.nzval)
    indJ = convert(Array{Cint}, prob.J.rowval)
    locJ = convert(Array{Cint}, prob.J.colptr)
    neJ  = length(valJ)
    status  = Int32[0]
    nS      = Int32[0]
    nInf    = Int32[0]
    sInf    = [0.0]
    obj_val = [0.0]
    miniw   = Int32[0]
    minrw   = Int32[0]
    nnCon = prob.nc
    nnJac = prob.nc > 0 ? prob.n : 0
    start_code = start_mode_code(start)
    if snlog === nothing
        active_callbacks = ActiveSnoptCCallbacks(usrfun)
        with_active_snopt_callbacks(prob.ws, active_callbacks) do
            reset_callback_exception!(usrfun)
            GC.@preserve usrfun begin
                ccall((:f_snoptc, libsnopt7), Cvoid,
                      (Cint, Cstring,
                       Cint, Cint, Cint, Cint, Cint, Cint,
                       Cint, Cdouble,
                       Ptr{Cvoid},
                       Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint},
                       Ptr{Float64}, Ptr{Float64}, Ptr{Cint},
                       Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                       Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},
                       Ptr{Cint}, Ptr{Cint},
                       Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
                       Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
                      start_code, name,
                      prob.m_eff, prob.n, neJ, nnCon, prob.nnobj, nnJac,
                      0, 0.0,
                      usr_callback,
                      valJ, indJ, locJ,
                      prob.bl, prob.bu, prob.hs, prob.ws.x, pi_, prob.ws.lambda,
                      status, nS, nInf, sInf, obj_val,
                      miniw, minrw,
                      prob.ws.iu, prob.ws.leniu, prob.ws.ru, prob.ws.lenru,
                      prob.ws.iw, prob.ws.leniw, prob.ws.rw, prob.ws.lenrw)
            end
            rethrow_callback_exception!(usrfun)
            rethrow_active_callback_exception!(active_callbacks)
        end
    else
        snlog_fn = make_snlog(snlog)
        snlog_callback = snopt_callback_pointer(SNOPT_SNLOG_CALLBACK_PTR)
        null_callback = Ptr{Cvoid}(C_NULL)
        active_callbacks = ActiveSnoptCCallbacks(usrfun; snlog=snlog_fn)
        with_active_snopt_callbacks(prob.ws, active_callbacks) do
            reset_callback_exception!(usrfun, snlog_fn)
            GC.@preserve usrfun snlog_fn begin
                ccall((:f_snkerc, libsnopt7), Cvoid,
                      (Cint, Cstring,
                       Cint, Cint, Cint, Cint, Cint, Cint,
                       Cint, Cdouble,
                       Ptr{Cvoid},
                       Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint},
                       Ptr{Float64}, Ptr{Float64}, Ptr{Cint},
                       Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                       Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},
                       Ptr{Cint}, Ptr{Cint},
                       Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
                       Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
                      start_code, name,
                      prob.m_eff, prob.n, neJ, nnCon, prob.nnobj, nnJac,
                      0, 0.0,
                      usr_callback,
                      snlog_callback, null_callback, null_callback, null_callback,
                      valJ, indJ, locJ,
                      prob.bl, prob.bu, prob.hs, prob.ws.x, pi_, prob.ws.lambda,
                      status, nS, nInf, sInf, obj_val,
                      miniw, minrw,
                      prob.ws.iu, prob.ws.leniu, prob.ws.ru, prob.ws.lenru,
                      prob.ws.iw, prob.ws.leniw, prob.ws.rw, prob.ws.lenrw)
            end
            rethrow_callback_exception!(usrfun, snlog_fn)
            rethrow_active_callback_exception!(active_callbacks)
        end
    end
    prob.status  = Int(status[1])
    prob.obj_val = obj_val[1]
    prob.lambda  = prob.ws.lambda
    prob.ws.status = prob.status
    prob.ws.obj_val = prob.obj_val
    prob.ws.num_inf = Int(nInf[1])
    prob.ws.sum_inf = sInf[1]
    prob.ws.iterations = workspace_value(prob.ws.iw, IW_MINOR_ITNS)
    prob.ws.major_itns = workspace_value(prob.ws.iw, IW_MAJOR_ITNS)
    prob.ws.run_time   = workspace_value(prob.ws.rw, RW_RUN_TIME)
    copyto!(prob.x, prob.ws.x)
    return prob.status
end
