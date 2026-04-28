mutable struct SnoptWorkspace
    status::Int
    finalized::Bool

    leniw::Int
    lenrw::Int
    iw::Vector{Int32}
    rw::Vector{Float64}

    leniu::Int
    lenru::Int
    iu::Vector{Int32}
    ru::Vector{Float64}

    x::Vector{Float64}
    lambda::Vector{Float64}
    obj_val::Float64

    num_inf::Int
    sum_inf::Float64

    iterations::Int
    major_itns::Int
    run_time::Float64

    function SnoptWorkspace(leniw::Int, lenrw::Int)
        leniw > 0 || throw(ArgumentError("leniw must be positive"))
        lenrw > 0 || throw(ArgumentError("lenrw must be positive"))
        prob = new(0, false, leniw, lenrw,
                   zeros(Int32, leniw), zeros(Float64, lenrw),
                   0, 0,
                   Int32[0], [0.0],
                   Float64[], Float64[], 0.0,
                   0, 0.0, 0, 0, 0.0)
        finalizer(free!, prob)
        prob
    end
end

mutable struct SnoptA{F<:Function}
    ws::SnoptWorkspace
    nf::Int                           # number of F rows: objective + constraints
    n::Int                            # number of design variables
    objadd::Float64                   # objective constant term
    objrow::Int                       # objective row, 0 means no objective row
    iAfun::Vector{Int32}              # linear Jacobian row indices
    jAvar::Vector{Int32}              # linear Jacobian variable indices
    A::Vector{Float64}                # linear Jacobian values
    iGfun::Vector{Int32}              # nonlinear Jacobian row indices
    jGvar::Vector{Int32}              # nonlinear Jacobian variable indices
    xlow::Vector{Float64}
    xupp::Vector{Float64}
    flow::Vector{Float64}
    fupp::Vector{Float64}
    x::Vector{Float64}
    xstate::Vector{Int32}
    xmul::Vector{Float64}
    F::Vector{Float64}
    Fstate::Vector{Int32}
    Fmul::Vector{Float64}
    status::Int
    nS::Int
    num_inf::Int
    sum_inf::Float64
    usrfun::F
end

mutable struct SnoptB{F1<:Function, F2<:Function}
    ws::SnoptWorkspace
    n::Int                            # num design variables
    nc::Int                           # num nonlinear constraints
    m_eff::Int                        # effective m passed to Fortran (>= 1; nc when nc>0, else 1)
    x::Vector{Float64}                # [n+m_eff] extended point (initial / final)
    bl::Vector{Float64}               # [n+m_eff] lower bounds
    bu::Vector{Float64}               # [n+m_eff] upper bounds
    hs::Vector{Int32}                 # [n+m_eff] basis status
    J::SparseMatrixCSC{Float64,Int32} # m_eff×n Jacobian (sparsity structure)
    obj_val::Float64                  # filled after solve
    status::Int                       # SNOPT inform code, filled after solve
    lambda::Vector{Float64}           # multipliers, filled after solve
    objfun::F1
    confun::F2
end

const SnoptProblem = SnoptB

mutable struct SnoptC{F<:Function}
    ws::SnoptWorkspace
    n::Int                            # num design variables
    nc::Int                           # num nonlinear constraints
    m_eff::Int                        # effective m passed to Fortran
    x::Vector{Float64}                # [n+m_eff] extended point (initial / final)
    bl::Vector{Float64}               # [n+m_eff] lower bounds
    bu::Vector{Float64}               # [n+m_eff] upper bounds
    hs::Vector{Int32}                 # [n+m_eff] basis status
    J::SparseMatrixCSC{Float64,Int32} # m_eff×n Jacobian (sparsity structure)
    obj_val::Float64                  # filled after solve
    status::Int                       # SNOPT inform code, filled after solve
    lambda::Vector{Float64}           # multipliers, filled after solve
    usrfun::F
end

function free!(prob::SnoptWorkspace)
    prob.finalized && return nothing
    prob.finalized = true
    isempty(libsnopt7) && return nothing
    try
        ccall((:f_snend, libsnopt7),
              Cvoid, (Ptr{Cint}, Cint, Ptr{Float64}, Cint),
              prob.iw, prob.leniw, prob.rw, prob.lenrw)
    catch
        # Finalizers may run during shutdown when the shared library is gone.
    end
    return nothing
end

function start_mode_code(start::AbstractString)::Cint
    key = lowercase(strip(start))
    key == "cold" && return Cint(0)
    key == "warm" && return Cint(1)
    key == "hot"  && return Cint(2)
    throw(ArgumentError("SNOPT start mode must be Cold, Warm, or Hot; got $(repr(start))"))
end

function snopt!(prob::SnoptB)
    return snoptb!(prob)
end

function snoptb!(prob::SnoptB)
    nc    = prob.nc
    nnCon = nc
    nnJac = nc > 0 ? prob.n : 0
    inform = snoptb!(prob.ws, "Cold", "Julia   ",
                     prob.m_eff, prob.n, nnCon, prob.n, nnJac,
                     0.0, 0,
                     prob.confun, prob.objfun,
                     prob.J, prob.bl, prob.bu, prob.hs, prob.x)
    prob.obj_val = prob.ws.obj_val
    prob.status  = inform
    prob.lambda  = prob.ws.lambda
    copyto!(prob.x, prob.ws.x)
    return inform
end

function snopt!(prob::SnoptA)
    return snopta!(prob)
end

function snopt!(prob::SnoptC)
    return snoptc!(prob)
end

snopt_no_progress(::NamedTuple) = true

workspace_value(ws_iw::Vector{Int32}, index::Int) =
    length(ws_iw) >= index ? max(Int(ws_iw[index]), 0) : 0

function progress_iters(ws_iw::Vector{Int32})
    return (workspace_value(ws_iw, IW_MAJOR_ITNS),
            workspace_value(ws_iw, IW_MINOR_ITNS))
end

call_progress(::Nothing, ::NamedTuple) = true
call_progress(callback, event::NamedTuple) = callback(event) !== false

legacy_progress_callback(log_fn::Function) =
    event -> event.kind === :objective ? log_fn(event.major_iter, event.x, event.f) : true

"""
    make_objfun(eval_obj, eval_grad, ws_iw; callback=nothing)

Create the SNOPT objective callback. If `callback` is provided, it is called as
`callback(event::NamedTuple)::Bool` after objective-value evaluations. Returning
`false` requests SNOPT termination. The default `nothing` records no progress
and avoids per-evaluation event allocation.

Objective events contain `kind = :objective`, `mode`, `major_iter`,
`minor_iter`, `x`, and `f`.
"""
function make_objfun(eval_obj::Function, eval_grad::Function,
                     ws_iw::Vector{Int32}; callback=nothing)
    function objfun(mode_::Ptr{Cint}, nnobj_::Ptr{Cint}, x_::Ptr{Cdouble},
                    f_::Ptr{Cdouble}, g_::Ptr{Cdouble}, _::Ptr{Cint},
                    _::Ptr{UInt8}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                    _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        mode  = unsafe_load(mode_)
        nnobj = Int(unsafe_load(nnobj_))
        x     = unsafe_wrap(Array, x_, nnobj)
        if mode == 0 || mode == 2
            f = eval_obj(x)
            unsafe_store!(f_, f)
            if callback !== nothing
                major_itns, minor_itns = progress_iters(ws_iw)
                event = (kind = :objective, mode = Int(mode),
                         major_iter = major_itns, minor_iter = minor_itns,
                         x = copy(x), f = f)
                if !call_progress(callback, event)
                    unsafe_store!(mode_, Cint(-2))
                    return
                end
            end
        end
        if mode == 1 || mode == 2
            g = unsafe_wrap(Array, g_, nnobj)
            eval_grad(g, x)
        end
        return
    end
    return objfun
end

function make_objfun(eval_obj::Function, eval_grad::Function, log_fn::Function,
                     ws_iw::Vector{Int32})
    return make_objfun(eval_obj, eval_grad, ws_iw;
                       callback=legacy_progress_callback(log_fn))
end

"""
    make_confun(eval_con, eval_jac, J, ws_iw; callback=nothing)

Create the SNOPT constraint callback. If `callback` is provided, it is called as
`callback(event::NamedTuple)::Bool` after constraint-value evaluations.
Returning `false` requests SNOPT termination.

Constraint events contain `kind = :constraint`, `mode`, `major_iter`,
`minor_iter`, `x`, and `c`. The default `nothing` records no progress and avoids
per-evaluation event allocation.
"""
function make_confun(eval_con::Function, eval_jac::Function, J,
                     ws_iw::Vector{Int32}; callback=nothing)
    function confun(mode_::Ptr{Cint}, nncon_::Ptr{Cint}, nnjac_::Ptr{Cint},
                    _::Ptr{Cint}, x_::Ptr{Cdouble}, c_::Ptr{Cdouble},
                    J_::Ptr{Cdouble}, _::Ptr{Cint},
                    _::Ptr{UInt8}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                    _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        mode  = unsafe_load(mode_)
        nncon = Int(unsafe_load(nncon_))
        nnjac = Int(unsafe_load(nnjac_))
        x     = unsafe_wrap(Array, x_, nnjac)
        if mode == 0 || mode == 2
            c = unsafe_wrap(Array, c_, nncon)
            eval_con(c, x)
            if callback !== nothing
                major_itns, minor_itns = progress_iters(ws_iw)
                event = (kind = :constraint, mode = Int(mode),
                         major_iter = major_itns, minor_iter = minor_itns,
                         x = copy(x), c = copy(c))
                if !call_progress(callback, event)
                    unsafe_store!(mode_, Cint(-2))
                    return
                end
            end
        end
        if mode == 1 || mode == 2
            jnzval = unsafe_wrap(Array, J_, nnz(J))
            eval_jac(jnzval, x)
        end
        return
    end
    return confun
end

make_confun(eval_con::Function, eval_jac::Function, J) =
    make_confun(eval_con, eval_jac, J, Int32[])

"""
    make_usrfun_a(eval_F; eval_G=nothing, callback=nothing)

Create the combined callback used by SNOPT-A. `eval_F(F, x)` fills the
objective/constraint row values. If derivatives are requested, `eval_G(G, x)`
fills the nonlinear derivative values in the order specified by `iGfun` and
`jGvar`. Returning `false` from `callback(event)` requests SNOPT termination.
"""
function make_usrfun_a(eval_F::Function; eval_G=nothing, callback=nothing)
    function usrfun(status_::Ptr{Cint}, n_::Ptr{Cint}, x_::Ptr{Cdouble},
                    needF_::Ptr{Cint}, neF_::Ptr{Cint}, F_::Ptr{Cdouble},
                    needG_::Ptr{Cint}, neG_::Ptr{Cint}, G_::Ptr{Cdouble},
                    _::Ptr{UInt8}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                    _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        n     = Int(unsafe_load(n_))
        needF = Int(unsafe_load(needF_))
        neF   = Int(unsafe_load(neF_))
        needG = Int(unsafe_load(needG_))
        neG   = Int(unsafe_load(neG_))
        x     = unsafe_wrap(Array, x_, n)

        if needF > 0
            F = unsafe_wrap(Array, F_, neF)
            eval_F(F, x)
            if callback !== nothing
                event = (kind = :function, x = copy(x), F = copy(F))
                if !call_progress(callback, event)
                    unsafe_store!(status_, Cint(-2))
                    return
                end
            end
        end
        if needG > 0
            if eval_G === nothing
                unsafe_store!(status_, Cint(-1))
                return
            end
            G = unsafe_wrap(Array, G_, neG)
            eval_G(G, x)
        end
        return
    end
    return usrfun
end

"""
    make_usrfun_c(eval_obj, eval_grad, eval_con, eval_jac, J, ws_iw; callback=nothing)

Create the combined callback used by SNOPT-C. This is the SNOPT-C equivalent
of `make_objfun` plus `make_confun`.
"""
function make_usrfun_c(eval_obj::Function, eval_grad::Function,
                       eval_con::Function, eval_jac::Function, J,
                       ws_iw::Vector{Int32}; callback=nothing)
    function usrfun(mode_::Ptr{Cint}, nnobj_::Ptr{Cint}, nncon_::Ptr{Cint},
                    nnjac_::Ptr{Cint}, nnL_::Ptr{Cint}, negcon_::Ptr{Cint},
                    x_::Ptr{Cdouble}, fobj_::Ptr{Cdouble}, gobj_::Ptr{Cdouble},
                    fcon_::Ptr{Cdouble}, gcon_::Ptr{Cdouble}, status_::Ptr{Cint},
                    _::Ptr{UInt8}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                    _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        mode   = Int(unsafe_load(mode_))
        nnobj  = Int(unsafe_load(nnobj_))
        nncon  = Int(unsafe_load(nncon_))
        nnjac  = Int(unsafe_load(nnjac_))
        nnL    = Int(unsafe_load(nnL_))
        negcon = Int(unsafe_load(negcon_))
        x      = unsafe_wrap(Array, x_, nnL)

        if mode == 0 || mode == 2
            f = eval_obj(x)
            unsafe_store!(fobj_, f)
            c = unsafe_wrap(Array, fcon_, nncon)
            eval_con(c, x)
            if callback !== nothing
                major_itns, minor_itns = progress_iters(ws_iw)
                event = (kind = :combined, mode = mode,
                         major_iter = major_itns, minor_iter = minor_itns,
                         x = copy(x), f = f, c = copy(c))
                if !call_progress(callback, event)
                    unsafe_store!(status_, Cint(-2))
                    return
                end
            end
        end
        if mode == 1 || mode == 2
            g = unsafe_wrap(Array, gobj_, nnobj)
            eval_grad(g, x)
            jnzval = unsafe_wrap(Array, gcon_, negcon)
            if negcon == nnz(J)
                eval_jac(jnzval, x)
            else
                jtmp = unsafe_wrap(Array, gcon_, nnz(J))
                eval_jac(jtmp, x)
            end
        end
        return
    end
    return usrfun
end

function make_usrfun_c(eval_obj::Function, eval_grad::Function,
                       eval_con::Function, eval_jac::Function, J)
    return make_usrfun_c(eval_obj, eval_grad, eval_con, eval_jac, J, Int32[])
end

function make_dummy_confun()
    function dummy_confun(_::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                          _::Ptr{Cdouble}, _::Ptr{Cdouble}, _::Ptr{Cdouble},
                          _::Ptr{Cint}, _::Ptr{UInt8}, _::Ptr{Cint},
                          _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        return
    end
    return dummy_confun
end

# Offsets into SNOPT's integer/real workspace arrays (SNOPT 7 manual, Appendix B).
# These are 1-based Julia indices into iw/rw.
const IW_MINOR_ITNS = 421  # cumulative minor iterations
const IW_MAJOR_ITNS = 422  # cumulative major iterations
const RW_RUN_TIME   = 462  # CPU time in seconds

workspace_value(ws_rw::Vector{Float64}, index::Int) =
    length(ws_rw) >= index ? ws_rw[index] : 0.0

# The MinGW Windows wrapper expects genuinely empty filenames here.
# Replacing them with "NUL" leaves the workspace partially initialized and the
# first solve can fail with bogus storage errors.
const SNOPT_DEVNULL = Sys.iswindows() ? "" : "/dev/null"

snopt_output_file(path::String) = isempty(path) ? SNOPT_DEVNULL : path

const SNOPT_STATUS = Dict(
    1  => :Solve_Succeeded,
    2  => :Feasible_Point_Found,
    3  => :Solved_To_Acceptable_Level,
    4  => :Solved_To_Acceptable_Level,
    5  => :Solved_To_Acceptable_Level,
    6  => :Solved_To_Acceptable_Level,
    11 => :Infeasible_Problem_Detected,
    12 => :Infeasible_Problem_Detected,
    13 => :Infeasible_Problem_Detected,
    14 => :Infeasible_Problem_Detected,
    15 => :Infeasible_Problem_Detected,
    16 => :Infeasible_Problem_Detected,
    21 => :Unbounded_Problem_Detected,
    22 => :Unbounded_Problem_Detected,
    31 => :Maximum_Iterations_Exceeded,
    32 => :Maximum_Iterations_Exceeded,
    33 => :Maximum_Iterations_Exceeded,
    34 => :Maximum_CpuTime_Exceeded,
    41 => :Numerical_Difficulties,
    42 => :Numerical_Difficulties,
    43 => :Numerical_Difficulties,
    44 => :Numerical_Difficulties,
    45 => :Numerical_Difficulties,
    51 => :User_Supplied_Function_Error,
    52 => :User_Supplied_Function_Error,
    56 => :User_Supplied_Function_Error,
    61 => :User_Supplied_Function_Undefined,
    62 => :User_Supplied_Function_Undefined,
    63 => :User_Supplied_Function_Undefined,
    71 => :User_Requested_Stop,
    72 => :User_Requested_Stop,
    73 => :User_Requested_Stop,
    74 => :User_Requested_Stop,
    81 => :Insufficient_Memory,
    82 => :Insufficient_Memory,
    83 => :Insufficient_Memory,
    84 => :Insufficient_Memory,
    91 => :Invalid_Problem_Definition,
    92 => :Invalid_Problem_Definition,
    141 => :Internal_Error,
    142 => :Internal_Error,
    999 => :Internal_Error)

"""
    initialize(printfile, summfile)

Allocate a workspace using a conservative default size suitable for small to
medium problems (up to ~100 variables and constraints). For larger problems,
use the explicit-size overload and compute workspace lengths with:

    leniw = 500 + 100*(n + nc)
    lenrw = 500 + 200*(n + nc)

where `n` is the number of design variables and `nc` the number of nonlinear
constraints. Problems with very dense Jacobians may need a larger `lenrw`.
"""
function initialize(printfile::String, summfile::String)
    initialize(printfile, summfile, 30500, 3000)
end

function initialize(printfile::String, summfile::String, leniw::Int, lenrw::Int)
    prob = SnoptWorkspace(leniw, lenrw)
    printpath = snopt_output_file(printfile)
    summpath = snopt_output_file(summfile)
    ccall((:f_sninitx, libsnopt7), Cvoid,
          (Cstring, Cint, Cstring, Cint,
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          printpath, Cint(ncodeunits(printpath)), summpath, Cint(ncodeunits(summpath)),
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return prob
end


function specs_status_message(info::Int)
    info == 101 && return "Specs file read successfully."
    info == 131 && return "No Specs file specified (iSpecs ≤ 0 or iSpecs > 99)."
    info == 132 && return "End-of-file while looking for Specs file (Begin not found)."
    info == 133 && return "End-of-file before finding End."
    info == 134 && return "Endrun found before any valid options."
    info > 134  && return "$(info - 134) error(s) while reading Specs file."
    return "Unknown specs inform code: $info."
end

function read_options(prob::SnoptWorkspace, specsfile::String)
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

function snopta!(prob::SnoptA; start::String = "Cold", name::String = "Julia")
    @assert prob.n == length(prob.x) == length(prob.xlow) == length(prob.xupp)
    @assert prob.n == length(prob.xstate) == length(prob.xmul)
    @assert prob.nf == length(prob.F) == length(prob.flow) == length(prob.fupp)
    @assert prob.nf == length(prob.Fstate) == length(prob.Fmul)
    @assert length(prob.iAfun) == length(prob.jAvar) == length(prob.A)
    @assert length(prob.iGfun) == length(prob.jGvar)

    prob.ws.iu = Int32[0]
    prob.ws.ru = [0.0]

    usrfun = prob.usrfun
    usr_callback = @cfunction($usrfun, Cvoid,
                              (Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                               Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                               Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                               Ptr{UInt8}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                               Ptr{Cdouble}, Ptr{Cint}))

    status = Int32[0]
    nS     = Int32[prob.nS]
    nInf   = Int32[0]
    sInf   = [0.0]
    miniw  = Int32[0]
    minrw  = Int32[0]

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
                 hs::Vector{Int32}, x::Vector{Float64})

    @assert n + m == length(x) == length(bl) == length(bu)
    @assert n + m == length(hs)

    prob.iu = Int32[0]
    prob.ru = [0.0]

    prob.x      = copy(x)
    prob.lambda = zeros(Float64, n + m)
    pi_         = zeros(Float64, m)

    obj_callback = @cfunction($objfun, Cvoid,
                             (Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                              Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint},
                              Ptr{UInt8}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                              Ptr{Cdouble}, Ptr{Cint}))

    con_callback = @cfunction($confun, Cvoid,
                             (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                              Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint},
                              Ptr{UInt8}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                              Ptr{Cdouble}, Ptr{Cint}))

    valJ = J.nzval
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

function snoptc!(prob::SnoptC; start::String = "Cold", name::String = "Julia")
    @assert prob.n + prob.m_eff == length(prob.x) == length(prob.bl) == length(prob.bu)
    @assert prob.n + prob.m_eff == length(prob.hs)

    prob.ws.iu = Int32[0]
    prob.ws.ru = [0.0]

    prob.ws.x      = copy(prob.x)
    prob.ws.lambda = zeros(Float64, prob.n + prob.m_eff)
    pi_            = zeros(Float64, prob.m_eff)

    usrfun = prob.usrfun
    usr_callback = @cfunction($usrfun, Cvoid,
                              (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                               Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                               Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                               Ptr{Cdouble}, Ptr{Cint}, Ptr{UInt8},
                               Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                               Ptr{Cdouble}, Ptr{Cint}))

    valJ = prob.J.nzval
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
          start_mode_code(start), name,
          prob.m_eff, prob.n, neJ, nnCon, prob.n, nnJac,
          0, 0.0,
          usr_callback,
          valJ, indJ, locJ,
          prob.bl, prob.bu, prob.hs, prob.ws.x, pi_, prob.ws.lambda,
          status, nS, nInf, sInf, obj_val,
          miniw, minrw,
          prob.ws.iu, prob.ws.leniu, prob.ws.ru, prob.ws.lenru,
          prob.ws.iw, prob.ws.leniw, prob.ws.rw, prob.ws.lenrw)

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

set_option!(prob::Union{SnoptA, SnoptB, SnoptC}, args...) = set_option!(prob.ws, args...)

function check_option_errors(errors, label)
    errors == 0 && return Int(errors)
    throw(ArgumentError("SNOPT rejected option $(label) with $(Int(errors)) error(s)"))
end

function set_option!(prob::SnoptWorkspace, optstring::String)
    isempty(strip(optstring)) &&
        throw(ArgumentError("SNOPT option string must not be empty or whitespace-only"))
    errors = Int32[0]
    ccall((:f_snset, libsnopt7), Cvoid,
          (Cstring, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          optstring, Cint(ncodeunits(optstring)), errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return check_option_errors(errors[1], repr(optstring))
end

function set_option!(prob::SnoptWorkspace, keyword::String, value::Int)
    isempty(strip(keyword)) &&
        throw(ArgumentError("SNOPT option keyword must not be empty or whitespace-only"))
    errors = Int32[0]
    ccall((:f_snseti, libsnopt7), Cvoid,
          (Cstring, Cint, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          keyword, Cint(ncodeunits(keyword)), value, errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return check_option_errors(errors[1], "$(repr(keyword)) => $(value)")
end

function set_option!(prob::SnoptWorkspace, keyword::String, value::Float64)
    isempty(strip(keyword)) &&
        throw(ArgumentError("SNOPT option keyword must not be empty or whitespace-only"))
    errors = Int32[0]
    ccall((:f_snsetr, libsnopt7), Cvoid,
          (Cstring, Cint, Cdouble, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          keyword, Cint(ncodeunits(keyword)), value, errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return check_option_errors(errors[1], "$(repr(keyword)) => $(value)")
end
