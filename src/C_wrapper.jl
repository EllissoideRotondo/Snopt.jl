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

struct SnoptMemory
    info::Int
    miniw::Int
    minrw::Int
end

struct SnoptResult
    status::Int
    status_symbol::Symbol
    objective::Float64
    x::Vector{Float64}
    lambda::Vector{Float64}
    num_inf::Int
    sum_inf::Float64
    iterations::Int
    major_itns::Int
    run_time::Float64
    memory::SnoptMemory
end

struct SnoptMajorLog
    iteration::Int
    major_iter::Int
    minor_iter::Int
    n_superbasics::Int
    n_swaps::Int
    objective::Float64
    merit::Float64
    penalty_norm::Float64
    step::Float64
    primal_infeasibility::Float64
    dual_infeasibility::Float64
    max_violation::Float64
    relative_violation::Float64
    condition_hessian::Float64
    objective_scale::Float64
    objective_add::Float64
    f_objective::Float64
    f_merit::Float64
    minimize::Int
    n::Int
    nb::Int
    nncon::Int
    nnobj::Int
    kt_conditions::NTuple{2, Bool}
    x::Vector{Float64}
    fcon::Vector{Float64}
    fx::Vector{Float64}
    ycon::Vector{Float64}
    hs::Vector{Int32}
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

function snopt!(prob::SnoptB; start::String = "Cold", name::String = "Julia",
                snlog=nothing)
    return snoptb!(prob; start, name, snlog)
end

function snoptb!(prob::SnoptB; start::String = "Cold", name::String = "Julia",
                 snlog=nothing)
    nc    = prob.nc
    nnCon = nc
    nnJac = nc > 0 ? prob.n : 0
    inform = snoptb!(prob.ws, start, name,
                     prob.m_eff, prob.n, nnCon, prob.n, nnJac,
                     0.0, 0,
                     prob.confun, prob.objfun,
                     prob.J, prob.bl, prob.bu, prob.hs, prob.x;
                     snlog)
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

snopt_no_progress(event) = true

workspace_value(ws_iw::Vector{Int32}, index::Int) =
    length(ws_iw) >= index ? max(Int(ws_iw[index]), 0) : 0

function progress_iters(ws_iw::Vector{Int32})
    return (workspace_value(ws_iw, IW_MAJOR_ITNS),
            workspace_value(ws_iw, IW_MINOR_ITNS))
end

call_progress(::Nothing, event) = true
call_progress(callback, event) = callback(event) !== false

legacy_progress_callback(log_fn::Function) =
    event -> event.kind === :objective ? log_fn(event.major_iter, event.x, event.f) : true

copy_cdouble_vector(ptr::Ptr{Cdouble}, len::Integer) =
    len <= 0 ? Float64[] : copy(unsafe_wrap(Array, ptr, Int(len)))

copy_cint32_vector(ptr::Ptr{Cint}, len::Integer) =
    len <= 0 ? Int32[] : copy(unsafe_wrap(Array, ptr, Int(len)))

"""
    make_snlog(callback)

Create a Julia callback compatible with SNOPT's `snLog` hook. `callback` is
called with a [`SnoptMajorLog`](@ref) after each major-iteration log event.
Returning `false` requests termination from SNOPT.
"""
function make_snlog(callback)
    function snlog(iAbort_::Ptr{Cint}, KTcond_::Ptr{Cint},
                   MjrPrt_::Ptr{Cint}, minimz_::Ptr{Cint},
                   n_::Ptr{Cint}, nb_::Ptr{Cint}, nnCon0_::Ptr{Cint},
                   nnObj_::Ptr{Cint}, nS_::Ptr{Cint},
                   itn_::Ptr{Cint}, nMajor_::Ptr{Cint},
                   nMinor_::Ptr{Cint}, nSwap_::Ptr{Cint},
                   condHz_::Ptr{Cdouble}, iObj_::Ptr{Cint},
                   sclObj_::Ptr{Cdouble}, ObjAdd_::Ptr{Cdouble},
                   fObj_::Ptr{Cdouble}, fMrt_::Ptr{Cdouble},
                   PenNrm_::Ptr{Cdouble}, step_::Ptr{Cdouble},
                   prInf_::Ptr{Cdouble}, duInf_::Ptr{Cdouble},
                   vimax_::Ptr{Cdouble}, virel_::Ptr{Cdouble},
                   hs_::Ptr{Cint}, ne_::Ptr{Cint},
                   nlocJ_::Ptr{Cint}, locJ_::Ptr{Cint},
                   indJ_::Ptr{Cint}, Jcol_::Ptr{Cdouble},
                   Ascale_::Ptr{Cdouble}, bl_::Ptr{Cdouble},
                   bu_::Ptr{Cdouble}, Fx_::Ptr{Cdouble},
                   fCon_::Ptr{Cdouble}, yCon_::Ptr{Cdouble},
                   x_::Ptr{Cdouble}, cu_::Ptr{UInt8},
                   lencu_::Ptr{Cint}, iu_::Ptr{Cint},
                   leniu_::Ptr{Cint}, ru_::Ptr{Cdouble},
                   lenru_::Ptr{Cint}, cw_::Ptr{UInt8},
                   lencw_::Ptr{Cint}, iw_::Ptr{Cint},
                   leniw_::Ptr{Cint}, rw_::Ptr{Cdouble},
                   lenrw_::Ptr{Cint})::Cvoid
        n = Int(unsafe_load(n_))
        nb = Int(unsafe_load(nb_))
        nncon = Int(unsafe_load(nnCon0_))
        ktcond = unsafe_wrap(Array, KTcond_, 2)
        penalty = unsafe_wrap(Array, PenNrm_, 4)
        objective_add = unsafe_load(ObjAdd_)
        f_objective = unsafe_load(fObj_)
        f_merit = unsafe_load(fMrt_)
        objective_scale = unsafe_load(sclObj_)
        iobj = Int(unsafe_load(iObj_))
        x_values = copy_cdouble_vector(x_, nb)
        linear_objective = (iobj > 0 && n + iobj <= length(x_values)) ?
            objective_scale * x_values[n + iobj] : 0.0
        event = SnoptMajorLog(
            Int(unsafe_load(itn_)),
            Int(unsafe_load(nMajor_)),
            Int(unsafe_load(nMinor_)),
            Int(unsafe_load(nS_)),
            Int(unsafe_load(nSwap_)),
            objective_add + linear_objective + f_objective,
            objective_add + linear_objective + f_merit,
            Float64(penalty[3]),
            unsafe_load(step_),
            unsafe_load(prInf_),
            unsafe_load(duInf_),
            unsafe_load(vimax_),
            unsafe_load(virel_),
            unsafe_load(condHz_),
            objective_scale,
            objective_add,
            f_objective,
            f_merit,
            Int(unsafe_load(minimz_)),
            n,
            nb,
            nncon,
            Int(unsafe_load(nnObj_)),
            (ktcond[1] != 0, ktcond[2] != 0),
            x_values,
            copy_cdouble_vector(fCon_, nncon),
            copy_cdouble_vector(Fx_, nncon),
            copy_cdouble_vector(yCon_, nncon),
            copy_cint32_vector(hs_, nb)
        )
        unsafe_store!(iAbort_, call_progress(callback, event) ? Cint(0) : Cint(1))
        return
    end
    return snlog
end

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

const SNOPT_MEMORY_WORKSPACE = 500

memory_estimate_success(info::Int) = info == 100 || info == 104

function check_memory_estimate(memory::SnoptMemory)
    memory_estimate_success(memory.info) && return memory
    throw(ErrorException("SNOPT memory estimator failed with info code $(memory.info)"))
end

function option_keyword(key::AbstractString)
    keyword = String(strip(String(key)))
    isempty(keyword) &&
        throw(ArgumentError("SNOPT option keyword must not be empty or whitespace-only"))
    return keyword
end

function option_keyword(key::Symbol)
    return option_keyword(replace(String(key), "_" => " "))
end

function option_keyword(key)
    throw(ArgumentError("SNOPT option keyword must be a String or Symbol, got $(typeof(key))"))
end

function apply_option!(ws::SnoptWorkspace, option::Pair)
    key = option_keyword(first(option))
    value = last(option)
    if value isa Bool
        throw(ArgumentError("SNOPT option $(repr(key)) does not accept Bool values"))
    elseif value isa Integer
        set_option!(ws, key, Int(value))
    elseif value isa AbstractFloat
        float_value = Float64(value)
        isfinite(float_value) ||
            throw(ArgumentError("SNOPT option $(repr(key)) requires a finite floating-point value"))
        set_option!(ws, key, float_value)
    elseif value isa AbstractString
        string_value = String(strip(String(value)))
        isempty(string_value) &&
            throw(ArgumentError("SNOPT option $(repr(key)) string value must not be empty or whitespace-only"))
        set_option!(ws, string(key, " ", string_value))
    elseif value isa Symbol
        set_option!(ws, string(key, " ", option_keyword(value)))
    else
        throw(ArgumentError("SNOPT option $(repr(key)) value must be an integer, finite float, string, or symbol; got $(typeof(value))"))
    end
    return ws
end

function apply_options!(ws::SnoptWorkspace, options::Nothing)
    return ws
end

function apply_options!(ws::SnoptWorkspace, options::AbstractVector{<:Pair})
    for option in options
        apply_option!(ws, option)
    end
    return ws
end

function apply_options!(ws::SnoptWorkspace, options)
    throw(ArgumentError("SNOPT options must be a Vector of Pairs, got $(typeof(options))"))
end

"""
    snmemb(ws, m, n, neJ, negCon, nnCon, nnJac, nnObj) -> SnoptMemory

Estimate the SNOPTB/SNOPTC integer and real workspace lengths using SNOPT's
`snMemB` routine. The shared library exposes this through the C ABI wrapper
`f_snmem`, which calls `snMem`/`snMemB` internally.

Call `initialize` before using this workspace method, and apply any options
that may affect memory before calling `snmemb`.
"""
function snmemb(ws::SnoptWorkspace, m::Integer, n::Integer, neJ::Integer,
                negCon::Integer, nnCon::Integer, nnJac::Integer,
                nnObj::Integer)
    info  = Int32[0]
    miniw = Int32[0]
    minrw = Int32[0]
    ccall((:f_snmem, libsnopt7), Cvoid,
          (Ptr{Cint}, Cint, Cint, Cint, Cint, Cint, Cint, Cint,
           Ptr{Cint}, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          info,
          Int(m), Int(n), Int(neJ), Int(negCon), Int(nnCon),
          Int(nnObj), Int(nnJac),
          miniw, minrw,
          ws.iw, ws.leniw, ws.rw, ws.lenrw)
    memory = SnoptMemory(Int(info[1]), Int(miniw[1]), Int(minrw[1]))
    ws.status = memory.info
    return memory
end

"""
    snmemb(m, n, neJ, negCon, nnCon, nnJac, nnObj; options=nothing,
           printfile="", summfile="")

Initialize a temporary bootstrap workspace, apply any SNOPT options, and return
the SNOPTB/SNOPTC memory estimate.
"""
function snmemb(m::Integer, n::Integer, neJ::Integer, negCon::Integer,
                nnCon::Integer, nnJac::Integer, nnObj::Integer;
                options=nothing, printfile::String = "", summfile::String = "")
    ws = initialize(printfile, summfile, SNOPT_MEMORY_WORKSPACE,
                    SNOPT_MEMORY_WORKSPACE)
    try
        apply_options!(ws, options)
        return snmemb(ws, m, n, neJ, negCon, nnCon, nnJac, nnObj)
    finally
        free!(ws)
    end
end

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
                 hs::Vector{Int32}, x::Vector{Float64};
                 snlog=nothing)

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

    if snlog === nothing
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
    else
        snlog_fn = make_snlog(snlog)
        snlog_callback = @cfunction($snlog_fn, Cvoid,
            (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
             Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
             Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
             Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{UInt8}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
             Ptr{Cdouble}, Ptr{Cint}, Ptr{UInt8}, Ptr{Cint},
             Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}))
        null_callback = Ptr{Cvoid}(C_NULL)
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
