"""
    SnoptWorkspace

SNOPT's integer (`iw`) and real (`rw`) working storage plus the bookkeeping needed to
initialize and finalize one SNOPT session. Created by [`initialize`](@ref), which calls
SNOPT's `sninit`, and released by `close`/`free!`, which calls SNOPT's `snend`. SNOPT
requires each work array to hold at least 500 elements; smaller requests are rejected.
A workspace is normally managed for you by [`snopt`](@ref); use it directly only when
driving the low-level [`SnoptA`](@ref)/[`SnoptB`](@ref)/[`SnoptC`](@ref) entry points.

SNOPT solves are process-serial: run one solve at a time per Julia process, and
use multiple Julia processes for parallel solves. Creating multiple
`SnoptWorkspace` objects does not make independent solver sessions. Calling
[`initialize`](@ref) again closes any previous active workspace before creating the
new one.
"""
mutable struct SnoptWorkspace
    status::Int
    finalized::Bool
    init_id::Int       # ID of the f_sninitx call; 0 = not yet initialized via f_sninitx
    leniw::Int
    lenrw::Int
    tempfiles::Vector{String}
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
        # SNOPT's sninit writes a fixed-size header into iw/rw and requires at
        # least 500 elements in each. Smaller arrays let f_sninitx write out of
        # bounds, which silently corrupts the heap and later segfaults, so reject
        # them before any allocation reaches the Fortran side.
        leniw >= 500 || throw(ArgumentError("leniw must be >= 500 (SNOPT work-array minimum), got $leniw"))
        lenrw >= 500 || throw(ArgumentError("lenrw must be >= 500 (SNOPT work-array minimum), got $lenrw"))
        prob = new(0, false, 0, leniw, lenrw, String[],
                   zeros(Int32, leniw), zeros(Float64, lenrw),
                   0, 0,
                   Int32[0], [0.0],
                   Float64[], Float64[], 0.0,
                   0, 0.0, 0, 0, 0.0)
        finalizer(free!, prob)
        prob
    end
end

"""
    AbstractSnoptProblem

Supertype for the low-level SNOPT problem objects [`SnoptA`](@ref), [`SnoptB`](@ref),
and [`SnoptC`](@ref). Each wraps a [`SnoptWorkspace`](@ref SNOPT.SnoptWorkspace) together
with the bound, point, and callback data for one of SNOPT's three Fortran entry points.
"""
abstract type AbstractSnoptProblem end

"""
    SnoptA

Low-level problem for SNOPT's `snOptA` interface, in which a single user function
`usrfun(F, x)` returns the stacked objective/constraint row vector `F` and the
sparse derivative pattern is given separately as linear (`iAfun`/`jAvar`/`A`) and
nonlinear (`iGfun`/`jGvar`) triples. Solve in place with [`snopta!`](@ref). Build the
user function with [`make_usrfun_a`](@ref).
"""
mutable struct SnoptA{F<:Function} <: AbstractSnoptProblem
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

"""
    SnoptB

Low-level problem for SNOPT's `snOptB` interface, with separate objective
(`objfun`) and constraint (`confun`) callbacks. This is the type the high-level
[`snopt`](@ref) entry point builds and solves, and it is also exported under the
alias [`SnoptProblem`](@ref). Variables and slacks are stored in the extended
vectors `x`, `bl`, `bu`, `hs` of length `n + m_eff`, and the constraint Jacobian
sparsity is held in `J`. Solve in place with [`snoptb!`](@ref) (or the alias
[`snopt!`](@ref)). Construct the callbacks with [`make_objfun`](@ref) and
[`make_confun`](@ref).
"""
mutable struct SnoptB{F1<:Function, F2<:Function} <: AbstractSnoptProblem
    ws::SnoptWorkspace
    n::Int                            # num design variables
    nc::Int                           # num nonlinear constraints
    m_eff::Int                        # effective m passed to Fortran (>= 1; nc when nc>0, else 1)
    nnobj::Int                        # num nonlinear objective variables (<= n)
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

"""
    SnoptProblem

Alias for [`SnoptB`](@ref), the default low-level problem type built and solved by
[`snopt`](@ref).
"""
const SnoptProblem = SnoptB

"""
    SnoptC

Low-level problem for SNOPT's `snOptC` interface, in which a single user function
evaluates the objective, objective gradient, constraints, and constraint Jacobian
together (the combined analogue of [`SnoptB`](@ref)'s split callbacks). Solve in
place with [`snoptc!`](@ref). Build the user function with [`make_usrfun_c`](@ref).
"""
mutable struct SnoptC{F<:Function} <: AbstractSnoptProblem
    ws::SnoptWorkspace
    n::Int                            # num design variables
    nc::Int                           # num nonlinear constraints
    m_eff::Int                        # effective m passed to Fortran
    nnobj::Int                        # num nonlinear objective variables (<= n)
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

"""
    SnoptMemory

Result of SNOPT's workspace-memory estimator (see [`snmemb`](@ref)). `info` is the
SNOPT inform code from the estimate (100 or 104 on success), and `miniw`/`minrw` are
the minimum integer- and real-workspace lengths SNOPT needs to solve the problem.
"""
struct SnoptMemory
    info::Int
    miniw::Int
    minrw::Int
end

"""
    SnoptResult

Outcome of a high-level [`snopt`](@ref) solve. Fields:

  * `status`: SNOPT inform code (see [`SNOPT_STATUS`](@ref)).
  * `status_symbol`: symbolic interpretation of `status`, e.g. `:Solve_Succeeded`.
  * `objective`: final objective value.
  * `x`: final values of the `n` design variables.
  * `lambda`: Lagrange multipliers for the variables followed by the nonlinear
    constraints, in the same order as `[x; c(x)]` in SNOPT's extended problem.
    Active lower-bound multipliers are positive in SNOPT's convention.
  * `num_inf`, `sum_inf`: number and sum of constraint infeasibilities.
  * `iterations`, `major_itns`: total minor and major iteration counts.
  * `run_time`: SNOPT-reported solve time in seconds.
  * `memory`: the [`SnoptMemory`](@ref) estimate used to size the workspace.
"""
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

"""
    SnoptMajorLog

Snapshot of SNOPT's state at one major-iteration log event, delivered to the
`snlog` callback of [`snopt`](@ref) (and built internally by [`make_snlog`](@ref)).
It mirrors the quantities SNOPT prints on its major-iteration log line —
iteration counters, objective and merit-function values, primal/dual
infeasibilities, step length, penalty and Hessian-condition estimates — and also
exposes the current point `x`, constraint values `fcon`, and multipliers `ycon`.

Important fields:

  * `iteration`, `major_iter`, `minor_iter`: SNOPT iteration counters.
  * `objective`, `merit`: objective and merit-function values including any
    objective offset and linear objective row.
  * `f_objective`, `f_merit`: nonlinear objective and merit-function components
    reported by SNOPT before adding the offset/linear objective row.
  * `primal_infeasibility`, `dual_infeasibility`, `max_violation`,
    `relative_violation`: feasibility and optimality diagnostics from SNOPT's log.
  * `step`, `penalty_norm`, `condition_hessian`: step and algorithm diagnostics.
  * `x`: the current extended SNOPT point, including design variables and slacks.
  * `fcon`, `fx`, `ycon`: nonlinear constraint values, row values, and multipliers.
  * `hs`: SNOPT basis-state array for the extended variables.

Return `false` from the callback to request early termination; any other return
value lets SNOPT continue.
"""
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
