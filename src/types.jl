"""
    SnoptWorkspace

Store SNOPT's integer and real work arrays for one process-wide session.

Create a workspace with [`initialize`](@ref). Close it with `close` or manage it
with an `initialize do` block. Each work array must contain at least 500 values.

SNOPT permits one active workspace per process. Calling `initialize` closes the
previous active workspace. Creating several objects does not create parallel
solver sessions.

[`snopt`](@ref) manages this type automatically. Use it directly only with
[`SnoptA`](@ref), [`SnoptB`](@ref), or [`SnoptC`](@ref).
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
    nS::Int
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
                   0, 0.0, 0, 0, 0.0, 0)
        finalizer(free!, prob)
        prob
    end
end

"""
    AbstractSnoptProblem

Supertype for [`SnoptA`](@ref), [`SnoptB`](@ref), and [`SnoptC`](@ref).

Each problem owns dimensions, bounds, points, callbacks, and a
[`SnoptWorkspace`](@ref SNOPT.SnoptWorkspace).
"""
abstract type AbstractSnoptProblem end

"""
    SnoptA

Low-level problem for SNOPT's `snOptA` interface, in which a single user function
`usrfun(F, x)` returns the stacked objective/constraint row vector `F` and the
sparse derivative pattern is given separately as linear (`iAfun`/`jAvar`/`A`) and
nonlinear (`iGfun`/`jGvar`) triples. Solve in place with [`snopta!`](@ref). Build the
user function with [`make_usrfun_a`](@ref).

Constructed positionally, in field order:

    SnoptA(ws, nf, n, objadd, objrow,
           iAfun, jAvar, A, iGfun, jGvar,
           xlow, xupp, flow, fupp,
           x, xstate, xmul,
           F, Fstate, Fmul,
           status, nS, num_inf, sum_inf,
           usrfun)

with `ws` a [`SnoptWorkspace`](@ref); `nf` rows and `n` variables; the linear
part as `Int32` row/column indices `iAfun`/`jAvar` with values `A` (equal
lengths), and the nonlinear pattern as `Int32` `iGfun`/`jGvar` (equal lengths);
bounds `xlow`/`xupp` of length `n` and `flow`/`fupp` of length `nf`; the point
`x`, states `xstate::Vector{Int32}`, and multipliers `xmul` of length `n`; row
values `F`, states `Fstate::Vector{Int32}`, and multipliers `Fmul` of length
`nf`. The scalars `status`, `nS`, `num_inf`, `sum_inf` are outputs — pass
`0, 0, 0, 0.0` (a nonzero `nS` seeds a warm start's superbasics count). The
test suite's snOptA testsets contain complete worked constructions.
"""
mutable struct SnoptA{F} <: AbstractSnoptProblem
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

Constructed positionally, in field order:

    SnoptB(ws, n, nc, m_eff, nnobj,
           x, bl, bu, hs, J,
           obj_val, status, lambda,
           objfun, confun[, nS])

with `n` design variables, `nc` nonlinear constraints, `m_eff = max(nc, 1)`
effective rows (SNOPT requires at least one row, so unconstrained problems carry
a dummy), and `nnobj <= n` nonlinear objective variables (usually `n`). The
extended arrays `x`, `bl`, `bu`, and `hs::Vector{Int32}` have length
`n + m_eff` — design variables first, then one slack per row — and `J` is the
`m_eff × n` `SparseMatrixCSC{Float64,Int32}` Jacobian sparsity. `obj_val`,
`status`, and `lambda` are outputs — pass `0.0, 0, Float64[]`. The trailing `nS`
defaults to `0`; pass a previous solve's superbasics count when warm-starting.
"""
mutable struct SnoptB{F1, F2} <: AbstractSnoptProblem
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
    nS::Int                           # superbasics count, retained for warm starts
end

# Keeps the pre-0.3 positional form working; nS defaults to 0 (cold start).
SnoptB(ws, n, nc, m_eff, nnobj, x, bl, bu, hs, J, obj_val, status, lambda,
       objfun, confun) =
    SnoptB(ws, n, nc, m_eff, nnobj, x, bl, bu, hs, J, obj_val, status, lambda,
           objfun, confun, 0)

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

Constructed positionally, in field order (identical to [`SnoptB`](@ref) except a
single `usrfun` replaces the `objfun`/`confun` pair):

    SnoptC(ws, n, nc, m_eff, nnobj,
           x, bl, bu, hs, J,
           obj_val, status, lambda,
           usrfun[, nS])

with `n` design variables, `nc` nonlinear constraints, `m_eff = max(nc, 1)`
effective rows (SNOPT requires at least one row, so unconstrained problems carry
a dummy), and `nnobj <= n` nonlinear objective variables (usually `n`). The
extended arrays `x`, `bl`, `bu`, and `hs::Vector{Int32}` have length
`n + m_eff` — design variables first, then one slack per row — and `J` is the
`m_eff × n` `SparseMatrixCSC{Float64,Int32}` Jacobian sparsity. `obj_val`,
`status`, and `lambda` are outputs — pass `0.0, 0, Float64[]`. The trailing `nS`
defaults to `0`; pass a previous solve's superbasics count when warm-starting.
"""
mutable struct SnoptC{F} <: AbstractSnoptProblem
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
    nS::Int                           # superbasics count, retained for warm starts
end

# Keeps the pre-0.3 positional form working; nS defaults to 0 (cold start).
SnoptC(ws, n, nc, m_eff, nnobj, x, bl, bu, hs, J, obj_val, status, lambda,
       usrfun) =
    SnoptC(ws, n, nc, m_eff, nnobj, x, bl, bu, hs, J, obj_val, status, lambda,
           usrfun, 0)

"""
    SnoptMemory

Result from [`snmemb`](@ref), SNOPT's workspace estimator.

- `info` is the SNOPT result code. Codes 100 and 104 indicate success.
- `miniw` is the minimum integer-workspace length.
- `minrw` is the minimum real-workspace length.
"""
struct SnoptMemory
    info::Int
    miniw::Int
    minrw::Int
end

"""
    SnoptBasis

Basis data saved after a solve.

- `hs` stores one basis state for each extended variable.
- `nS` is the number of superbasic variables. They define SNOPT's reduced
  search subspace.
- `n` is the number of design variables.
- `m` is SNOPT's effective row count.

The invariant is `length(hs) == n + m`. An unconstrained high-level solve still
has one dummy row. Pass this object to [`snopt`](@ref) for a warm start.
"""
struct SnoptBasis
    hs::Vector{Int32}
    nS::Int
    n::Int
    m::Int
end

"""
    SnoptResult

Result from the high-level [`snopt`](@ref) function.

- `status`: SNOPT integer result code. See [`SNOPT_STATUS`](@ref).
- `status_symbol`: Symbolic interpretation of `status`.
- `objective`: Final objective value.
- `x`: Final design variables.
- `lambda`: Variable multipliers followed by constraint multipliers.
- `num_inf`: Number of remaining infeasibilities.
- `sum_inf`: Sum of remaining infeasibilities.
- `iterations`: Total minor iterations.
- `major_itns`: Total major iterations.
- `run_time`: SNOPT-reported solve time in seconds.
- `memory`: [`SnoptMemory`](@ref) used to size the workspace.
- `basis`: [`SnoptBasis`](@ref) for a later warm start.

The basis is meaningful only after SNOPT starts. An evaluation callback can
stop before SNOPT creates valid basis data.
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
    basis::SnoptBasis
end

"""
    SnoptMajorLog

State captured for one major-iteration `snlog` event.

- `iteration`, `major_iter`, and `minor_iter` are SNOPT counters.
- `objective` includes any objective offset and linear row.
- `merit` combines the objective with a feasibility penalty.
- `f_objective` and `f_merit` contain the nonlinear components.
- `primal_infeasibility` and `dual_infeasibility` measure convergence.
- `max_violation` and `relative_violation` measure feasibility violations.
- `step`, `penalty_norm`, and `condition_hessian` describe the algorithm state.
- `x` contains design variables followed by slacks.
- `fcon`, `fx`, and `ycon` contain row values and multipliers.
- `hs` contains basis states for the extended variables.

[`snopt`](@ref) delivers this object to `snlog`. Return `false` to request a
stop. Return any other value to continue.
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

"""
    SnoptStopEvent

State captured for one major-iteration `snstop` event.

This object contains every [`SnoptMajorLog`](@ref) field. It also contains:

- `m`, `max_superbasics`, and `negcon`: Extended problem dimensions.
- `gobj`: Current objective gradient.
- `gcon`: Stored constraint Jacobian values.
- `fx`: Nonlinear constraint row values.
- `pi`: Row multipliers.
- `rc`: Reduced costs for extended variables.
- `rg`: Reduced gradient for superbasic variables.
- `bl` and `bu`: Scaled bounds used by SNOPT.

[`snopt`](@ref) delivers this object to `snstop`. Return `false` to request a
stop. The solve then returns `:User_Requested_Stop`.
"""
struct SnoptStopEvent
    iteration::Int
    major_iter::Int
    minor_iter::Int
    n_superbasics::Int
    max_superbasics::Int
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
    m::Int
    n::Int
    nb::Int
    nncon::Int
    nnobj::Int
    negcon::Int
    kt_conditions::NTuple{2, Bool}
    x::Vector{Float64}
    bl::Vector{Float64}
    bu::Vector{Float64}
    fcon::Vector{Float64}
    fx::Vector{Float64}
    gcon::Vector{Float64}
    gobj::Vector{Float64}
    ycon::Vector{Float64}
    pi::Vector{Float64}
    rc::Vector{Float64}
    rg::Vector{Float64}
    hs::Vector{Int32}
end
