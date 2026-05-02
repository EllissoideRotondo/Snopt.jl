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

abstract type AbstractSnoptProblem end

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

mutable struct SnoptB{F1<:Function, F2<:Function} <: AbstractSnoptProblem
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

mutable struct SnoptC{F<:Function} <: AbstractSnoptProblem
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
