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
