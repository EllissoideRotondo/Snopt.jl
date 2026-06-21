# Tracks the init_id of the workspace most recently passed to f_sninitx.
# On Linux, SNOPT's Fortran common blocks are global and are associated with the
# most recently initialized workspace. Calling f_snend on a superseded workspace
# corrupts the active workspace's state, causing subsequent solves to fail with
# status 82 (Insufficient_Memory) or 91 (Invalid_Problem_Definition).
# Using atomics here is intentional: ReentrantLock must not be acquired inside
# a Julia GC finalizer (risk of deadlock), but atomic CAS is safe.
const _SNOPT_ACTIVE_ID  = Threads.Atomic{Int}(0)
const _SNOPT_ID_COUNTER = Threads.Atomic{Int}(0)
const _SNOPT_ACTIVE_WORKSPACE = Ref{Any}(nothing)

function reset_snopt_defaults!(prob::SnoptWorkspace)
    optstring = "Defaults"
    errors = Int32[0]
    ccall((:f_snset, libsnopt7), Cvoid,
          (Cstring, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          optstring, Cint(ncodeunits(optstring)), errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    errors[1] == 0 ||
        error("SNOPT rejected Defaults option during workspace initialization")
    return prob
end

function free!(prob::SnoptWorkspace)
    prob.finalized && return nothing
    prob.finalized = true
    if !isempty(libsnopt7)
        # Only call f_snend for the workspace that last called f_sninitx. Any
        # older workspace that gets GC'd after being superseded skips the
        # Fortran call.
        id = prob.init_id
        should_end = id == 0 || Threads.atomic_cas!(_SNOPT_ACTIVE_ID, id, 0) == id
        if should_end
            try
                ccall((:f_snend, libsnopt7),
                      Cvoid, (Ptr{Cint}, Cint, Ptr{Float64}, Cint),
                      prob.iw, prob.leniw, prob.rw, prob.lenrw)
            catch
                # Finalizers may run during shutdown when the shared library
                # is gone.
            end
        end
    end
    for path in prob.tempfiles
        try
            isfile(path) && rm(path; force=true)
        catch
            # Temporary output cleanup should never make finalization fail.
        end
    end
    empty!(prob.tempfiles)
    _SNOPT_ACTIVE_WORKSPACE[] === prob && (_SNOPT_ACTIVE_WORKSPACE[] = nothing)
    return nothing
end

function close_active_workspace!()
    active = _SNOPT_ACTIVE_WORKSPACE[]
    if active isa SnoptWorkspace && !active.finalized
        free!(active)
    else
        _SNOPT_ACTIVE_WORKSPACE[] = nothing
    end
    return nothing
end

Base.close(prob::SnoptWorkspace) = free!(prob)
Base.isopen(prob::SnoptWorkspace) = !prob.finalized

function require_open_workspace(prob::SnoptWorkspace, action::AbstractString)
    isopen(prob) ||
        throw(ArgumentError("$(action) requires an open SNOPT workspace"))
    return prob
end

const IW_MINOR_ITNS = 421  # iw(421): cumulative minor iterations - SNOPT 7.7 iw layout

const IW_MAJOR_ITNS = 422  # iw(422): cumulative major iterations - SNOPT 7.7 iw layout

const RW_RUN_TIME   = 462  # rw(462): CPU run time in seconds    - SNOPT 7.7 rw layout

workspace_value(ws_rw::Vector{Float64}, index::Int) =
    length(ws_rw) >= index ? max(ws_rw[index], 0.0) : 0.0
# The MinGW Windows wrapper expects genuinely empty filenames here.
# Replacing them with "NUL" leaves the workspace partially initialized and the
# first solve can fail with bogus storage errors.

const SNOPT_DEVNULL = Sys.iswindows() ? "" : "/dev/null"

snopt_output_file(path::String) = isempty(path) ? SNOPT_DEVNULL : path

function snopt_output_files(printfile::String, summfile::String)
    printpath = snopt_output_file(printfile)
    summpath = snopt_output_file(summfile)
    tempfiles = String[]
    if isempty(printfile) && isempty(summfile)
        # The Linux library can become stateful in surprising ways when both
        # SNOPT output channels are opened on the null device across mixed solves.
        # Keep the print channel suppressed and give the summary channel a real,
        # throwaway file.
        summpath = tempname()
        push!(tempfiles, summpath)
    end
    return printpath, summpath, tempfiles
end

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
    initialize(f, printfile, summfile[, leniw, lenrw])

Allocate a workspace using a conservative default size suitable for small to
medium problems (up to ~100 variables and constraints). For larger problems,
use the explicit-size overload and compute workspace lengths with:
    leniw = 500 + 100*(n + nc)
    lenrw = 500 + 200*(n + nc)
where `n` is the number of design variables and `nc` the number of nonlinear
constraints. Problems with very dense Jacobians may need a larger `lenrw`.

The function form supports Julia's do-block cleanup pattern:

    initialize("", "") do ws
        # build and solve a low-level SnoptA/SnoptB/SnoptC problem
    end

The workspace is closed with `close(ws)` when the block exits, including if the
block throws.

"""

function initialize(printfile::String, summfile::String)
    initialize(printfile, summfile, 30500, 60000)
end

function initialize(printfile::String, summfile::String, leniw::Int, lenrw::Int)
    has_snopt() || error(
        "SNOPT library not loaded. Set SNOPTDIR (or DYLD_LIBRARY_PATH on macOS) " *
        "to the directory containing libsnopt7 and restart Julia, " *
        "or call SNOPT.find_snopt_lib() to diagnose.")
    close_active_workspace!()
    prob = SnoptWorkspace(leniw, lenrw)
    printpath, summpath, tempfiles = snopt_output_files(printfile, summfile)
    append!(prob.tempfiles, tempfiles)
    new_id = Threads.atomic_add!(_SNOPT_ID_COUNTER, 1) + 1
    prob.init_id = new_id
    Threads.atomic_xchg!(_SNOPT_ACTIVE_ID, new_id)
    ccall((:f_sninitx, libsnopt7), Cvoid,
          (Cstring, Cint, Cstring, Cint,
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          printpath, Cint(ncodeunits(printpath)), summpath, Cint(ncodeunits(summpath)),
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    reset_snopt_defaults!(prob)
    _SNOPT_ACTIVE_WORKSPACE[] = prob
    return prob
end

function initialize(f::Function, printfile::String, summfile::String)
    ws = initialize(printfile, summfile)
    try
        return f(ws)
    finally
        close(ws)
    end
end

function initialize(f::Function, printfile::String, summfile::String,
                    leniw::Int, lenrw::Int)
    ws = initialize(printfile, summfile, leniw, lenrw)
    try
        return f(ws)
    finally
        close(ws)
    end
end
