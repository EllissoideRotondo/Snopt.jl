const SNOPT_MEMORY_WORKSPACE = 1000

memory_estimate_success(info::Int) = info == 100 || info == 104

function check_memory_estimate(memory::SnoptMemory)
    memory_estimate_success(memory.info) && return memory
    throw(ErrorException("SNOPT memory estimator failed with info code $(memory.info)"))
end

function validate_snmemb_dimensions(m::Integer, n::Integer, neJ::Integer,
                                    negCon::Integer, nnCon::Integer,
                                    nnObj::Integer, nnJac::Integer)
    m > 0 || throw(ArgumentError("m must be positive, got $m"))
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    neJ >= 0 || throw(ArgumentError("neJ must be nonnegative, got $neJ"))
    negCon >= 0 || throw(ArgumentError("negCon must be nonnegative, got $negCon"))
    nnCon >= 0 || throw(ArgumentError("nnCon must be nonnegative, got $nnCon"))
    nnObj >= 0 || throw(ArgumentError("nnObj must be nonnegative, got $nnObj"))
    nnJac >= 0 || throw(ArgumentError("nnJac must be nonnegative, got $nnJac"))
    nnObj <= n || throw(ArgumentError("nnObj must be <= n, got nnObj=$nnObj and n=$n"))
    nnJac <= n || throw(ArgumentError("nnJac must be <= n, got nnJac=$nnJac and n=$n"))
    nnCon <= m || throw(ArgumentError("nnCon must be <= m, got nnCon=$nnCon and m=$m"))
    negCon <= neJ || throw(ArgumentError("negCon must be <= neJ, got negCon=$negCon and neJ=$neJ"))
    return nothing
end

function snmemb(ws::SnoptWorkspace, m::Integer, n::Integer, neJ::Integer,
                negCon::Integer, nnCon::Integer, nnObj::Integer,
                nnJac::Integer)
    require_open_workspace(ws, "snmemb")
    validate_snmemb_dimensions(m, n, neJ, negCon, nnCon, nnObj, nnJac)
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
    snmemb(m, n, neJ, negCon, nnCon, nnObj, nnJac; options=nothing,
           printfile="", summfile="")

Initialize a temporary bootstrap workspace, apply any SNOPT options, and return
the `snOptB`/`snOptC` memory estimate as a [`SnoptMemory`](@ref).

The dimensions match SNOPT's `snMemB` arguments: `m` total constraint rows, `n`
variables, `neJ` Jacobian nonzeros, `negCon` nonlinear constraint Jacobian
nonzeros, `nnCon` nonlinear constraints, `nnObj` nonlinear objective variables,
and `nnJac` variables that appear nonlinearly in the constraint Jacobian.

"""
function snmemb(m::Integer, n::Integer, neJ::Integer, negCon::Integer,
                nnCon::Integer, nnObj::Integer, nnJac::Integer;
                options=nothing, printfile::String = "", summfile::String = "")
    ws = initialize(printfile, summfile, SNOPT_MEMORY_WORKSPACE,
                    SNOPT_MEMORY_WORKSPACE)
    try
        apply_options!(ws, options)
        return snmemb(ws, m, n, neJ, negCon, nnCon, nnObj, nnJac)
    finally
        free!(ws)
    end
end
