const SNOPT_MEMORY_WORKSPACE = 1000

memory_estimate_success(info::Int) = info == 100 || info == 104

function check_memory_estimate(memory::SnoptMemory)
    memory_estimate_success(memory.info) && return memory
    throw(ErrorException("SNOPT memory estimator failed with info code $(memory.info)"))
end

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
