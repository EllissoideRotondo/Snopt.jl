module Snopt

using Libdl
using SparseArrays: SparseMatrixCSC, nnz

# Mutable typed global so __init__ can set the resolved path at runtime.
# All ccall sites reference this variable; Julia re-evaluates it on each call,
# which is negligible overhead compared to any SNOPT solve.
global libsnopt7::String = ""

function loadable_library_path(libpath::AbstractString)
    isempty(libpath) && return ""
    d = Libdl.dlopen_e(libpath)
    d == C_NULL && return ""
    Libdl.dlclose(d)
    return String(libpath)
end

function find_snopt_lib()
    libname = string("lib", "snopt7", ".", Libdl.dlext)
    snoptdir = get(ENV, "SNOPTDIR", "")
    if !isempty(snoptdir)
        return loadable_library_path(joinpath(snoptdir, libname))
    end

    paths_to_try = String[]
    if Sys.iswindows()
        if haskey(ENV, "PATH")
            append!(paths_to_try, split(ENV["PATH"], ';'))
        end
    else
        if haskey(ENV, "LD_LIBRARY_PATH")
            append!(paths_to_try, split(ENV["LD_LIBRARY_PATH"], ':'))
        end
        if Sys.isapple() && haskey(ENV, "DYLD_LIBRARY_PATH")
            append!(paths_to_try, split(ENV["DYLD_LIBRARY_PATH"], ':'))
        end
    end

    for path in paths_to_try
        libpath = loadable_library_path(joinpath(path, libname))
        if !isempty(libpath)
            return libpath
        end
    end
    return ""
end

function __init__()
    global libsnopt7 = find_snopt_lib()
    if isempty(libsnopt7)
        @warn """
              Snopt.jl: SNOPT library not found. has_snopt() returns false.
              Set SNOPTDIR to the directory containing libsnopt7, or add it to the platform library path:
                  export SNOPTDIR=/path/to/snopt/lib
                  export LD_LIBRARY_PATH=/path/to/snopt/lib:\$LD_LIBRARY_PATH
                  export DYLD_LIBRARY_PATH=/path/to/snopt/lib:\$DYLD_LIBRARY_PATH  # macOS
              """
    else
        # Preload OpenMP companion library if it lives alongside libsnopt7.
        libiomp5 = replace(libsnopt7, "libsnopt7" => "libiomp5")
        isfile(libiomp5) && Libdl.dlopen(libiomp5)
    end
end

has_snopt() = !isempty(libsnopt7)

include("types.jl")
include("workspace.jl")
include("solve_wrappers.jl")
include("callbacks.jl")
include("options.jl")
include("memory.jl")
include("high_level.jl")
include("raw_api.jl")

export SnoptA
export SnoptB
export SnoptC
export AbstractSnoptProblem
export SnoptMajorLog
export SnoptMemory
export SnoptProblem
export SnoptResult
export make_snlog
export snopt
export snopt!
export snopta!
export snoptb!
export snoptc!
export snmemb
export set_option!
export read_options
export initialize
export make_usrfun_a
export make_usrfun_c
export make_objfun
export make_confun
export make_dummy_confun
export snopt_no_progress
export SNOPT_STATUS
export specs_status_message

end
