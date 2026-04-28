module Snopt

using Libdl
using SparseArrays: SparseMatrixCSC, nnz

# Mutable typed global so __init__ can set the resolved path at runtime.
# All ccall sites reference this variable; Julia re-evaluates it on each call,
# which is negligible overhead compared to any SNOPT solve.
global libsnopt7::String = ""

function path_sep()
    return Sys.iswindows() ? ';' : ':'
end

function maybe_prepend_process_path!(dir::AbstractString)
    isempty(dir) && return nothing
    isdir(dir) || return nothing
    sep = path_sep()
    current = get(ENV, "PATH", "")
    entries = isempty(current) ? String[] : split(current, sep)
    dir in entries && return nothing
    ENV["PATH"] = isempty(current) ? String(dir) : string(dir, sep, current)
    return nothing
end

function find_snopt_lib()
    libname = string("lib", "snopt7", ".", Libdl.dlext)
    paths_to_try = String[]
    if Sys.iswindows()
        if haskey(ENV, "PATH")
            append!(paths_to_try, split(ENV["PATH"], ';'))
        end
    elseif haskey(ENV, "LD_LIBRARY_PATH")
        append!(paths_to_try, split(ENV["LD_LIBRARY_PATH"], ':'))
    end
    if haskey(ENV, "SNOPTDIR")
        push!(paths_to_try, ENV["SNOPTDIR"])
    end
    for path in Iterators.reverse(paths_to_try)
        libpath = joinpath(path, libname)
        d = Libdl.dlopen_e(libpath)
        if d != C_NULL
            Libdl.dlclose(d)
            return libpath
        end
    end
    return ""
end

function __init__()
    if Sys.iswindows()
        haskey(ENV, "SNOPTDIR") && maybe_prepend_process_path!(ENV["SNOPTDIR"])
        haskey(ENV, "SNOPT_GFORTRAN_BINDIR") && maybe_prepend_process_path!(ENV["SNOPT_GFORTRAN_BINDIR"])
        #@warn("""
        #      Snopt.jl: Windows is not yet supported.
        #      Please use Linux or macOS and set SNOPTDIR to the directory containing libsnopt7.
        #      """)
    end
    global libsnopt7 = find_snopt_lib()
    if isempty(libsnopt7)
        @warn """
              Snopt.jl: SNOPT library not found. has_snopt() returns false.
              Set SNOPTDIR to the directory containing libsnopt7, or add it to LD_LIBRARY_PATH:
                  export SNOPTDIR=/path/to/snopt/lib
              """
    else
        # Preload OpenMP companion library if it lives alongside libsnopt7.
        libiomp5 = replace(libsnopt7, "libsnopt7" => "libiomp5")
        isfile(libiomp5) && Libdl.dlopen(libiomp5)
    end
end

has_snopt() = !isempty(libsnopt7)

include("C_wrapper.jl")

export SnoptA
export SnoptB
export SnoptC
export SnoptProblem
export snopt!
export snopta!
export snoptb!
export snoptc!
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
