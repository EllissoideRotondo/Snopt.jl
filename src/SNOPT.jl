module SNOPT

using Libdl
using SparseArrays: SparseMatrixCSC, nnz

# Mutable typed global so __init__ can set the resolved path at runtime.
# All ccall sites reference this variable; Julia re-evaluates it on each call,
# which is negligible overhead compared to any SNOPT solve.
global libsnopt7::String = ""

# A library that dlopens is not necessarily usable: SNOPT can be built without
# the snopt-interface C shims, and then every ccall in this package would fail
# at solve time instead of at load time. Probe the symbols we actually call.
const REQUIRED_SNOPT_SYMBOLS = (:f_sninitx, :f_snend, :f_snset, :f_snmem, :f_snoptb)

function loadable_library_path(libpath::AbstractString)
    isempty(libpath) && return ""
    d = Libdl.dlopen_e(libpath)
    d == C_NULL && return ""
    try
        for sym in REQUIRED_SNOPT_SYMBOLS
            Libdl.dlsym_e(d, sym) == C_NULL && return ""
        end
    finally
        Libdl.dlclose(d)
    end
    return String(libpath)
end

"""
    find_snopt_lib() -> String

Search for a loadable `libsnopt7` and return its absolute path, or an empty string
if none is found. The search checks `SNOPTDIR` first (warning if it is set but the
library there cannot be loaded), then the platform library path
(`LD_LIBRARY_PATH`, also `DYLD_LIBRARY_PATH` on macOS, or `PATH` on Windows), and
finally the system loader's default search paths (for example `/usr/lib` via
`ld.so` on Linux). This is run once during `__init__` to set the global
`SNOPT.libsnopt7`; call it directly to diagnose why a library is not being picked
up. See also [`has_snopt`](@ref).
"""
function find_snopt_lib()
    libname = string("lib", "snopt7", ".", Libdl.dlext)
    snoptdir = get(ENV, "SNOPTDIR", "")
    if !isempty(snoptdir)
        libpath = loadable_library_path(joinpath(snoptdir, libname))
        isempty(libpath) || return libpath
        @warn "SNOPTDIR is set but no loadable SNOPT library was found there; " *
              "falling back to the platform library path" SNOPTDIR = snoptdir libname
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

    # Last resort: let the system loader search its default paths (ld.so cache,
    # /usr/lib, ...), where a bare directory scan above would never look.
    handle = Libdl.dlopen_e(libname)
    if handle != C_NULL
        libpath = String(Libdl.dlpath(handle))
        Libdl.dlclose(handle)
        return loadable_library_path(libpath)
    end
    return ""
end

function __init__()
    global libsnopt7 = find_snopt_lib()
    if !isempty(libsnopt7)
        # Preload OpenMP companion library if it lives alongside libsnopt7.
        # Use the non-throwing dlopen so a missing or incompatible companion
        # never aborts module initialization; libsnopt7 itself already loaded.
        libiomp5 = replace(libsnopt7, "libsnopt7" => "libiomp5")
        isfile(libiomp5) && Libdl.dlopen_e(libiomp5)
    end
    init_callback_pointers!()
end

"""
    has_snopt() -> Bool

Return `true` if a usable SNOPT library was located and loaded during module
initialization. When this is `false`, the resolved path in the global
`SNOPT.libsnopt7` is empty and any solve raises an error explaining how to provide
the library; see
[`find_snopt_lib`](@ref) to diagnose.
"""
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
