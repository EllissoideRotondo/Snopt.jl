```@meta
CurrentModule = SNOPT
```

# Installation

## Add the package

```julia
import Pkg
Pkg.add("SNOPT")
```

The package installs and loads without the SNOPT library present — it only
depends on the `Libdl` and `SparseArrays` standard libraries. Without a usable
`libsnopt7`, [`SNOPT.has_snopt`](@ref) returns `false` and any attempt to solve
raises an informative error.

## Provide the SNOPT library

You must supply a SNOPT shared library built for your platform:

| Platform | Library file     |
|----------|------------------|
| Linux    | `libsnopt7.so`   |
| macOS Intel | `libsnopt7.dylib`|
| Windows  | `libsnopt7.dll`  |

SNOPT.jl targets **SNOPT 7.7**. It reads iteration counters and the reported run
time from fixed offsets in SNOPT's work arrays, which are internal to that
release series, so a different major version may load and then report wrong
counters.

The library must also export the
[snopt-interface](https://github.com/snopt/snopt-interface) `f_*` C entry
points (`f_sninitx`, `f_snoptb`, `f_snkera`, ...). Vendor-supplied binaries
usually include them; a library built from the bare Fortran sources does not,
and needs the snopt-interface shims compiled in. SNOPT.jl probes for every
`f_*` symbol it calls before accepting a library, so a build without them is
rejected at load time (with a warning and a fallback to the next search
location) rather than failing at the first solve.

!!! note "License file"
    Depending on how your SNOPT distribution is licensed, the library may
    require the `SNOPT_LICENSE` environment variable to point at your license
    file. A mislicensed library can load fine and still refuse to solve —
    consult the setup instructions that came with your SNOPT distribution.

The most reliable way to point the package at it is the `SNOPTDIR` environment
variable, set to the **directory** that contains the library:

```bash
export SNOPTDIR=/path/to/snopt/lib
```

`SNOPTDIR` is the recommended setup on Linux and macOS. If it is unset (or set
but no loadable library is found there, which emits a warning), SNOPT.jl also
searches the platform library-path variables:

```bash
export LD_LIBRARY_PATH=/path/to/snopt/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/path/to/snopt/lib:$DYLD_LIBRARY_PATH   # macOS
```

On Windows, the library is searched on the `PATH`. As a last resort the system
loader's default search paths are tried too, so a `libsnopt7` installed in a
standard location such as `/usr/lib` is found without any environment variable.

!!! tip "OpenMP companion library"
    If an OpenMP runtime named `libiomp5` sits next to `libsnopt7` in the same
    directory, it is preloaded automatically. A missing or incompatible companion
    never aborts loading — only `libsnopt7` itself is required.

The environment must be set **before** `using SNOPT`; the library path is
resolved once, in the module's `__init__`. If you change `SNOPTDIR` afterwards,
restart Julia.

## Verify the setup

```julia
using SNOPT

SNOPT.has_snopt()        # true if the library was found and loaded
SNOPT.libsnopt7          # the resolved absolute path (empty string if not found)
SNOPT.find_snopt_lib()   # re-run the search to diagnose path problems
```

## Platform notes

**Linux** is tested with a compatible `libsnopt7`.

**macOS on Intel** should work with a compatible x86_64 `libsnopt7.dylib`, but it
has not been tested by the maintainers. Apple Silicon is not currently tested or
supported.

**Windows** requires a `libsnopt7.dll` compiled from the SNOPT Fortran source
with [MinGW](https://www.mingw-w64.org/); the Intel-compiled distribution is not
ABI-compatible with the `ccall` signatures this package uses. If recompiling is
not practical, running under [WSL](https://learn.microsoft.com/en-us/windows/wsl/)
with the Linux library is a working alternative.
