```@meta
CurrentModule = SNOPT
```

# Installation

## Requirements

Install Julia 1.10 or later. Obtain a licensed SNOPT 7 shared library.
The library must include SNOPT's `snopt-interface` C functions.

SNOPT.jl targets SNOPT 7.7. Other major versions may use different workspace
layouts. A mismatched version can report incorrect iteration or timing data.

## Add the package

Use your Julia package registry when it contains SNOPT:

```julia
import Pkg
Pkg.add("SNOPT")
```

Install the current source version with:

```julia
import Pkg
Pkg.develop(url = "https://github.com/EllissoideRotondo/SNOPT.jl")
```

The package loads without the SNOPT library. [`SNOPT.has_snopt`](@ref) then
returns `false`, and solve attempts report a setup error.

## Provide the shared library

| Platform | Filename |
| --- | --- |
| Linux | `libsnopt7.so` |
| Intel macOS | `libsnopt7.dylib` |
| Windows | `libsnopt7.dll` |

Set `SNOPTDIR` to the directory containing that file.

Linux and macOS:

```bash
export SNOPTDIR=/path/to/snopt/lib
```

Windows PowerShell:

```powershell
$env:SNOPTDIR = "C:\path\to\snopt\lib"
```

Set the variable before `using SNOPT`. Restart Julia after changing it.

Some licenses also require `SNOPT_LICENSE`. Follow the instructions supplied
with your SNOPT distribution.

## Required C interface

The library must export the `f_*` functions from
[snopt-interface](https://github.com/snopt/snopt-interface). Examples include
`f_sninitx`, `f_snoptb`, and `f_snkera`.

Vendor libraries usually contain these functions. A library built only from
the Fortran source usually does not. SNOPT.jl checks every required function
before accepting a library.

## Fallback search paths

When `SNOPTDIR` is unset, SNOPT.jl checks standard platform search paths.

```bash
export LD_LIBRARY_PATH=/path/to/snopt/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/path/to/snopt/lib:$DYLD_LIBRARY_PATH
```

Windows uses `PATH`. The system loader also checks its default locations.

If `libiomp5` is beside `libsnopt7`, SNOPT.jl tries to preload it.
`libiomp5` is an OpenMP runtime library. Failure to preload it does not stop
library discovery.

## Verify the setup

Run:

```bash
julia -e 'using SNOPT; @assert SNOPT.has_snopt(); println(SNOPT.libsnopt7)'
```

For more detail, inspect:

```julia
using SNOPT

SNOPT.has_snopt()
SNOPT.libsnopt7
SNOPT.find_snopt_lib()
```

`find_snopt_lib()` repeats the search. It helps diagnose path or symbol errors.

## Platform notes

- Linux is tested with a compatible `libsnopt7.so`.
- Intel macOS is expected to work, but is not tested by the maintainers.
- Apple Silicon is not currently supported.
- Windows requires a MinGW-built `libsnopt7.dll`.

The vendor's Intel-built Windows library has an incompatible calling convention.
Use the MinGW build or Windows Subsystem for Linux.
