```@meta
CurrentModule = SNOPT
```

# API reference

```@index
```

## High-level solve

```@docs
snopt
SnoptResult
SnoptMajorLog
SnoptMemory
SNOPT_STATUS
```

## Problem types

```@docs
AbstractSnoptProblem
SnoptB
SnoptProblem
SnoptC
SnoptA
```

## In-place solvers

```@docs
snopt!
snoptb!
snoptc!
snopta!
```

## Workspace, options, and memory

```@docs
initialize
set_option!
read_options
specs_status_message
snmemb
```

## Callback builders

```@docs
make_objfun
make_confun
make_dummy_confun
make_usrfun_c
make_usrfun_a
make_snlog
snopt_no_progress
```

## Library loading

These helpers are not exported; access them as `SNOPT.has_snopt`, etc.

```@docs
SNOPT.has_snopt
SNOPT.find_snopt_lib
SNOPT.SnoptWorkspace
```
