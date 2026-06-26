```@meta
CurrentModule = SNOPT
```

# Optimization.jl integration

For modeling-first workflows, the preferred interface will be
[Optimization.jl](https://github.com/SciML/Optimization.jl), which can handle
problem assembly and automatic differentiation through the SciML optimization
stack.

That support is currently in progress. Until it is available, use [`snopt`](@ref)
directly for ordinary nonlinear programs, or the [low-level interface](@ref
Low-level-interface) when you need direct access to SNOPT's `snOptA`, `snOptB`,
or `snOptC` interfaces.
