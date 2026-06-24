function start_mode_code(start::AbstractString)::Cint
    key = lowercase(strip(start))
    key == "cold" && return Cint(0)
    key == "warm" && return Cint(1)
    key == "hot"  && return Cint(2)
    throw(ArgumentError("SNOPT start mode must be Cold, Warm, or Hot; got $(repr(start))"))
end

"""
    snopt!(prob::AbstractSnoptProblem; start="Cold", name="Julia", snlog=nothing) -> Int

Solve a low-level problem in place, dispatching on its type to [`snopta!`](@ref),
[`snoptb!`](@ref), or [`snoptc!`](@ref). The problem's result fields (`status`,
`obj_val`, multipliers, and final `x`) are overwritten and the SNOPT inform code is
returned. `start` selects the SNOPT start mode (`"Cold"`, `"Warm"`, or `"Hot"`),
`name` is the ≤8-character problem name SNOPT prints, and `snlog` is an optional
major-iteration callback. `snlog` is honored by the `SnoptB` and `SnoptC` methods;
on the `SnoptA` method it is accepted for signature uniformity but ignored, since
`snOptA` has no major-iteration log hook.
"""
function snopt!(prob::SnoptB; start::String = "Cold", name::String = "Julia",
                snlog=nothing)
    return snoptb!(prob; start, name, snlog)
end

"""
    snoptb!(prob::SnoptB; start="Cold", name="Julia", snlog=nothing) -> Int

Solve a [`SnoptB`](@ref) problem in place through SNOPT's `snOptB` interface and
return the SNOPT inform code. The final point, objective, and multipliers are written
back into `prob`. Pass `snlog` to receive a [`SnoptMajorLog`](@ref) at each major
iteration (this routes the solve through SNOPT's `snKerB` reverse-communication
kernel). Also reachable through the alias [`snopt!`](@ref).
"""
function snoptb!(prob::SnoptB; start::String = "Cold", name::String = "Julia",
                 snlog=nothing)
    nc    = prob.nc
    nnCon = nc
    nnJac = nc > 0 ? prob.n : 0
    inform = snoptb!(prob.ws, start, name,
                     prob.m_eff, prob.n, nnCon, prob.nnobj, nnJac,
                     0.0, 0,
                     prob.confun, prob.objfun,
                     prob.J, prob.bl, prob.bu, prob.hs, prob.x;
                     snlog)
    prob.obj_val = prob.ws.obj_val
    prob.status  = inform
    prob.lambda  = prob.ws.lambda
    return inform
end

function snopt!(prob::SnoptA; start::String = "Cold", name::String = "Julia",
                snlog=nothing)
    # snOptA has no major-iteration log hook; snlog is accepted for signature
    # uniformity with the SnoptB/SnoptC methods and ignored.
    return snopta!(prob; start, name)
end

function snopt!(prob::SnoptC; start::String = "Cold", name::String = "Julia",
                snlog=nothing)
    return snoptc!(prob; start, name, snlog)
end
