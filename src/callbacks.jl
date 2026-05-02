snopt_no_progress(event) = true

workspace_value(ws_iw::Vector{Int32}, index::Int) =
    length(ws_iw) >= index ? max(Int(ws_iw[index]), 0) : 0

function progress_iters(ws_iw::Vector{Int32})
    return (workspace_value(ws_iw, IW_MAJOR_ITNS),
            workspace_value(ws_iw, IW_MINOR_ITNS))
end

call_progress(::Nothing, event) = true

call_progress(callback, event) = callback(event) !== false

legacy_progress_callback(log_fn::Function) =
    event -> event.kind === :objective ? log_fn(event.major_iter, event.x, event.f) : true

copy_cdouble_vector(ptr::Ptr{Cdouble}, len::Integer) =
    len <= 0 ? Float64[] : copy(unsafe_wrap(Array, ptr, Int(len)))

copy_cint32_vector(ptr::Ptr{Cint}, len::Integer) =
    len <= 0 ? Int32[] : copy(unsafe_wrap(Array, ptr, Int(len)))

mutable struct SnoptCallbackState
    exception::Any
end

SnoptCallbackState() = SnoptCallbackState(nothing)

struct StatefulCallback{F} <: Function
    f::F
    state::SnoptCallbackState
end

(callback::StatefulCallback)(args...) = callback.f(args...)

function register_callback_state!(callback, state::SnoptCallbackState)
    return StatefulCallback(callback, state)
end

callback_state(callback::StatefulCallback) = callback.state
callback_state(_) = nothing

function record_callback_exception!(state::SnoptCallbackState, err)
    state.exception === nothing && (state.exception = err)
    return nothing
end

function reset_callback_exception!(callbacks...)
    for callback in callbacks
        state = callback_state(callback)
        state !== nothing && (state.exception = nothing)
    end
    return nothing
end

function rethrow_callback_exception!(callbacks...)
    for callback in callbacks
        state = callback_state(callback)
        if state !== nothing && state.exception !== nothing
            err = state.exception
            state.exception = nothing
            throw(err)
        end
    end
    return nothing
end

"""
    make_snlog(callback)
Create a Julia callback compatible with SNOPT's `snLog` hook. `callback` is
called with a [`SnoptMajorLog`](@ref) after each major-iteration log event.
Returning `false` requests termination from SNOPT.
SNOPT passes these arguments in the same order used by `snLog` in the SNOPT 7
interface. The `@cfunction` signature in `snoptb!` must stay in this order.

"""

function make_snlog(callback)
    state = SnoptCallbackState()

    function snlog(iAbort_::Ptr{Cint}, KTcond_::Ptr{Cint},
                   MjrPrt_::Ptr{Cint}, minimz_::Ptr{Cint},
                   n_::Ptr{Cint}, nb_::Ptr{Cint}, nnCon0_::Ptr{Cint},
                   nnObj_::Ptr{Cint}, nS_::Ptr{Cint},
                   itn_::Ptr{Cint}, nMajor_::Ptr{Cint},
                   nMinor_::Ptr{Cint}, nSwap_::Ptr{Cint},
                   condHz_::Ptr{Cdouble}, iObj_::Ptr{Cint},
                   sclObj_::Ptr{Cdouble}, ObjAdd_::Ptr{Cdouble},
                   fObj_::Ptr{Cdouble}, fMrt_::Ptr{Cdouble},
                   PenNrm_::Ptr{Cdouble}, step_::Ptr{Cdouble},
                   prInf_::Ptr{Cdouble}, duInf_::Ptr{Cdouble},
                   vimax_::Ptr{Cdouble}, virel_::Ptr{Cdouble},
                   hs_::Ptr{Cint}, ne_::Ptr{Cint},
                   nlocJ_::Ptr{Cint}, locJ_::Ptr{Cint},
                   indJ_::Ptr{Cint}, Jcol_::Ptr{Cdouble},
                   Ascale_::Ptr{Cdouble}, bl_::Ptr{Cdouble},
                   bu_::Ptr{Cdouble}, Fx_::Ptr{Cdouble},
                   fCon_::Ptr{Cdouble}, yCon_::Ptr{Cdouble},
                   x_::Ptr{Cdouble}, cu_::Ptr{UInt8},
                   lencu_::Ptr{Cint}, iu_::Ptr{Cint},
                   leniu_::Ptr{Cint}, ru_::Ptr{Cdouble},
                   lenru_::Ptr{Cint}, cw_::Ptr{UInt8},
                   lencw_::Ptr{Cint}, iw_::Ptr{Cint},
                   leniw_::Ptr{Cint}, rw_::Ptr{Cdouble},
                   lenrw_::Ptr{Cint})::Cvoid
        try
            n = Int(unsafe_load(n_))
            nb = Int(unsafe_load(nb_))
            nncon = Int(unsafe_load(nnCon0_))
            ktcond = unsafe_wrap(Array, KTcond_, 2)
            penalty = unsafe_wrap(Array, PenNrm_, 4)
            objective_add = unsafe_load(ObjAdd_)
            f_objective = unsafe_load(fObj_)
            f_merit = unsafe_load(fMrt_)
            objective_scale = unsafe_load(sclObj_)
            iobj = Int(unsafe_load(iObj_))
            x_values = copy_cdouble_vector(x_, nb)
            linear_objective = (iobj > 0 && n + iobj <= length(x_values)) ?
                objective_scale * x_values[n + iobj] : 0.0
            event = SnoptMajorLog(
                Int(unsafe_load(itn_)),
                Int(unsafe_load(nMajor_)),
                Int(unsafe_load(nMinor_)),
                Int(unsafe_load(nS_)),
                Int(unsafe_load(nSwap_)),
                objective_add + linear_objective + f_objective,
                objective_add + linear_objective + f_merit,
                Float64(penalty[3]),
                unsafe_load(step_),
                unsafe_load(prInf_),
                unsafe_load(duInf_),
                unsafe_load(vimax_),
                unsafe_load(virel_),
                unsafe_load(condHz_),
                objective_scale,
                objective_add,
                f_objective,
                f_merit,
                Int(unsafe_load(minimz_)),
                n,
                nb,
                nncon,
                Int(unsafe_load(nnObj_)),
                (ktcond[1] != 0, ktcond[2] != 0),
                x_values,
                copy_cdouble_vector(fCon_, nncon),
                copy_cdouble_vector(Fx_, nncon),
                copy_cdouble_vector(yCon_, nncon),
                copy_cint32_vector(hs_, nb)
            )
            unsafe_store!(iAbort_, call_progress(callback, event) ? Cint(0) : Cint(1))
        catch err
            record_callback_exception!(state, err)
            unsafe_store!(iAbort_, Cint(1))
        end
        return
    end
    return register_callback_state!(snlog, state)
end

"""
    make_objfun(eval_obj, eval_grad, ws_iw; callback=nothing)
Create the SNOPT objective callback. If `callback` is provided, it is called as
`callback(event::NamedTuple)::Bool` after objective-value evaluations. Returning
`false` requests SNOPT termination. The default `nothing` records no progress
and avoids per-evaluation event allocation.
Objective events contain `kind = :objective`, `mode`, `major_iter`,
`minor_iter`, `x`, and `f`.

"""

function make_objfun(eval_obj::Function, eval_grad::Function,
                     ws_iw::Vector{Int32}; callback=nothing)
    state = SnoptCallbackState()

    function objfun(mode_::Ptr{Cint}, nnobj_::Ptr{Cint}, x_::Ptr{Cdouble},
                    f_::Ptr{Cdouble}, g_::Ptr{Cdouble}, _::Ptr{Cint},
                    _::Ptr{UInt8}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                    _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        try
            mode  = unsafe_load(mode_)
            nnobj = Int(unsafe_load(nnobj_))
            x     = unsafe_wrap(Array, x_, nnobj)
            if mode == 0 || mode == 2
                f = eval_obj(x)
                unsafe_store!(f_, f)
                if callback !== nothing
                    major_itns, minor_itns = progress_iters(ws_iw)
                    event = (kind = :objective, mode = Int(mode),
                             major_iter = major_itns, minor_iter = minor_itns,
                             x = copy(x), f = f)
                    if !call_progress(callback, event)
                        unsafe_store!(mode_, Cint(-2))
                        return
                    end
                end
            end
            if mode == 1 || mode == 2
                g = unsafe_wrap(Array, g_, nnobj)
                eval_grad(g, x)
            end
        catch err
            record_callback_exception!(state, err)
            unsafe_store!(mode_, Cint(-1))
        end
        return
    end
    return register_callback_state!(objfun, state)
end

function make_objfun(eval_obj::Function, eval_grad::Function, log_fn::Function,
                     ws_iw::Vector{Int32})
    return make_objfun(eval_obj, eval_grad, ws_iw;
                       callback=legacy_progress_callback(log_fn))
end

"""
    make_confun(eval_con, eval_jac, J, ws_iw; callback=nothing)
Create the SNOPT constraint callback. If `callback` is provided, it is called as
`callback(event::NamedTuple)::Bool` after constraint-value evaluations.
Returning `false` requests SNOPT termination.
Constraint events contain `kind = :constraint`, `mode`, `major_iter`,
`minor_iter`, `x`, and `c`. The default `nothing` records no progress and avoids
per-evaluation event allocation.

"""

function make_confun(eval_con::Function, eval_jac::Function, J,
                     ws_iw::Vector{Int32}; callback=nothing)
    state = SnoptCallbackState()

    function confun(mode_::Ptr{Cint}, nncon_::Ptr{Cint}, nnjac_::Ptr{Cint},
                    _::Ptr{Cint}, x_::Ptr{Cdouble}, c_::Ptr{Cdouble},
                    J_::Ptr{Cdouble}, _::Ptr{Cint},
                    _::Ptr{UInt8}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                    _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        try
            mode  = unsafe_load(mode_)
            nncon = Int(unsafe_load(nncon_))
            nnjac = Int(unsafe_load(nnjac_))
            x     = unsafe_wrap(Array, x_, nnjac)
            if mode == 0 || mode == 2
                c = unsafe_wrap(Array, c_, nncon)
                eval_con(c, x)
                if callback !== nothing
                    major_itns, minor_itns = progress_iters(ws_iw)
                    event = (kind = :constraint, mode = Int(mode),
                             major_iter = major_itns, minor_iter = minor_itns,
                             x = copy(x), c = copy(c))
                    if !call_progress(callback, event)
                        unsafe_store!(mode_, Cint(-2))
                        return
                    end
                end
            end
            if mode == 1 || mode == 2
                jnzval = unsafe_wrap(Array, J_, nnz(J))
                eval_jac(jnzval, x)
            end
        catch err
            record_callback_exception!(state, err)
            unsafe_store!(mode_, Cint(-1))
        end
        return
    end
    return register_callback_state!(confun, state)
end

make_confun(eval_con::Function, eval_jac::Function, J) =
    make_confun(eval_con, eval_jac, J, Int32[])

"""
    make_usrfun_a(eval_F; eval_G=nothing, callback=nothing)
Create the combined callback used by SNOPT-A. `eval_F(F, x)` fills the
objective/constraint row values. If derivatives are requested, `eval_G(G, x)`
fills the nonlinear derivative values in the order specified by `iGfun` and
`jGvar`. Returning `false` from `callback(event)` requests SNOPT termination.

"""

function make_usrfun_a(eval_F::Function; eval_G=nothing, callback=nothing)
    state = SnoptCallbackState()

    function usrfun(status_::Ptr{Cint}, n_::Ptr{Cint}, x_::Ptr{Cdouble},
                    needF_::Ptr{Cint}, neF_::Ptr{Cint}, F_::Ptr{Cdouble},
                    needG_::Ptr{Cint}, neG_::Ptr{Cint}, G_::Ptr{Cdouble},
                    _::Ptr{UInt8}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                    _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        try
            n     = Int(unsafe_load(n_))
            needF = Int(unsafe_load(needF_))
            neF   = Int(unsafe_load(neF_))
            needG = Int(unsafe_load(needG_))
            neG   = Int(unsafe_load(neG_))
            x     = unsafe_wrap(Array, x_, n)
            if needF > 0
                F = unsafe_wrap(Array, F_, neF)
                eval_F(F, x)
                if callback !== nothing
                    event = (kind = :function, x = copy(x), F = copy(F))
                    if !call_progress(callback, event)
                        unsafe_store!(status_, Cint(-2))
                        return
                    end
                end
            end
            if needG > 0
                if eval_G === nothing
                    unsafe_store!(status_, Cint(-1))
                    return
                end
                G = unsafe_wrap(Array, G_, neG)
                eval_G(G, x)
            end
        catch err
            record_callback_exception!(state, err)
            unsafe_store!(status_, Cint(-1))
        end
        return
    end
    return register_callback_state!(usrfun, state)
end

"""
    make_usrfun_c(eval_obj, eval_grad, eval_con, eval_jac, J, ws_iw; callback=nothing)
Create the combined callback used by SNOPT-C. This is the SNOPT-C equivalent
of `make_objfun` plus `make_confun`.

"""

function make_usrfun_c(eval_obj::Function, eval_grad::Function,
                       eval_con::Function, eval_jac::Function, J,
                       ws_iw::Vector{Int32}; callback=nothing)
    state = SnoptCallbackState()

    function usrfun(mode_::Ptr{Cint}, nnobj_::Ptr{Cint}, nncon_::Ptr{Cint},
                    nnjac_::Ptr{Cint}, nnL_::Ptr{Cint}, negcon_::Ptr{Cint},
                    x_::Ptr{Cdouble}, fobj_::Ptr{Cdouble}, gobj_::Ptr{Cdouble},
                    fcon_::Ptr{Cdouble}, gcon_::Ptr{Cdouble}, status_::Ptr{Cint},
                    _::Ptr{UInt8}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                    _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        try
            mode   = Int(unsafe_load(mode_))
            nnobj  = Int(unsafe_load(nnobj_))
            nncon  = Int(unsafe_load(nncon_))
            nnjac  = Int(unsafe_load(nnjac_))
            nnL    = Int(unsafe_load(nnL_))
            negcon = Int(unsafe_load(negcon_))
            x      = unsafe_wrap(Array, x_, nnL)
            if mode == 0 || mode == 2
                f = eval_obj(x)
                unsafe_store!(fobj_, f)
                c = unsafe_wrap(Array, fcon_, nncon)
                eval_con(c, x)
                if callback !== nothing
                    major_itns, minor_itns = progress_iters(ws_iw)
                    event = (kind = :combined, mode = mode,
                             major_iter = major_itns, minor_iter = minor_itns,
                             x = copy(x), f = f, c = copy(c))
                    if !call_progress(callback, event)
                        unsafe_store!(status_, Cint(-2))
                        return
                    end
                end
            end
            if mode == 1 || mode == 2
                g = unsafe_wrap(Array, gobj_, nnobj)
                eval_grad(g, x)
                jnzval = unsafe_wrap(Array, gcon_, negcon)
                if negcon == nnz(J)
                    eval_jac(jnzval, x)
                else
                    jtmp = unsafe_wrap(Array, gcon_, nnz(J))
                    eval_jac(jtmp, x)
                end
            end
        catch err
            record_callback_exception!(state, err)
            unsafe_store!(status_, Cint(-1))
        end
        return
    end
    return register_callback_state!(usrfun, state)
end

function make_usrfun_c(eval_obj::Function, eval_grad::Function,
                       eval_con::Function, eval_jac::Function, J)
    return make_usrfun_c(eval_obj, eval_grad, eval_con, eval_jac, J, Int32[])
end

function make_dummy_confun()
    function dummy_confun(_::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cint},
                          _::Ptr{Cdouble}, _::Ptr{Cdouble}, _::Ptr{Cdouble},
                          _::Ptr{Cint}, _::Ptr{UInt8}, _::Ptr{Cint},
                          _::Ptr{Cint}, _::Ptr{Cint}, _::Ptr{Cdouble}, _::Ptr{Cint})::Cvoid
        return
    end
    return dummy_confun
end
