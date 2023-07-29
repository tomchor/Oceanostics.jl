#+++ Basic definitions
Base.@kwdef struct MaxUVelocity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
end

Base.@kwdef struct MaxVVelocity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
end

Base.@kwdef struct MaxWVelocity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
end

@inline function (mu::MaxUVelocity)(sim)
    u_max = maximum(abs, sim.model.velocities.u)
    message = @sprintf("%.2e", u_max)
    mu.with_prefix     && (message = "|u|ₘₐₓ = " * message)
    mu.with_units && (message = message * " m/s")
    return message
end

@inline function (mv::MaxVVelocity)(sim)
    v_max = maximum(abs, sim.model.velocities.v)
    message = @sprintf("%.2e", v_max)
    mv.with_prefix     && (message = "|v|ₘₐₓ = " * message)
    mv.with_units && (message = message * " m/s")
    return message
end

@inline function (mw::MaxWVelocity)(sim)
    w_max = maximum(abs, sim.model.velocities.w)
    message = @sprintf("%.2e", w_max)
    mw.with_prefix     && (message = "|w|ₘₐₓ = " * message)
    mw.with_units && (message = message * " m/s")
    return message
end
#---

#+++ MaxVelocities
function MaxVelocities(; with_prefix = true, with_units = true)
    max_u = MaxUVelocity(with_prefix = false, with_units = false)
    max_v = MaxVVelocity(with_prefix = false, with_units = false)
    max_w = MaxWVelocity(with_prefix = false, with_units = false)

    message = "[" * max_u + max_v + max_w * "]"
    with_prefix && (message = "|u⃗|ₘₐₓ =" * message)
    with_units  && (message = message * "m/s")
    return message
end
#---
