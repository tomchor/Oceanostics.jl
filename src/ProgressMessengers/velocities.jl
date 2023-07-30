#+++ Basic definitions
Base.@kwdef struct MaxUVelocity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

Base.@kwdef struct MaxVVelocity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

Base.@kwdef struct MaxWVelocity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

@inline function (mu::MaxUVelocity)(sim)
    u_max = maximum(abs, sim.model.velocities.u)
    message = @sprintf("%.2e", u_max)
    mu.with_prefix     && (message = "|u|ₘₐₓ = " * message)
    mu.with_units && (message = message * " m/s")
    return_or_print(message, mu)
end

@inline function (mv::MaxVVelocity)(sim)
    v_max = maximum(abs, sim.model.velocities.v)
    message = @sprintf("%.2e", v_max)
    mv.with_prefix     && (message = "|v|ₘₐₓ = " * message)
    mv.with_units && (message = message * " m/s")
    return_or_print(message, mv)
end

@inline function (mw::MaxWVelocity)(sim)
    w_max = maximum(abs, sim.model.velocities.w)
    message = @sprintf("%.2e", w_max)
    mw.with_prefix     && (message = "|w|ₘₐₓ = " * message)
    mw.with_units && (message = message * " m/s")
    return_or_print(message, mw)
end
#---

#+++ MaxVelocities
Base.@kwdef struct MaxVelocities{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    max_u       :: PM
    max_v       :: PM
    max_w       :: PM
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

MaxVelocities(; max_u = MaxUVelocity(with_prefix = false, with_units = false, print = false),
                max_v = MaxVVelocity(with_prefix = false, with_units = false, print = false),
                max_w = MaxWVelocity(with_prefix = false, with_units = false, print = false),
                with_prefix = true,
                with_units = true,
                print = false) = MaxVelocities{AbstractProgressMessenger}(max_u, max_v, max_w, with_prefix, with_units, print)

function (maxvel::MaxVelocities)(simulation)
    message = ("[" * maxvel.max_u + maxvel.max_v + maxvel.max_w * "]")(simulation)
    maxvel.with_prefix && (message = "|u⃗|ₘₐₓ = " * message)
    maxvel.with_units  && (message = message * " m/s")
    return_or_print(message, maxvel)
end
#---
