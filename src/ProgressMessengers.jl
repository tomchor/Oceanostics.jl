module ProgressMessengers
using Printf
import Base: +

export AbstractProgressMessenger
export MaxUVelocity, MaxVVelocity, MaxWVelocity
export MaxVelocities


abstract type AbstractProgressMessenger end

#+++ Basic definitions
Base.@kwdef struct MaxUVelocity <: AbstractProgressMessenger
    with_units :: Bool = false
end

Base.@kwdef struct MaxVVelocity <: AbstractProgressMessenger
    with_units :: Bool = false
end

Base.@kwdef struct MaxWVelocity <: AbstractProgressMessenger
    with_units :: Bool = false
end

@inline function (mu::MaxUVelocity)(sim)
    u_max = maximum(abs, sim.model.velocities.u)
    message = @sprintf("|u|ₘₐₓ = %.2e", u_max)
    mu.with_units && (message = message * " m/s")
    return message
end

@inline function (mv::MaxVVelocity)(sim)
    v_max = maximum(abs, sim.model.velocities.v)
    message = @sprintf("|v|ₘₐₓ = %.2e", v_max)
    mv.with_units && (message = message * " m/s")
    return message
end

@inline function (mw::MaxWVelocity)(sim)
    w_max = maximum(abs, sim.model.velocities.w)
    message = @sprintf("|w|ₘₐₓ = %.2e", w_max)
    mw.with_units && (message = message * " m/s")
    return message
end
#---
 
#+++ MaxVelocities
Base.@kwdef struct MaxVelocities{F} <: AbstractProgressMessenger
    func :: F
    with_units :: Bool = false
end

function MaxVelocities(; with_units=false)
    max_u = MaxUVelocity(with_units = false)
    max_v = MaxVVelocity(with_units = false)
    max_w = MaxWVelocity(with_units = false)
    return MaxVelocities(max_u + max_v + max_w, with_units)
end

function (muvw::MaxVelocities)(sim)
    message = muvw.func(sim)
    muvw.with_units && (message = message * " m/s")
    return message
end
#---

@inline +(a::AbstractProgressMessenger,   b::AbstractProgressMessenger)   = sim -> a(sim) * ",    " * b(sim)

const FunctionOrProgressMessenger = Union{Function, AbstractProgressMessenger}
@inline +(a::AbstractProgressMessenger,   b::FunctionOrProgressMessenger) = sim -> a(sim) * ",    " * b(sim)
@inline +(a::FunctionOrProgressMessenger, b::AbstractProgressMessenger)   = sim -> a(sim) * ",    " * b(sim)

const StringOrProgressMessenger = Union{String, AbstractProgressMessenger}
@inline +(a::AbstractProgressMessenger, b::StringOrProgressMessenger) = sim -> a(sim) * ",    $b"
@inline +(a::StringOrProgressMessenger, b::AbstractProgressMessenger) = sim -> "$a,    " + b(sim)

end # module
