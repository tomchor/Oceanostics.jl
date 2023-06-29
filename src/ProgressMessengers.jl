module ProgressMessengers
using Printf
import Base: +, *

export AbstractProgressMessenger
export MaxUVelocity, MaxVVelocity, MaxWVelocity
export FunctionMessenger


#+++ Basic type and addition operations with functions and strings
abstract type AbstractProgressMessenger end

@inline +(a::AbstractProgressMessenger,   b::AbstractProgressMessenger)   = sim -> a(sim) * ",    " * b(sim)
@inline *(a::AbstractProgressMessenger,   b::AbstractProgressMessenger)   = sim -> a(sim) * " " * b(sim)

const FunctionOrProgressMessenger = Union{Function, AbstractProgressMessenger}
@inline +(a::AbstractProgressMessenger,   b::FunctionOrProgressMessenger) = sim -> a(sim) * ",    " * b(sim)
@inline +(a::FunctionOrProgressMessenger, b::AbstractProgressMessenger)   = sim -> a(sim) * ",    " * b(sim)
@inline *(a::AbstractProgressMessenger,   b::FunctionOrProgressMessenger) = sim -> a(sim) * " " * b(sim)
@inline *(a::FunctionOrProgressMessenger, b::AbstractProgressMessenger)   = sim -> a(sim) * " " * b(sim)

const StringOrProgressMessenger = Union{String, AbstractProgressMessenger}
@inline +(a::AbstractProgressMessenger, b::StringOrProgressMessenger) = sim -> a(sim) * ",    $b"
@inline +(a::StringOrProgressMessenger, b::AbstractProgressMessenger) = sim -> "$a,    " * b(sim)
@inline *(a::AbstractProgressMessenger, b::StringOrProgressMessenger) = sim -> a(sim) * " $b"
@inline *(a::StringOrProgressMessenger, b::AbstractProgressMessenger) = sim -> "$a " * b(sim)
#---

#+++ Basic definitions
Base.@kwdef struct MaxUVelocity <: AbstractProgressMessenger
    prefix     :: Bool = true
    with_units :: Bool = true
end

Base.@kwdef struct MaxVVelocity <: AbstractProgressMessenger
    prefix     :: Bool = true
    with_units :: Bool = true
end

Base.@kwdef struct MaxWVelocity <: AbstractProgressMessenger
    prefix     :: Bool = true
    with_units :: Bool = true
end

@inline function (mu::MaxUVelocity)(sim)
    u_max = maximum(abs, sim.model.velocities.u)
    message = @sprintf("%.2e", u_max)
    mu.prefix     && (message = "|u|ₘₐₓ = " * message)
    mu.with_units && (message = message * " m/s")
    return message
end

@inline function (mv::MaxVVelocity)(sim)
    v_max = maximum(abs, sim.model.velocities.v)
    message = @sprintf("%.2e", v_max)
    mv.prefix     && (message = "|v|ₘₐₓ = " * message)
    mv.with_units && (message = message * " m/s")
    return message
end

@inline function (mw::MaxWVelocity)(sim)
    w_max = maximum(abs, sim.model.velocities.w)
    message = @sprintf("%.2e", w_max)
    mw.prefix     && (message = "|w|ₘₐₓ = " * message)
    mw.with_units && (message = message * " m/s")
    return message
end
#---
 
#+++ FunctionMessenger
Base.@kwdef struct FunctionMessenger{F} <: AbstractProgressMessenger
    func :: F
end

function (muvw::FunctionMessenger)(sim)
    message = muvw.func(sim)
    return message
end
#---

end # module
