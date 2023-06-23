module ProgressMessengers
using Printf
import Base: +

export AbstractProgressMessenger
export MaxUVelocity, MaxVVelocity, MaxWVelocity
export MaxVelocities


abstract type AbstractProgressMessenger end

Base.@kwdef struct MaxUVelocity <: AbstractProgressMessenger
    with_units :: Bool = false
end

Base.@kwdef struct MaxVVelocity <: AbstractProgressMessenger
    with_units :: Bool = false
end

Base.@kwdef struct MaxWVelocity <: AbstractProgressMessenger
    with_units :: Bool = false
end

Base.@kwdef struct MaxVelocities{U, V, W} <: AbstractProgressMessenger
    max_u :: U
    max_v :: V
    max_w :: W
    with_units :: Bool = false
end

function MaxVelocities(; with_units = false)
    max_u = MaxUVelocity(with_units=false)
    max_v = MaxVVelocity(with_units=false)
    max_w = MaxWVelocity(with_units=false)
    return MaxVelocities(max_u, max_v, max_w, with_units)
end

@inline function (mu::MaxUVelocity)(sim)
    return "|u⃗| = " + mu.max_u + mu.max_v + mu.max_w
end

@inline function (mv::MaxVVelocity)(sim)
    v_max = maximum(abs, sim.model.velocities.v)
    message = @sprintf("|v|ₘₐₓ: %.2e", v_max)
    mv.with_units && (message = message * " m/s")
    return message
end

@inline function (mw::MaxWVelocity)(sim)
    w_max = maximum(abs, sim.model.velocities.w)
    message = @sprintf("|w|ₘₐₓ: %.2e", w_max)
    mw.with_units && (message = message * " m/s")
    return message
end



const FunctionOrProgressMessenger = Union{Function, AbstractProgressMessenger}
function +(a::AbstractProgressMessenger, b::FunctionOrProgressMessenger)
    return function c(sim)
               return a(sim) * ",    " * b(sim)
           end
end
+(a::FunctionOrProgressMessenger, b::AbstractProgressMessenger) = +(b, a)


const StringOrProgressMessenger = Union{String, AbstractProgressMessenger}
function +(a::AbstractProgressMessenger, b::StringOrProgressMessenger)
    return function c(sim)
               return a(sim) * ",    $b"
           end
end
+(a::StringOrProgressMessenger, b::AbstractProgressMessenger) = +(b, a)


end # module
