module ProgressMessengers
using DocStringExtensions

using Printf
using Oceananigans.Utils: prettytime
using Oceananigans.Simulations: iteration

import Base: +, *

export AbstractProgressMessenger
export MaxUVelocity, MaxVVelocity, MaxWVelocity
export MaxVelocities
export WalltimePerTimestep, Walltime
export FunctionMessenger

abstract type AbstractProgressMessenger end
 
const comma = ", "
const space = " "

#+++ FunctionMessenger
Base.@kwdef struct FunctionMessenger{F} <: AbstractProgressMessenger
    func :: F
end

function (fmessenger::FunctionMessenger)(sim)
    message = fmessenger.func(sim)
    return message
end
#---

#+++ Basic operations with functions and strings
@inline +(a::AbstractProgressMessenger,   b::AbstractProgressMessenger)   = FunctionMessenger(sim -> a(sim) * comma * b(sim))
@inline *(a::AbstractProgressMessenger,   b::AbstractProgressMessenger)   = FunctionMessenger(sim -> a(sim) * space * b(sim))

const FunctionOrProgressMessenger = Union{Function, AbstractProgressMessenger}
@inline +(a::AbstractProgressMessenger,   b::FunctionOrProgressMessenger) = FunctionMessenger(sim -> a(sim) * comma * b(sim))
@inline +(a::FunctionOrProgressMessenger, b::AbstractProgressMessenger)   = FunctionMessenger(sim -> a(sim) * comma * b(sim))
@inline *(a::AbstractProgressMessenger,   b::FunctionOrProgressMessenger) = FunctionMessenger(sim -> a(sim) * space * b(sim))
@inline *(a::FunctionOrProgressMessenger, b::AbstractProgressMessenger)   = FunctionMessenger(sim -> a(sim) * space * b(sim))

const StringOrProgressMessenger = Union{String, AbstractProgressMessenger}
@inline +(a::AbstractProgressMessenger, b::StringOrProgressMessenger) = FunctionMessenger(sim -> a(sim) * comma * b)
@inline +(a::StringOrProgressMessenger, b::AbstractProgressMessenger) = FunctionMessenger(sim -> a      * comma * b(sim))
@inline *(a::AbstractProgressMessenger, b::StringOrProgressMessenger) = FunctionMessenger(sim -> a(sim) * space * b)
@inline *(a::StringOrProgressMessenger, b::AbstractProgressMessenger) = FunctionMessenger(sim -> a      * space * b(sim))
#---

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

#+++ WalltimePerTimestep
Base.@kwdef mutable struct WalltimePerTimestep{T, I} <: AbstractProgressMessenger
    wall_seconds⁻  :: T  # Wall time at previous calback
    iteration⁻     :: I  # Iteration at previous calback
    with_prefix    :: Bool = true
    with_units     :: Bool = true
end

function WalltimePerTimestep(; wall_seconds⁻ = 1e-9*time_ns(),
                               iteration⁻ = 0,
                               with_prefix = true,
                               with_units = true)
    return WalltimePerTimestep(wall_seconds⁻, iteration⁻, with_prefix, with_units)
end

function (wpt::WalltimePerTimestep)(simulation)
    iter = iteration(simulation)

    seconds_since_last_callback = 1e-9 * time_ns() - wpt.wall_seconds⁻
    iterations_since_last_callback = iter == 0 ? Inf : iter - wpt.iteration⁻

    wall_time_per_step = seconds_since_last_callback / iterations_since_last_callback
    wpt.wall_seconds⁻ = 1e-9 * time_ns()
    wpt.iteration⁻ = iter

    message = wpt.with_units ? prettytime(wall_time_per_step) : string(wall_time_per_step)
    wpt.with_prefix && (message = "walltime / timestep = " * message)
    return message
end


Base.@kwdef mutable struct Walltime{T} <: AbstractProgressMessenger
    wall_seconds⁰  :: T  # Wall time at previous calback
    with_prefix    :: Bool = true
    with_units     :: Bool = true
end

function Walltime(; wall_seconds⁰ = 1e-9*time_ns(),
                    with_prefix = true,
                    with_units = true)
    return Walltime(wall_seconds⁰, with_prefix, with_units)
end

function (wt::Walltime)(simulation)
    current_wall_seconds = 1e-9 * time_ns() - wt.wall_seconds⁰
    message = wt.with_units ? prettytime(current_wall_seconds) : string(current_wall_seconds)
    wt.with_prefix && (message = "walltime = " * message)
    return message
end
#---


end # module
