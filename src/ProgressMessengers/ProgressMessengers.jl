module ProgressMessengers
using DocStringExtensions

using Printf

import Base: +, *

export AbstractProgressMessenger
export FunctionMessenger
export MaxUVelocity, MaxVVelocity, MaxWVelocity
export MaxVelocities
export Iteration, Time, TimeStep, PercentageProgress, WalltimePerTimestep, Walltime, BasicTimeMessenger, TimeMessenger, StopwatchMessenger
export MaxViscosity, AdvectiveCFLNumber, DiffusiveCFLNumber, BasicStabilityMessenger
export BasicMessenger, SingleLineMessenger, TimedMessenger

abstract type AbstractProgressMessenger end

const comma = ", "
const space = ""

#+++ FunctionMessenger
Base.@kwdef struct FunctionMessenger{F} <: AbstractProgressMessenger
    func :: F
end

function (fmessenger::FunctionMessenger)(sim)
    message = fmessenger.func(sim)
    return_or_print(message)
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

return_or_print(message, pm::AbstractProgressMessenger) = pm.print ? (@info message) : (return message)
return_or_print(message) = return message

include("velocities.jl")
include("timing.jl")
include("cfl.jl")

const CourantNumber = AdvectiveCFLNumber
const NormalizedMaxViscosity = DiffusiveCFLNumber

#+++ BasicMessenger
struct BasicMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    basic_time_messenger      :: PM
    maxvels                    :: PM
    basic_stability_messenger :: PM
    print                      :: Bool
end

BasicMessenger(; basic_time_messenger = BasicTimeMessenger(print = false),
                  maxvels = MaxVelocities(with_prefix = true, with_units = true, print = false),
                  basic_stability_messenger = BasicStabilityMessenger(print = false),
                  print = true) = BasicMessenger{AbstractProgressMessenger}(basic_time_messenger, maxvels, basic_stability_messenger, print)

function (sm::BasicMessenger)(simulation)
    message = (sm.basic_time_messenger + sm.maxvels + sm.basic_stability_messenger)(simulation)
    return_or_print(message, sm)
end
#---

#+++ SingleLineMessenger
struct SingleLineMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    time_messenger            :: PM
    maxvels                   :: PM
    basic_stability_messenger :: PM
    print                     :: Bool
end

SingleLineMessenger(; time_messenger = TimeMessenger(print = false),
                      maxvels = MaxVelocities(with_prefix = true, with_units = true, print = false),
                      basic_stability_messenger = BasicStabilityMessenger(print = false),
                      print = true) = SingleLineMessenger{AbstractProgressMessenger}(time_messenger, maxvels, basic_stability_messenger, print)

function (sm::SingleLineMessenger)(simulation)
    message = (sm.time_messenger + sm.maxvels + sm.basic_stability_messenger)(simulation)
    return_or_print(message, sm)
end
#---

#+++ TimedMessenger
struct TimedMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    basic_time_messenger      :: PM
    maxvels                    :: PM
    basic_stability_messenger :: PM
    print                      :: Bool
end

TimedMessenger(; basic_time_messenger = BasicTimeMessenger(print = false),
                  maxvels = MaxVelocities(with_prefix = true, with_units = true, print = false),
                  basic_stability_messenger = BasicStabilityMessenger(print = false),
                  print = true) = TimedMessenger{AbstractProgressMessenger}(basic_time_messenger, maxvels, basic_stability_messenger, print)

function (sm::TimedMessenger)(simulation)
    message = (sm.basic_time_messenger + sm.maxvels + sm.basic_stability_messenger)(simulation)
    return_or_print(message, sm)
end
#---

end # module
