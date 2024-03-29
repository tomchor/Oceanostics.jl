module ProgressMessengers
using DocStringExtensions

using Printf

import Base: +, *

export AbstractProgressMessenger
export FunctionMessenger
export MaxUVelocity, MaxVVelocity, MaxWVelocity
export MaxVelocities
export Iteration, SimulationTime, TimeStep, PercentageProgress, Walltime, StepDuration, BasicTimeMessenger, TimeMessenger, StopwatchMessenger
export MaxViscosity, AdvectiveCFLNumber, DiffusiveCFLNumber, BasicStabilityMessenger, StabilityMessenger
export CourantNumber, NormalizedMaxViscosity
export BasicMessenger, SingleLineMessenger, TimedMessenger

abstract type AbstractProgressMessenger end

comma = ",  "
space = ""
indented_newline = "\n          "

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
    basic_stability_messenger :: PM
    print                     :: Bool
end

BasicMessenger(; basic_time_messenger = BasicTimeMessenger(print = false),
                 basic_stability_messenger = BasicStabilityMessenger(print = false),
                 print = true) = BasicMessenger{AbstractProgressMessenger}(basic_time_messenger, basic_stability_messenger, print)

function (pm::BasicMessenger)(simulation)
    message = (pm.basic_time_messenger + pm.basic_stability_messenger)(simulation)
    return_or_print(message, pm)
end
#---

#+++ SingleLineMessenger
struct SingleLineMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    time_messenger      :: PM
    stability_messenger :: PM
    print               :: Bool
end

SingleLineMessenger(; time_messenger = TimeMessenger(print = false),
                      stability_messenger = StabilityMessenger(print = false),
                      print = true) = SingleLineMessenger{AbstractProgressMessenger}(time_messenger, stability_messenger, print)

function (pm::SingleLineMessenger)(simulation)
    message = (pm.time_messenger + pm.stability_messenger)(simulation)
    return_or_print(message, pm)
end
#---

#+++ TimedMessenger
struct TimedMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    stopwatch_messenger :: PM
    maxvels             :: PM
    stability_messenger :: PM
    print               :: Bool
end

TimedMessenger(; stopwatch_messenger = StopwatchMessenger(print = false),
                 maxvels = MaxVelocities(with_prefix = true, with_units = true, print = false),
                 stability_messenger = StabilityMessenger(print = false),
                 print = true) = TimedMessenger{AbstractProgressMessenger}(stopwatch_messenger, maxvels, stability_messenger, print)

function (pm::TimedMessenger)(simulation)
    message = (pm.stopwatch_messenger * indented_newline * pm.maxvels + pm.stability_messenger)(simulation)
    return_or_print(message, pm)
end
#---

end # module
