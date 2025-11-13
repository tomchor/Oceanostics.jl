import Oceananigans.Diagnostics
using Oceananigans.TurbulenceClosures: viscosity, diffusivity

#+++ AdvectiveCFLNumber
Base.@kwdef struct AdvectiveCFLNumber <: AbstractProgressMessenger
    with_prefix :: Bool = true
    print       :: Bool = false
end

@inline function (acfl::AdvectiveCFLNumber)(simulation)
    cfl = Diagnostics.AdvectiveCFL(simulation.Δt)(simulation.model)
    message = @sprintf("%.2g", cfl)
    acfl.with_prefix && (message = "advective CFL = " * message)
    return_or_print(message, acfl)
end
#---

#+++ DiffusiveCFLNumber
Base.@kwdef struct DiffusiveCFLNumber <: AbstractProgressMessenger
    with_prefix :: Bool = true
    print       :: Bool = false
end

@inline function (dcfl::DiffusiveCFLNumber)(simulation)
    cfl = Diagnostics.DiffusiveCFL(simulation.Δt)(simulation.model)
    message = @sprintf("%.2g", cfl)
    dcfl.with_prefix && (message = "diffusive CFL = " * message)
    return_or_print(message, dcfl)
end
#---

#+++ MaxViscosity
Base.@kwdef struct MaxViscosity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

tuple_to_op(ν) = ν
tuple_to_op(::Nothing) = nothing
tuple_to_op(ν_tuple::Tuple) = sum(ν_tuple)

@inline function (maxν::MaxViscosity)(simulation)
    ν = tuple_to_op(viscosity(simulation.model.closure, simulation.model.closure_fields))
    ν_max = maximum(abs, ν)
    message = @sprintf("%.2g", ν_max)
    maxν.with_prefix && (message = "νₘₐₓ = " * message)
    maxν.with_units  && (message = message * " m²/s")
    return_or_print(message, maxν)
end
#---

#+++ BasicStabilityMessenger
struct BasicStabilityMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    advective_cfl :: PM
    diffusive_cfl :: PM
    print         :: Bool
end

BasicStabilityMessenger(; advective_cfl = AdvectiveCFLNumber(with_prefix = true, print = false),
                           diffusive_cfl = DiffusiveCFLNumber(with_prefix = true, print = false),
                           print = true) = BasicStabilityMessenger{AbstractProgressMessenger}(advective_cfl, diffusive_cfl, print)

function (bsm::BasicStabilityMessenger)(simulation)
    message = (bsm.advective_cfl + bsm.diffusive_cfl)(simulation)
    return_or_print(message, bsm)
end
#---

#+++ StabilityMessenger
struct StabilityMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    basic_stab_messenger :: PM
    max_viscosity        :: PM
    print                :: Bool
end

StabilityMessenger(; basic_stab_messenger = BasicStabilityMessenger(print = false),
                     max_viscosity = MaxViscosity(with_prefix = true, with_units = true, print = false),
                     print = true) = StabilityMessenger{AbstractProgressMessenger}(basic_stab_messenger, max_viscosity, print)

function (sm::StabilityMessenger)(simulation)
    message = (sm.basic_stab_messenger + sm.max_viscosity)(simulation)
    return_or_print(message, sm)
end
#---
