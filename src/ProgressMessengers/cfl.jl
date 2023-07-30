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
    acfl.with_prefix && (message = "Advective CFL = " * message)
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
    dcfl.with_prefix && (message = "Diffusive CFL = " * message)
    return_or_print(message, dcfl)
end
#---

#+++ MaxViscosity
Base.@kwdef struct MaxViscosity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

@inline function (maxν::MaxViscosity)(simulation)
    ν_max = maximum(abs, viscosity(simulation.model.closure, simulation.model.diffusivity_fields))
    message = @sprintf("%.2g", ν_max)
    maxν.with_prefix && (message = "νₘₐₓ = " * message)
    maxν.with_units  && (message = message * " m²/s")
    return_or_print(message, maxν)
end
#---
