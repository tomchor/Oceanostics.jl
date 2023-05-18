using Oceananigans.Diagnostics: AdvectiveCFL, DiffusiveCFL
using Oceananigans.Simulations: TimeStepWizard
using Oceananigans.TurbulenceClosures: viscosity
using Oceananigans.Utils: prettytime
using Oceananigans: iteration, time
using Printf

export SimpleProgressMessenger, SingleLineProgressMessenger, TimedProgressMessenger

tuple_to_op(ν) = ν
tuple_to_op(::Nothing) = nothing
tuple_to_op(ν_tuple::Tuple) = sum(ν_tuple)

"""
    $(SIGNATURES)

Builds the message displayed by the progress messenger.
"""
function make_message(simulation, single_line=false; 
                      ν = viscosity(simulation.model.closure, simulation.model.diffusivity_fields),
                      SI_units = true,
                      initial_wall_time_seconds = 1e-9*time_ns())

    model = simulation.model
    Δt = simulation.Δt
    ν = tuple_to_op(ν)

    iter, t = iteration(simulation), time(simulation)
    progress = 100 * (t / simulation.stop_time)
    current_wall_time = 1e-9 * time_ns() - initial_wall_time_seconds

    u_max = maximum(abs, model.velocities.u)
    v_max = maximum(abs, model.velocities.v)
    w_max = maximum(abs, model.velocities.w)

    # Units
    u_units = SI_units ? " m s⁻¹" : ""
    ν_units = SI_units ? "m² s⁻¹" : ""
    t2str(t) = SI_units ? prettytime(t) : @sprintf("%13.2f", t)

    message = @sprintf("[%06.2f%%] iter: % 6d,     time: % 10s,     Δt: % 10s,     wall time: % 8s",
                      progress, iter, t2str(t), t2str(Δt), prettytime(current_wall_time))

    if !single_line
        message *= @sprintf("\n          └── max(|u⃗|): [%.2e, %.2e, %.2e]%s", u_max, v_max, w_max, u_units)
    end

    message *= @sprintf(",     adv CFL: %.2e", AdvectiveCFL(Δt)(model))

    if !(model.closure isa Tuple) # Oceananigans bug
        message *= @sprintf(",     diff CFL: %.2e", DiffusiveCFL(Δt)(model))
    end

    if !isnothing(ν)
        ν_max = maximum(abs, ν)
        message *= @sprintf(",     νₘₐₓ: %.2e%s", ν_max, ν_units)
    end

    message *= "\n"

    return message
end

print_message(args...; kw...) = @info make_message(args...; kw...)

"""
    $(SIGNATURES)

Return a `SimpleProgressMessenger`. 

`kwargs` are passed to `make_message()`.
"""
SimpleProgressMessenger(; kwargs...) = simulation -> print_message(simulation; kwargs...)

"""
    $(SIGNATURES)

Return a `SingleLineProgressMessenger`, where the messenger output fits in one line.

`kwargs` are passed to `make_message()`.
"""
SingleLineProgressMessenger(; kwargs...) = simulation -> print_message(simulation, true; kwargs...)

mutable struct TimedProgressMessenger{T, I, L} <: Function
    wall_time₀ :: T  # Wall time at simulation start
    wall_time⁻ :: T  # Wall time at previous calback
    iteration⁻ :: I  # Iteration at previous calback
           LES :: L
end

"""
    $(SIGNATURES)

Return a `TimedProgressMessenger`, where the time per model time step is calculated.

`kwargs` are passed to `make_message()`.
"""
TimedProgressMessenger(; LES=false, 
                       wall_time₀=1e-9*time_ns(), 
                       wall_time⁻=1e-9*time_ns(),
                       iteration⁻=0) = TimedProgressMessenger(wall_time₀, wall_time⁻, iteration⁻, LES)

function (pm::TimedProgressMessenger)(simulation)
    model = simulation.model
    Δt = simulation.Δt
    adv_cfl = AdvectiveCFL(simulation.Δt)(model)

    iter, t = iteration(simulation), time(simulation)

    progress = 100 * (t / simulation.stop_time)

    current_wall_time = 1e-9 * time_ns() - pm.wall_time₀
    time_since_last_callback = 1e-9 * time_ns() - pm.wall_time⁻
    iterations_since_last_callback = iter==0 ? Inf : iter - pm.iteration⁻
    wall_time_per_step = time_since_last_callback / iterations_since_last_callback
    pm.wall_time⁻ = 1e-9 * time_ns()
    pm.iteration⁻ = iter

    u_max = maximum(abs, model.velocities.u)
    v_max = maximum(abs, model.velocities.v)
    w_max = maximum(abs, model.velocities.w)

    message = @sprintf("[%06.2f%%] iteration: % 6d, time: % 10s, Δt: % 10s, wall time: % 8s (% 8s / time step)",
                       progress, iter, prettytime(t), prettytime(Δt), prettytime(current_wall_time), prettytime(wall_time_per_step))

    message *= @sprintf("\n          └── max(|u⃗|): [%.2e, %.2e, %.2e] m/s, CFL: %.2e",
                        u_max, v_max, w_max, adv_cfl)

    if pm.LES && !(model.closure isa Tuple)
        ν_max = maximum(abs, viscosity(model.closure, model.diffusivity_fields))
        message *= @sprintf(", νCFL: %.2e, ν_max: %.2e m²/s", DiffusiveCFL(simulation.Δt)(model), ν_max)
    end

    message *= "\n"

    @info message

    return nothing
end
