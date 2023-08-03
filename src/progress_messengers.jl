using Oceananigans.Diagnostics: AdvectiveCFL, DiffusiveCFL
using Oceananigans.Simulations: TimeStepWizard
using Oceananigans.TurbulenceClosures: viscosity
using Oceananigans.Utils: prettytime
using Oceananigans: iteration, time
using Printf

export TimedProgressMessenger

mutable struct TimedProgressMessenger{T, I, L} <: Function
    wall_time₀ :: T  # Wall time at simulation start
    wall_time⁻ :: T  # Wall time at previous calback
    iteration⁻ :: I  # Iteration at previous calback
           LES :: L
end

"""
    $(SIGNATURES)

Return a `TimedProgressMessenger`, where the time per model time step is calculated.
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
