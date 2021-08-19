using Oceananigans.Diagnostics: AdvectiveCFL, DiffusiveCFL
using Oceananigans.Simulations: TimeStepWizard
using Oceananigans.Utils: prettytime
using Printf

get_Δt(Δt) = Δt
get_Δt(wizard::TimeStepWizard) = wizard.Δt


function SimpleProgressMessenger_function(simulation; LES=false, SI_units=true,
                                           initial_wall_time_seconds=1e-9*time_ns())
    model = simulation.model
    Δt = simulation.Δt

    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (t / simulation.stop_time)

    current_wall_time = 1e-9 * time_ns() - initial_wall_time_seconds

    u_max = maximum(abs, model.velocities.u)
    v_max = maximum(abs, model.velocities.v)
    w_max = maximum(abs, model.velocities.w)

    if SI_units
        @info @sprintf("[%06.2f%%] i: % 6d,     time: % 10s,     Δt: % 10s,     wall time: % 8s",
                        progress, i, prettytime(t), prettytime(get_Δt(Δt)), prettytime(current_wall_time),)
        if LES
            ν_max = maximum(abs, model.diffusivities.νₑ)
            @info @sprintf("          └── max(|u⃗|): [%.2e, %.2e, %.2e] m/s,     adv CFL: %.2e,     diff CFL: %.2e,     νₘₐₓ: %.2e m²/s",
                           u_max, v_max, w_max, AdvectiveCFL(Δt)(model), DiffusiveCFL(Δt)(model), ν_max)
        else
            @info @sprintf("          └── max(|u⃗|): [%.2e, %.2e, %.2e] m/s,     adv CFL: %.2e,     diff CFL: %.2e",
                           u_max, v_max, w_max, AdvectiveCFL(Δt)(model), DiffusiveCFL(Δt)(model))
        end

    else
        @info @sprintf("[%06.2f%%] i: % 6d,     time: %13.2f,     Δt: %13.2f,     wall time: % 8s",
                       progress, i, t, get_Δt(Δt), prettytime(current_wall_time))
        if LES
            ν_max = maximum(abs, model.diffusivities.νₑ)
            @info @sprintf("          └── max(|u⃗|): [%.2e, %.2e, %.2e],     adv CFL: %.2e,     diff CFL: %.2e,     νₘₐₓ: %.2e",
                           u_max, v_max, w_max, AdvectiveCFL(Δt)(model), DiffusiveCFL(Δt)(model), ν_max)
        else
            @info @sprintf("          └── max(|u⃗|): [%.2e, %.2e, %.2e],     adv CFL: %.2e,     diff CFL: %.2e",
                           u_max, v_max, w_max, AdvectiveCFL(Δt)(model), DiffusiveCFL(Δt)(model))
        end
    end

    @info ""

    return nothing
end

function SimpleProgressMessenger(; kwargs...)
    return simulation -> SimpleProgressMessenger_function(simulation; kwargs...)
end






function SingleLineProgressMessenger_func(simulation; LES=false, SI_units=true,
                                          initial_wall_time_seconds=1e-9*time_ns())
    model = simulation.model
    Δt = simulation.Δt

    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (t / simulation.stop_time)

    current_wall_time = 1e-9 * time_ns() - initial_wall_time_seconds

    if SI_units
        if LES
            ν_max = maximum(abs, model.diffusivities.νₑ)
            @info @sprintf("[%06.2f%%] i: % 6d,     time: %10s,     Δt: %10s,     wall time: %8s,     adv CFL: %.2e,     diff CFL: %.2e,     νₘₐₓ: %.2e m²/s",
                           progress, i, prettytime(t), prettytime(get_Δt(Δt)), prettytime(current_wall_time),
                           AdvectiveCFL(Δt)(model), DiffusiveCFL(Δt)(model), ν_max)
        else
            @info @sprintf("[%06.2f%%] i: % 6d,     time: %10s,     Δt: %10s,     wall time: %8s,     adv CFL: %.2e,     diff CFL: %.2e",
                           progress, i, prettytime(t), prettytime(get_Δt(Δt)), prettytime(current_wall_time),
                           AdvectiveCFL(Δt)(model), DiffusiveCFL(Δt)(model))
        end

    else
        if LES
            ν_max = maximum(abs, model.diffusivities.νₑ)
            @info @sprintf("[%06.2f%%] i: % 6d,     time: %13.2f,     Δt: %13.2f,     wall time: % 8s,     adv CFL: %.2e,     diff CFL: %.2e,     νₘₐₓ: %.2e",
                           progress, i, t, get_Δt(Δt), prettytime(current_wall_time), AdvectiveCFL(Δt)(model), DiffusiveCFL(Δt)(model), ν_max)
        else
            @info @sprintf("[%06.2f%%] i: % 6d,     time: %13.2f,     Δt: %13.2f,     wall time: % 8s,     adv CFL: %.2e,     diff CFL: %.2e",
                           progress, i, t, get_Δt(Δt), prettytime(current_wall_time), AdvectiveCFL(Δt)(model), DiffusiveCFL(Δt)(model))
        end
    end

    return nothing
end

function SingleLineProgressMessenger(; kwargs...)
    return simulation -> SingleLineProgressMessenger_func(simulation; kwargs...)
end







mutable struct TimedProgressMessenger{T, A, D, L, Δ} <: Function
    wall_time₀ :: T  # Wall time at simulation start
    wall_time⁻ :: T  # Wall time at previous calback
       adv_cfl :: A
       dif_cfl :: D
           LES :: L
            Δt :: Δ
end

TimedProgressMessenger(Δt) =
    TimedProgressMessenger(
                      1e-9 * time_ns(),
                      1e-9 * time_ns(),
                      AdvectiveCFL(Δt),
                      DiffusiveCFL(Δt),
                      false,
                      Δt)

function TimedProgressMessenger(; Δt=0, LES=false, 
                       wall_time₀=1e-9*time_ns(), wall_time⁻=1e-9*time_ns(),
                       adv_cfl=nothing, dif_cfl=nothing)

    adv_cfl = adv_cfl==nothing ? AdvectiveCFL(Δt) : adv_cfl
    dif_cfl = dif_cfl==nothing ? DiffusiveCFL(Δt) : dif_cfl
    return TimedProgressMessenger(wall_time₀, wall_time⁻, adv_cfl, dif_cfl, LES, Δt)
end


function (pm::TimedProgressMessenger)(simulation)
    model = simulation.model

    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (t / simulation.stop_time)

    current_wall_time = 1e-9 * time_ns() - pm.wall_time₀
    time_since_last_callback = 1e-9 * time_ns() - pm.wall_time⁻
    wall_time_per_step = time_since_last_callback / simulation.iteration_interval
    pm.wall_time⁻ = 1e-9 * time_ns()

    u_max = maximum(abs, model.velocities.u)
    v_max = maximum(abs, model.velocities.v)
    w_max = maximum(abs, model.velocities.w)

    @info @sprintf("[%06.2f%%] iteration: % 6d, time: % 10s, Δt: % 10s, wall time: % 8s (% 8s / time step)",
                    progress, i, prettytime(t), prettytime(get_Δt(pm.Δt)), prettytime(current_wall_time), prettytime(wall_time_per_step))

    if pm.LES
        ν_max = maximum(abs, model.diffusivity_fields.νₑ)
        @info @sprintf("          └── max(|u⃗|): [%.2e, %.2e, %.2e] m/s, CFL: %.2e, νCFL: %.2e, ν_max: %.2e m²/s",
                        u_max, v_max, w_max, pm.adv_cfl(model), pm.dif_cfl(model), ν_max)
    else
        @info @sprintf("          └── max(|u⃗|): [%.2e, %.2e, %.2e] m/s, CFL: %.2e, νCFL: %.2e",
                        u_max, v_max, w_max, pm.adv_cfl(model), pm.dif_cfl(model))
    end

    @info ""

    return nothing
end
