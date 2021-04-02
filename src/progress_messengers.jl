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

