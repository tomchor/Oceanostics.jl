using Oceananigans.Utils: prettytime
using Oceananigans.Simulations: iteration, time

#+++ Iteration
Base.@kwdef struct Iteration <: AbstractProgressMessenger
    with_prefix :: Bool = true
    print       :: Bool = false
end

@inline function (itm::Iteration)(simulation)
    iter = iteration(simulation)
    message = @sprintf("%6d", iter)
    itm.with_prefix && (message = "iter = " * message)
    return_or_print(message, itm)
end
#---

#+++ Time
Base.@kwdef struct Time <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units ::  Bool = true
    print       :: Bool = false
end

@inline function (tm::Time)(simulation)
    t = time(simulation)
    message = tm.with_units ? prettytime(t) : @sprintf("%.2g", current_wall_seconds)
    tm.with_prefix && (message = "time = " * message)
    return_or_print(message, tm)
end
#---

#+++ PercentageProgress
Base.@kwdef struct PercentageProgress <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

@inline function (pp::PercentageProgress)(simulation)
    percentage_progress_time      = 100 * time(simulation) / simulation.stop_time
    percentage_progress_iteration = 100 * iteration(simulation) / simulation.stop_iteration
    if percentage_progress_time > percentage_progress_iteration
        message = @sprintf("%06.2f%%", percentage_progress_time)
        pp.with_units  && (message = message * " by simulation time")

    else
        message = @sprintf("%06.2f%%", percentage_progress_iteration)
        pp.with_units  && (message = message * " by iteration")
    end
    pp.with_prefix && (message = "progress = " * message)
    return_or_print(message, pp)
end
#---

#+++ WalltimePerTimestep
Base.@kwdef mutable struct WalltimePerTimestep{T, I} <: AbstractProgressMessenger
    wall_seconds⁻ :: T  # Wall time at previous calback
    iteration⁻    :: I  # Iteration at previous calback
    with_prefix   :: Bool = true
    with_units    :: Bool = true
    print         :: Bool = false
end

function WalltimePerTimestep(; wall_seconds⁻ = 1e-9*time_ns(),
                               iteration⁻ = 0,
                               with_prefix = true,
                               with_units = true,
                               print =  false)
    return WalltimePerTimestep(wall_seconds⁻, iteration⁻, with_prefix, with_units, print)
end

function (wpt::WalltimePerTimestep)(simulation)
    iter = iteration(simulation)

    seconds_since_last_callback = 1e-9 * time_ns() - wpt.wall_seconds⁻
    iterations_since_last_callback = iter == 0 ? Inf : iter - wpt.iteration⁻

    wall_time_per_step = seconds_since_last_callback / iterations_since_last_callback
    wpt.wall_seconds⁻ = 1e-9 * time_ns()
    wpt.iteration⁻ = iter

    message = wpt.with_units ? prettytime(wall_time_per_step) : @sprintf("%.2g", wall_time_per_step)
    wpt.with_prefix && (message = "walltime / timestep = " * message)
    return_or_print(message, wpt)
end
#---

#+++ Walltime
Base.@kwdef mutable struct Walltime{T} <: AbstractProgressMessenger
    wall_seconds⁰  :: T  # Wall time at previous calback
    with_prefix    :: Bool = true
    with_units     :: Bool = true
    print          :: Bool = false
end

function Walltime(; wall_seconds⁰ = 1e-9*time_ns(),
                    with_prefix = true,
                    with_units = true,
                    print = false)
    return Walltime(wall_seconds⁰, with_prefix, with_units, print)
end

function (wt::Walltime)(simulation)
    current_wall_seconds = 1e-9 * time_ns() - wt.wall_seconds⁰
    message = wt.with_units ? prettytime(current_wall_seconds) : @sprintf("%.2g", current_wall_seconds)
    wt.with_prefix && (message = "walltime = " * message)
    return_or_print(message, wt)
end
#---
