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
    message = tm.with_units ? prettytime(t) : @sprintf("%.2g", t)
    tm.with_prefix && (message = "time = " * message)
    return_or_print(message, tm)
end
#---

#+++ TimeStep
Base.@kwdef struct TimeStep <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units ::  Bool = true
    print       :: Bool = false
end

@inline function (ts::TimeStep)(simulation)
    Δt = simulation.Δt
    message = ts.with_units ? prettytime(Δt) : @sprintf("%.2g", Δt)
    ts.with_prefix && (message = "Δt = " * message)
    return_or_print(message, ts)
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

#+++ Walltime
Base.@kwdef mutable struct Walltime{T} <: AbstractProgressMessenger
    wall_seconds⁰  :: T    = 1e-9*time_ns() # Wall time at previous calback
    with_prefix    :: Bool = true
    with_units     :: Bool = true
    print          :: Bool = false
end

function (wt::Walltime)(simulation)
    current_wall_seconds = 1e-9 * time_ns() - wt.wall_seconds⁰
    message = wt.with_units ? prettytime(current_wall_seconds) : @sprintf("%.2g", current_wall_seconds)
    wt.with_prefix && (message = "walltime = " * message)
    return_or_print(message, wt)
end
#---

#+++ WalltimePerTimestep
Base.@kwdef mutable struct WalltimePerTimestep{T, I} <: AbstractProgressMessenger
    wall_seconds⁻ :: T    = 1e-9*time_ns() # Wall time at previous calback
    iteration⁻    :: I    = 0 # Iteration at previous calback
    with_prefix   :: Bool = true
    with_units    :: Bool = true
    print         :: Bool = false
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

#+++ BasicTimeMessenger
struct BasicTimeMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    percentage  :: PM
    time        :: PM
    Δt          :: PM
    walltime    :: PM
    print       :: Bool
end

BasicTimeMessenger(; percentage = PercentageProgress(with_prefix = false, with_units = false, print = false),
                      time = Time(with_prefix = true, with_units = true, print = false),
                      Δt = TimeStep(with_prefix = true, with_units = true, print = false),
                      walltime = Walltime(with_prefix = true, with_units = true, print = false),
                      print = true) = BasicTimeMessenger{AbstractProgressMessenger}(percentage, time, Δt, walltime, print)

function (stm::BasicTimeMessenger)(simulation)
    message = ("["*stm.percentage*"] " * stm.time + stm.Δt + stm.walltime)(simulation)
    return_or_print(message, stm)
end
#---

#+++ TimeMessenger
struct TimeMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    iteration             :: PM
    simple_time_messenger :: PM
    print                 :: Bool
end

TimeMessenger(; iteration = Iteration(with_prefix = false, print = false),
                simple_time_messenger = BasicTimeMessenger(print = false),
                print = true) = TimeMessenger{AbstractProgressMessenger}(iteration, simple_time_messenger, print)

function (tm::TimeMessenger)(simulation)
    message = ("iter = " * tm.iteration + tm.simple_time_messenger)(simulation)
    return_or_print(message, tm)
end
#---

#+++ StopwatchMessenger
struct StopwatchMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    time_messenger        :: PM
    walltime_per_timestep :: PM
    print                 :: Bool
end

StopwatchMessenger(; time_messenger = TimeMessenger(print = false),
                     walltime_per_timestep = WalltimePerTimestep(with_prefix = true, with_units = true, print = false),
                     print = true) = StopwatchMessenger{AbstractProgressMessenger}(time_messenger, walltime_per_timestep, print)

function (swm::StopwatchMessenger)(simulation)
    message = (swm.time_messenger + swm.walltime_per_timestep)(simulation)
    return_or_print(message, swm)
end
#---
