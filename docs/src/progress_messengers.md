# Progress messengers

The `ProgressMessengers` submodule provides composable simulation progress reporters
for use as Oceananigans `Callback`s. During long-running ocean simulations, it is
important to monitor quantities like the current time step, wall-clock time per
iteration, maximum velocities, and CFL numbers. This module supplies individual
reporter components for each of these quantities, along with pre-built combinations
that cover common use cases.

The key design idea is composability: messengers are combined with two operators:

- **`+`** concatenates messages separated by a comma (`, `), suitable for grouping
  independent quantities on one line.
- **`*`** concatenates messages with no separator, useful for building prefixed or
  bracketed expressions.

This lets users build custom progress output from atomic components without writing
formatting boilerplate. For example, `Iteration() + SimulationTime() + TimeStep()`
produces output like `iter =      5, time = 5 seconds, Δt = 1 second`.

The module provides three levels of pre-built messengers with increasing detail:

- **`BasicMessenger`**: percentage, simulation time, time step, wall time, advective
  CFL, and diffusive CFL.
- **`SingleLineMessenger`**: adds iteration count and wall time to the basic output.
- **`TimedMessenger`**: the most detailed, adding per-step wall time and maximum
  velocity components.

## Example

```jldoctest pm_example
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; closure=SmagorinskyLilly());

julia> simulation = Simulation(model, Δt=1, stop_time=10);

julia> progress = ProgressMessengers.Iteration() + ProgressMessengers.SimulationTime() + ProgressMessengers.AdvectiveCFLNumber();

julia> simulation.callbacks[:progress] = Callback(progress, IterationInterval(5));

julia> typeof(progress) <: Oceanostics.ProgressMessengers.AbstractProgressMessenger
true
```

## Pre-built messengers

```@docs
Oceanostics.ProgressMessengers.BasicMessenger
Oceanostics.ProgressMessengers.SingleLineMessenger
Oceanostics.ProgressMessengers.TimedMessenger
```

## Timing components

```@docs
Oceanostics.ProgressMessengers.Iteration
Oceanostics.ProgressMessengers.SimulationTime
Oceanostics.ProgressMessengers.TimeStep
Oceanostics.ProgressMessengers.PercentageProgress
Oceanostics.ProgressMessengers.Walltime
Oceanostics.ProgressMessengers.StepDuration
Oceanostics.ProgressMessengers.BasicTimeMessenger
Oceanostics.ProgressMessengers.TimeMessenger
Oceanostics.ProgressMessengers.StopwatchMessenger
```

## Velocity components

```@docs
Oceanostics.ProgressMessengers.MaxUVelocity
Oceanostics.ProgressMessengers.MaxVVelocity
Oceanostics.ProgressMessengers.MaxWVelocity
Oceanostics.ProgressMessengers.MaxVelocities
```

## Stability components

```@docs
Oceanostics.ProgressMessengers.AdvectiveCFLNumber
Oceanostics.ProgressMessengers.DiffusiveCFLNumber
Oceanostics.ProgressMessengers.MaxViscosity
Oceanostics.ProgressMessengers.BasicStabilityMessenger
Oceanostics.ProgressMessengers.StabilityMessenger
```
