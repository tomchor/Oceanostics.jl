# Progress messengers

The `ProgressMessengers` submodule provides composable simulation progress reporters
for use as Oceananigans `Callback`s.

Messengers are combined with `+` (comma-separated output) and `*` (concatenated output),
making it easy to build custom progress messages from individual components.

## Quick start

```julia
using Oceananigans, Oceanostics

simulation.callbacks[:progress] = Callback(ProgressMessengers.SingleLineMessenger(), IterationInterval(10))
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
