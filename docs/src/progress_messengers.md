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

## Colored numeric output

Numeric values produced by every messenger are rendered in a distinct terminal
color (light yellow by default) so they stand out from the surrounding labels
and units. The color is controlled by a single process-global `Crayon` and can
be changed at runtime:

```julia
using Oceanostics.ProgressMessengers

set_number_color!(crayon"bold magenta")             # named color, bold
set_number_color!(Crayon(foreground = (255, 128, 0))) # 24-bit RGB
```

The `@crayon_str` string macro is re-exported from `ProgressMessengers`, so the
`crayon"…"` form works without adding [Crayons.jl](https://github.com/KristofferC/Crayons.jl)
as a direct dependency. Any messenger — pre-built or one you write yourself —
picks up the new color on the next call.

To turn coloring off, assign a no-op crayon:

```julia
set_number_color!(Crayon())
```

Notes:
- `NUMBER_CRAYON` is process-global; concurrent simulations in the same Julia
  process share one color.
- Output captured to non-TTY destinations (log files, some CI consoles) will
  contain raw ANSI escape sequences. Set a no-op crayon as above to keep logs
  plain.

## Pre-built messengers

| Type                  | Description |
|:----------------------|:------------|
| `BasicMessenger`      | Percentage progress, simulation time, time step, wall time, advective CFL, diffusive CFL |
| `SingleLineMessenger` | Iteration count, plus everything in `BasicMessenger` |
| `TimedMessenger`      | Per-step wall time, max velocities, plus everything in `SingleLineMessenger` |

## Timing components

| Type                  | Description |
|:----------------------|:------------|
| `Iteration`           | Current iteration number |
| `SimulationTime`      | Current simulation time |
| `TimeStep`            | Current time step `Δt` |
| `PercentageProgress`  | Progress as a percentage (by time or iteration) |
| `Walltime`            | Elapsed wall-clock time since the start |
| `StepDuration`        | Wall-clock time per time step |
| `BasicTimeMessenger`  | Combines `PercentageProgress`, `SimulationTime`, `TimeStep`, and `Walltime` |
| `TimeMessenger`       | Adds `Iteration` to `BasicTimeMessenger` |
| `StopwatchMessenger`  | Adds `StepDuration` to `TimeMessenger` |

## Velocity components

| Type             | Description |
|:-----------------|:------------|
| `MaxUVelocity`   | Maximum absolute `u` velocity |
| `MaxVVelocity`   | Maximum absolute `v` velocity |
| `MaxWVelocity`   | Maximum absolute `w` velocity |
| `MaxVelocities`  | All three components formatted as a vector |

## Stability components

| Type                     | Description |
|:-------------------------|:------------|
| `AdvectiveCFLNumber`     | Advective CFL number (aliased as `CourantNumber`) |
| `DiffusiveCFLNumber`     | Diffusive CFL number (aliased as `NormalizedMaxViscosity`) |
| `MaxViscosity`           | Maximum viscosity |
| `BasicStabilityMessenger`| Combines advective and diffusive CFL |
| `StabilityMessenger`     | Adds `MaxViscosity` to `BasicStabilityMessenger` |
