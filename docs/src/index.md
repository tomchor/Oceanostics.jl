```@eval
using Markdown
Markdown.parse_file(joinpath(@__DIR__, "..", "..", "README.md"))
```

## Quick example

The example below illustrates a few of Oceanostics' features. Check the Examples for more detailed
usage.

```jldoctest; filter = r"┌ Info:.*"s
julia> using Oceananigans

julia> using Oceanostics

julia> grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid, closure=SmagorinskyLilly());

julia> simulation = Simulation(model, Δt=1, stop_time=10);

julia> simulation.callbacks[:progress] = Callback(ProgressMessengers.TimedMessenger(), IterationInterval(5));

julia> ke = KineticEnergyEquation.KineticEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×5×6 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")

julia> ε = KineticEnergyEquation.KineticEnergyDissipationRate(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×5×6 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: viscous_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("NamedTuple", "NamedTuple", "NamedTuple")

julia> run!(simulation)
[ Info: Initializing simulation...
┌ Info: iter =      0,  [000.00%] time = 0 seconds,  Δt = 1 second,  walltime = 621.022 ms,  walltime / timestep = 0 seconds
└           |u⃗|ₘₐₓ = [0.00e+00,  0.00e+00,  0.00e+00] m/s,  advective CFL = 0,  diffusive CFL = 0,  νₘₐₓ = 0 m²/s
[ Info:     ... simulation initialization complete (8.970 seconds)
[ Info: Executing initial time step...
[ Info:     ... initial time step complete (3.415 ms).
┌ Info: iter =      5,  [050.00%] time = 5 seconds,  Δt = 1 second,  walltime = 9.035 seconds,  walltime / timestep = 1.683 seconds
└           |u⃗|ₘₐₓ = [0.00e+00,  0.00e+00,  0.00e+00] m/s,  advective CFL = 0,  diffusive CFL = 0,  νₘₐₓ = 0 m²/s
[ Info: Simulation is stopping after running for 9.030 seconds.
[ Info: Simulation time 10 seconds equals or exceeds stop time 10 seconds.
┌ Info: iter =     10,  [100.00%] time = 10 seconds,  Δt = 1 second,  walltime = 9.052 seconds,  walltime / timestep = 3.340 ms
└           |u⃗|ₘₐₓ = [0.00e+00,  0.00e+00,  0.00e+00] m/s,  advective CFL = 0,  diffusive CFL = 0,  νₘₐₓ = 0 m²/s
```

