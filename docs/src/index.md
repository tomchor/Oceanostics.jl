```@eval
using Markdown
Markdown.parse_file(joinpath(@__DIR__, "..", "..", "README.md"))
```

!!! note "This documentation is still under construction"
    We are still working on these docs. If you see any errors or if you have any helpful suggestions please 
    open [an issue](https://github.com/tomchor/Oceanostics.jl/issues/new) or
    [a pull request](https://github.com/tomchor/Oceanostics.jl/pulls) on github.


## Quick example

The example below is a simple illustration of how to use a few of Oceanostics features:

```jldoctest
julia> using Oceananigans

julia> using Oceanostics

julia> grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid=grid, closure=SmagorinskyLilly());

julia> simulation = Simulation(model, Δt=1, stop_time=20);

julia> simulation.callbacks[:progress] = Callback(TimedProgressMessenger(LES=false), IterationInterval(5));

julia> ke = KineticEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×5×6 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: turbulent_kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("4×5×6 Field{Face, Center, Center} on RectilinearGrid on CPU", "4×5×6 Field{Center, Face, Center} on RectilinearGrid on CPU", "4×5×7 Field{Center, Center, Face} on RectilinearGrid on CPU", "0", "0", "0")

julia> ε = KineticEnergyDissipationRate(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×5×6 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: viscous_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("(νₑ=4×5×6 Field{Center, Center, Center} on RectilinearGrid on CPU,)", "(u=4×5×6 Field{Face, Center, Center} on RectilinearGrid on CPU, v=4×5×6 Field{Center, Face, Center} on RectilinearGrid on CPU, w=4×5×7 Field{Center, Center, Face} on RectilinearGrid on CPU)", "(closure=SmagorinskyLilly: C=0.16, Cb=1.0, Pr=NamedTuple(), clock=Clock(time=0 seconds, iteration=0), buoyancy=Nothing)")

julia> simulation.output_writers[:netcdf_writer] = NetCDFOutputWriter(model, (; ke, ε), filename="out.nc", schedule=TimeInterval(2));

julia> run!(simulation)
[ Info: Initializing simulation...
┌ Info: [000.00%] iteration:      0, time:  0 seconds, Δt:   1 second, wall time: 7.418 seconds (0 seconds / time step)
└           └── max(|u⃗|): [0.00e+00, 0.00e+00, 0.00e+00] m/s, CFL: 0.00e+00
[ Info:     ... simulation initialization complete (1.300 seconds)
[ Info: Executing initial time step...
[ Info:     ... initial time step complete (1.525 ms).
┌ Info: [025.00%] iteration:      5, time:  5 seconds, Δt:   1 second, wall time: 8.689 seconds (254.179 ms / time step)
└           └── max(|u⃗|): [0.00e+00, 0.00e+00, 0.00e+00] m/s, CFL: 0.00e+00
┌ Info: [050.00%] iteration:     10, time: 10 seconds, Δt:   1 second, wall time: 8.699 seconds (2.101 ms / time step)
└           └── max(|u⃗|): [0.00e+00, 0.00e+00, 0.00e+00] m/s, CFL: 0.00e+00
┌ Info: [075.00%] iteration:     15, time: 15 seconds, Δt:   1 second, wall time: 8.714 seconds (2.981 ms / time step)
└           └── max(|u⃗|): [0.00e+00, 0.00e+00, 0.00e+00] m/s, CFL: 0.00e+00
[ Info: Simulation is stopping after running for 1.346 seconds.
[ Info: Simulation time 20 seconds equals or exceeds stop time 20 seconds.
┌ Info: [100.00%] iteration:     20, time: 20 seconds, Δt:   1 second, wall time: 8.725 seconds (2.222 ms / time step)
└           └── max(|u⃗|): [0.00e+00, 0.00e+00, 0.00e+00] m/s, CFL: 0.00e+00
```

