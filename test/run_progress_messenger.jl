using Oceananigans
using Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
closure = ScalarDiffusivity(ν=1e-6, κ=1e-7)
buoyancy = Buoyancy(model=BuoyancyTracer())
coriolis = FPlane(1e-4)
model = NonhydrostaticModel(; grid, closure, buoyancy, coriolis, tracers=:b)

messenger = SingleLineProgressMessenger(initial_wall_time_seconds=1e-9 * time_ns())
simulation = Simulation(model; Δt=1e-2, stop_iteration=1)
messenger(simulation)
