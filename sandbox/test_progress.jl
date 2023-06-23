using Revise
using Oceananigans
using Oceanostics
using Printf

grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid)
simulation = Simulation(model, Δt=1, stop_time=10);


@show max_u = MaxUVelocity(with_units=true)
@show max_v = MaxVVelocity(with_units=true)
@show max_w = MaxWVelocity(with_units=true)

dpause
mv = MaximumVelocities(formatted=true)
pm = mv
mv(simulation)

