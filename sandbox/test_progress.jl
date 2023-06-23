using Revise
using Oceananigans
using Oceanostics
using Printf

grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid)
simulation = Simulation(model, Δt=1, stop_time=10);


using Oceanostics.ProgressMessengers

@show mu = MaxUVelocity(with_units=true)
@show mv = MaxVVelocity(with_units=true)
@show mw = MaxWVelocity(with_units=true)

mv = MaximumVelocities()


pm = mv
pm(simulation)

