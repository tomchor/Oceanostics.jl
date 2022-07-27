using Oceananigans
using Oceanostics

grid = RectilinearGrid(size=(40,40,40), extent=(1,1,1))
model = NonhydrostaticModel(grid=grid)

simulation = Simulation(model, Δt=1/10, stop_time=50)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=10)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))
simulation.callbacks[:progress] = Callback(Oceanostics.TimedProgressMessenger(), TimeInterval(2))
run!(simulation)
