using Revise
using Oceananigans
using Oceanostics
using Oceanostics.ProgressMessengers

grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1));
model = NonhydrostaticModel(; grid);
simulation = Simulation(model, Δt=1, stop_time=10);



mv = MaxVelocities(with_units=true)
@show mv(simulation)


progress_messenger(simulation) = @info (Iteration()
                                        + Time()
                                        + Walltime() + MaxVelocities()
                                        + AdvectiveCFLNumber()
                                        + DiffusiveCFLNumber() 
                                        + MaxViscosity() + WalltimePerTimestep()
                                        )(simulation)
simulation.callbacks[:progress0] = Callback(TimedProgressMessenger())
simulation.callbacks[:progress1] = Callback(SingleLineMessenger())

run!(simulation)
