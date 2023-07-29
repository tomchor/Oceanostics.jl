using Revise
using Oceananigans
using Oceanostics
using Oceanostics.ProgressMessengers

grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1));
model = NonhydrostaticModel(; grid);
simulation = Simulation(model, Δt=1, stop_time=10);

max_u = MaxUVelocity();
max_v = MaxVVelocity();
max_w = MaxWVelocity();

@show max_u(simulation) max_v(simulation) max_w(simulation)

max_u = MaxUVelocity(with_prefix=false, with_units=false);
max_v = MaxVVelocity(with_prefix=false, with_units=false);
max_w = MaxWVelocity(with_prefix=false, with_units=false);


max_vels = Oceanostics.ProgressMessengers.MaxVelocities()

@show max_vels(simulation)

st2 = WalltimePerTimestep()
@show st2(simulation)

time_step!(model, 1)
@show st2(simulation)

wt = Walltime()
@show wt(simulation)
time_step!(model, 1)
@show wt(simulation)
time_step!(model, 1)
@show wt(simulation)

acfl = AdvectiveCFLNumber()
@show acfl(simulation)

dcfl = DiffusiveCFLNumber()
@show dcfl(simulation)

mν = MaxViscosity()
@show mν(simulation)

