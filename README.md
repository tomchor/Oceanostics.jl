# Oceanostics

Useful diagnostics to use with [Oceananigans](https://github.com/CliMA/Oceananigans.jl). Mostly `AbstractOperations`s and a few useful progress messengers.

To add the latest registered version from Julia:
```julia
julia>]
(v1.6) pkg> add Oceanostics
```

If you want the latest developments (which may or may not be unstable) you can add the latest version from github in the `main` branch:

```julia
julia> using Pkg

julia> Pkg.add(url="https://github.com/tomchor/Oceanostics.jl.git", rev="main")
```
The keyword `rev` let's you pick which github branch you want.


## Simple example

The example below is a simple illustration of the use of `TimedProgressMessenger` (which keeps track
of how long each time step takes) and `KineticEnergy`, which computed the kinetic energy of a flow.

(Note that `(; tke, ε)` is a shorthand for `(tke=tke, ε=ε)`.)

```julia
using Oceananigans
using Oceanostics

grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, closure=SmagorinskyLilly())
simulation = Simulation(model, Δt=1, stop_iteration=10, progress=Oceanostics.TimedProgressMessenger(; LES=false))

ke = Field(KineticEnergy(model))
ε = Field(IsotropicViscousDissipationRate(model))
simulation.output_writers[:netcdf_writer] = NetCDFOutputWriter(model, (; ke, ε), filepath="out.nc", schedule=TimeInterval(2))
run!(simulation)
```

## Caveats

- Not every diagnostic has been thoroughly tested (we're still working on testing everything with CI).
- Most diagnostics are written very generally since most uses of averages, etc. do not assume any
  specific kind of averaging procedure. Chances are it "wastes" computations for a given specific application.


<!-- ## Notes on notation and usage

For now I'm assuming that lowercase variables are pertubations around a mean and uppercase
variables are the mean (any kind of mean or even background fields). So, for example,
kinetic energy is calculated as (the following is a pseudo-code):

```julia
ke(u, v, w) = 1/2*(u^2 + v^2 + w^2)
```

And it is up to the user to make sure that the function is called with the perturbations
(to actually get turbulent kinetic energy), or the full velocity fields if the desired
output is total kinetic energy. So for turbulent kinetic energy one might call the
function as

```julia
U = Field(Average(model.velocities.u, dims=(1, 2)))
V = Field(Average(model.velocities.v, dims=(1, 2)))
TKE = ke(model.velocities.u-U, model.velocities.v-V, model.velocities.w)
```
-->
