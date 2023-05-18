# Oceanostics

Latest Documentation: [![Build Status](https://img.shields.io/badge/documentation-in%20development-orange)](https://tomchor.github.io/Oceanostics.jl/dev/)



Oceanostics is a Julia package created to facilitate getting diagnostics quantities in
[Oceananigans](https://github.com/CliMA/Oceananigans.jl) simulations. It was created solely as a
companion package to [Oceananigans](https://github.com/CliMA/Oceananigans.jl), so we refer users
first to the [Oceananigans documentation](https://clima.github.io/OceananigansDocumentation/stable/)
before getting started with Oceanostics.


## Installation

To install the latest registered tagged version from Julia:
```julia
julia> ]
(@v1.8) pkg> add Oceanostics
```

If you want the latest developments (which may or may not be unstable) you can add the latest version from github in the `main` branch:

```julia
julia> using Pkg

julia> Pkg.add(url="https://github.com/tomchor/Oceanostics.jl.git", rev="main")
```
The keyword `rev` let's you pick which github branch you want.


## Simple example

The example below is a simple illustration of how to use a few of Oceanostics features:

```julia
using Oceananigans
using Oceanostics

grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, closure=SmagorinskyLilly())
simulation = Simulation(model, Δt=1, stop_time=20)
simulation.callbacks[:progress] = Callback(TimedProgressMessenger(LES=false), IterationInterval(5))

ke = Field(KineticEnergy(model))
ε = Field(KineticEnergyDissipationRate(model))
simulation.output_writers[:netcdf_writer] = NetCDFOutputWriter(model, (; ke, ε), filename="out.nc", schedule=TimeInterval(2))
run!(simulation)
```

(Note that `(; tke, ε)` is a shorthand for `(tke=tke, ε=ε)`.)

