# # Two dimensional turbulence example
#
# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

# ## Model and simulation setup

# We begin by creating a model with an isotropic diffusivity and fifth-order advection on a 128²
# grid.

using Oceananigans

grid = RectilinearGrid(size=(128, 128), extent=(2π, 2π), topology=(Periodic, Periodic, Flat))

model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            closure = ScalarDiffusivity(ν=1e-5))

# Let's give the model zero-mean grid-scale white noise as the initial condition

using Statistics

u, v, w = model.velocities

noise(x, y, z) = rand()
set!(model, u=noise, v=noise)

u .-= mean(u)
v .-= mean(v)

# We use this model to create a simulation with a `TimeStepWizard` to maxime the Δt

simulation = Simulation(model, Δt=0.2, stop_time=50)

wizard = TimeStepWizard(cfl=0.8, diffusive_cfl=0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))


# ## Model diagnostics
#
# Up until now we have only used Oceananigans, but we can make use of Oceanostics for the first
# diagnostic we'll set-up: a progress messenger

using Oceanostics

progress = TimedProgressMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))


# Using Oceanostics we can easily calculate two important dignostics, the kinetic energy KE and 
# its dissipation rate ε

KE = KineticEnergy(model)
ε = KineticEnergyDissipationRate(model)

# And we can define their volume-integrated forms

∫KE = Integral(KE)
∫ε = Integral(ε)

# We output the previous quantities to a NetCDF file

output_fields = (; KE, ε, ∫KE, ∫ε)
filename = "two_dimensional_turbulence"
simulation.output_writers[:nc] = NetCDFOutputWriter(model, output_fields,
                                                    filename = joinpath(@__DIR__, filename),
                                                    schedule = TimeInterval(0.6),
                                                    overwrite_existing = true)


# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Now we'll read the results using Rasters.jl, which works somewhat similarly to Python's Xarray

using Rasters

ds = RasterStack(simulation.output_writers[:nc].filepath)

# In order to plot results, we use Makie.jl, for which Rasters.jl already has recipes

using CairoMakie

set_theme!(Theme(fontsize = 24))

@info "Making a neat movie of vorticity and speed..."

fig = Figure(resolution = (800, 500))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = DataAspect())


ax1 = Axis(fig[2, 1]; title = "Kinetic energy", axis_kwargs...)
ax2 = Axis(fig[2, 2]; title = "Kinetic energy dissipation rate", axis_kwargs...)

# Now we plot the snapshots and set the title

n = Observable(1)

KEₙ = @lift ds.KE[:, :, 1, $n]
εₙ = @lift ds.ε[:, :, 1, $n]

hm_KE = heatmap!(ax1, KEₙ, colormap = :matter)
Colorbar(fig[3, 1], hm_KE; vertical=false, height=8, ticklabelsize=12)

hm_ε = heatmap!(ax2, εₙ, colormap = :inferno)
Colorbar(fig[3, 2], hm_ε; vertical=false, height=8, ticklabelsize=12)

times = dims(ds, :Ti)
title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)

resize_to_layout!(fig)
current_figure() # hide
fig

# Finally, we record a movie.

frames = 1:length(times)

@info "Making a neat animation of vorticity and speed..."

record(fig, filename * ".mp4", frames, framerate=24) do i
    n[] = i
end
nothing #hide

# ![](two_dimensional_turbulence.mp4)
