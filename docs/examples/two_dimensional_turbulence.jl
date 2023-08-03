# # [Two-dimensional turbulence example](@id two_d_turbulence_example)
#
# In this example (based on the homonymous [Oceananigans
# one](https://clima.github.io/OceananigansDocumentation/stable/generated/two_dimensional_turbulence/))
# we simulate a 2D flow initialized with random noise and observe the flow evolve.
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie, Rasters"
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

# We use this model to create a simulation with a `TimeStepWizard` to maximize the Δt

simulation = Simulation(model, Δt=0.2, stop_time=50)

wizard = TimeStepWizard(cfl=0.8, diffusive_cfl=0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))


# ## Model diagnostics
#
# Up until now we have only used Oceananigans, but we can make use of Oceanostics for the first
# diagnostic we'll set-up: a progress messenger. Here we use a `SingleLineMessenger`, which
# displays relevant information in only one line.

using Oceanostics

progress = ProgressMessengers.SingleLineMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))


# Using Oceanostics we can easily calculate two important diagnostics, the kinetic energy KE and 
# its dissipation rate ε

KE = KineticEnergy(model)
ε = KineticEnergyDissipationRate(model)

# And we can define their volume-integrals

∫KE = Integral(KE)
∫ε = Integral(ε)

# We also create another integrated quantity that appears in the TKE evolution equation: the
# `KineticEnergyDiffusiveTerm`, which in our case is
#
# ```math
# \varepsilon^D = u_i \partial_j \tau_{ij}
# ```
# where ``\tau_{ij}`` is the diffusive flux of ``i`` momentum in the ``j``-th direction.
#

∫εᴰ = Integral(KineticEnergyDiffusiveTerm(model))

# The idea in calculating this term is that, in integrated form, all transport terms should equal
# zero and `∫εᴰ` should equal `∫ε`.
#
# We output the previous quantities to a NetCDF file

output_fields = (; KE, ε, ∫KE, ∫ε, ∫εᴰ)
filename = "two_dimensional_turbulence"
simulation.output_writers[:nc] = NetCDFOutputWriter(model, output_fields,
                                                    filename = joinpath(@__DIR__, filename),
                                                    schedule = TimeInterval(0.6),
                                                    overwrite_existing = true)


# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Now we'll read the results using Rasters.jl, which works somewhat similarly to Python's Xarray and
# can speed-up the work the workflow

using Rasters

ds = RasterStack(simulation.output_writers[:nc].filepath)

# In order to plot results, we use Makie.jl, for which Rasters.jl already has some recipes

using CairoMakie

set_theme!(Theme(fontsize = 24))
fig = Figure()

axis_kwargs = (xlabel = "x", ylabel = "y",
               aspect = DataAspect(),
               height = 300, width = 300)

ax1 = Axis(fig[2, 1]; title = "Kinetic energy", axis_kwargs...)
ax2 = Axis(fig[2, 2]; title = "Kinetic energy dissip rate", axis_kwargs...)

# Now we plot the snapshots and set the title

n = Observable(1)

# `n` above is a [`Makie.Observable`](https://docs.makie.org/stable/documentation/nodes/index.html),
# which allows us to animate things easily. Creating observable `KE` and `ε` can be done simply with

KEₙ = @lift ds.KE[zC=1, Ti=$n]
εₙ = @lift ds.ε[zC=1, Ti=$n]

# Note that, in Rasters, the `time` coordinate gets shortened to `Ti`.
#
# Now we plot the heatmaps, each with its own colorbar below

hm_KE = heatmap!(ax1, KEₙ, colormap = :plasma, colorrange=(0, 5e-2))
Colorbar(fig[3, 1], hm_KE; vertical=false, height=8, ticklabelsize=12)

hm_ε = heatmap!(ax2, εₙ, colormap = :inferno, colorrange=(0, 5e-5))
Colorbar(fig[3, 2], hm_ε; vertical=false, height=8, ticklabelsize=12)

# We now plot the time evolution of our integrated quantities

axis_kwargs = (xlabel = "Time",
               height=150, width=300)

ax3 = Axis(fig[4, 1]; axis_kwargs...)
times = dims(ds, :Ti)
lines!(ax3, Array(times), ds.∫KE)

ax4 = Axis(fig[4, 2]; axis_kwargs...)
lines!(ax4, Array(times), ds.∫ε, label="∫εdV")
lines!(ax4, Array(times), ds.∫εᴰ, label="∫εᴰdV", linestyle=:dash)
axislegend(ax4, labelsize=14)

# Now we mark the time by placing a vertical line in the bottom plots:

tₙ = @lift times[$n]
vlines!(ax3, tₙ, color=:black, linestyle=:dash)
vlines!(ax4, tₙ, color=:black, linestyle=:dash)

# and by creating a useful title

title = @lift "Time = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false);

# Next we adjust the total figure size based on our panels, which makes it look like this

resize_to_layout!(fig)
current_figure() # hide
fig

# Finally, we record a movie.

@info "Animating..."
record(fig, filename * ".mp4", 1:length(times), framerate=24) do i
    n[] = i
end
nothing #hide

# ![](two_dimensional_turbulence.mp4)
#
# Although simple, this example and the animation above already illustrate a couple of interesting
# things. First, the KE dissipation rate `ε` is distributed at much smaller scales than the KE,
# which is expected due to the second-order derivatives present in `ε`.
#
# Second, again as expected, the volume-integrated KE dissipation rate is the same as the
# volume-integrated KE diffusion term (since all the non-dissipation parts of the term
# volume-integrate to zero). In fact, both `KineticEnergyDissipationRate` and
# `KineticEnergyDiffusiveTerm` in Oceanostics are implemented in an energy-conserving form (i.e.,
# they use the exact same discretization scheme and interpolations as used in Oceananigans), so they
# agree to machine-precision, and are great for closing budgets.
