# # Kelvin-Helmholtz instability
#
# This example simulates a simple 2D Kelvin-Helmholtz instability and is based on the similar
# [Oceananigans
# example](https://clima.github.io/OceananigansDocumentation/stable/literated/kelvin_helmholtz_instability/).
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie, Rasters"
# ```

# ## Model and simulation setup

# We begin by creating a model with an isotropic diffusivity and fifth-order advection on a `xz`
# 128² grid using a buoyancy `b` as the active scalar. We'll work here with nondimensional
# quantities.

using Oceananigans

N = 128
L = 10
grid = RectilinearGrid(size=(N, N), x=(-L/2, +L/2), z=(-L/2, +L/2), topology=(Periodic, Flat, Bounded))

model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3,
                            advection = UpwindBiased(order=5),
                            closure = ScalarDiffusivity(ν=2e-5, κ=2e-5),
                            buoyancy = BuoyancyTracer(), tracers = :b)

# We use hyperbolic tangent functions for the initial conditions and set the maximum Richardson
# number below the threshold of 1/4. We also add some grid-scale small-amplitude noise to `u` to
# kick the instability off:


noise(x, z) = 2e-2 * randn()
shear_flow(x, z) = tanh(z) + noise(x, z)

Ri₀ = 0.1; h = 1/4
stratification(x, z) = h * Ri₀ * tanh(z / h)

set!(model, u=shear_flow, b=stratification)

#
# Next create an adaptive-time-step simulation using the model above:

simulation = Simulation(model, Δt=0.1, stop_time=100)

wizard = TimeStepWizard(cfl=0.8, max_Δt=1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))


# ## Model diagnostics
#
# We set-up a progress messenger using the `TimedMessenger`, which displays, among other
# information, the time step duration

using Oceanostics

progress = ProgressMessengers.TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))


# We can also define some useful diagnostics for of the flow, starting with the `RichardsonNumber`

Ri = RichardsonNumber(model)

# We also set-up the `QVelocityGradientTensorInvariant`, which is usually used for visualizing
# vortices in the flow:
Q = QVelocityGradientTensorInvariant(model)

# Q is one of the velocity gradient tensor invariants and it measures the amount of vorticity versus
# the strain in the flow and, when it's positive, indicates a vortex. This method of vortex
# visualization is called the [Q-criterion](https://tinyurl.com/mwv6fskc).
#
# Let's also keep track of the amount of buoyancy mixing by measuring the buoyancy
# variance dissipation rate and diffusive term. When volume-integrated, these two quantities should
# be equal.

∫χᴰ = Integral(TracerVarianceDissipationRate(model, :b))
∫χ = Integral(TracerVarianceDiffusiveTerm(model, :b))


# Now we write these quantities, along with `b`, to a NetCDF:

output_fields = (; Ri, Q, model.tracers.b, ∫χ, ∫χᴰ)

using NCDatasets
filename = "kelvin_helmholtz"
simulation.output_writers[:nc] = NetCDFWriter(model, output_fields,
                                              filename = joinpath(@__DIR__, filename),
                                              schedule = TimeInterval(1),
                                              overwrite_existing = true)


# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Now we'll read the results using Rasters.jl, which works somewhat similarly to Python's Xarray and
# can speed-up the work the workflow

using Rasters

ds = RasterStack(simulation.output_writers[:nc].filepath)

# We now use Makie to create the figure and its axes

using CairoMakie

set_theme!(Theme(fontsize = 24))
fig = Figure()

kwargs = (xlabel="x", ylabel="z", height=150, width=250)
ax1 = Axis(fig[2, 1]; title = "Ri", kwargs...)
ax2 = Axis(fig[2, 2]; title = "Q", kwargs...)
ax3 = Axis(fig[2, 3]; title = "b", kwargs...);

# Next we use `Observable`s to lift the values and plot heatmaps and their colorbars

n = Observable(1)

Riₙ = @lift set(ds.Ri[Ti=$n, y_aca=Near(0)], :x_caa => X, :z_aaf => Z)
hm1 = heatmap!(ax1, Riₙ; colormap = :bwr, colorrange = (-1, +1))
Colorbar(fig[3, 1], hm1, vertical=false, height=8)

Qₙ = @lift set(ds.Q[Ti=$n, y_aca=Near(0)], :x_caa => X, :z_aac => Z)
hm2 = heatmap!(ax2, Qₙ; colormap = :inferno, colorrange = (0, 0.2))
Colorbar(fig[3, 2], hm2, vertical=false, height=8)

bₙ = @lift set(ds.b[Ti=$n, y_aca=Near(0)], :x_caa => X, :z_aac => Z)
hm3 = heatmap!(ax3, bₙ; colormap = :balance, colorrange = (-2.5e-2, +2.5e-2))
Colorbar(fig[3, 3], hm3, vertical=false, height=8);

# We now plot the time evolution of our integrated quantities

axb = Axis(fig[4, 1:3]; xlabel="Time", height=100)
times = dims(ds, :Ti)
lines!(axb, Array(times), Array(ds.∫χ),  label = "∫χdV")
lines!(axb, Array(times), Array(ds.∫χᴰ), label = "∫χᴰdV", linestyle=:dash)
axislegend(position=:lb, labelsize=14)

# Now we mark the time by placing a vertical line in the bottom panel and adding a helpful title

tₙ = @lift times[$n]
vlines!(axb, tₙ, color=:black, linestyle=:dash)

title = @lift "Time = " * string(round(times[$n], digits=2))
fig[1, 1:3] = Label(fig, title, fontsize=24, tellwidth=false);

# Finally, we adjust the figure dimensions to fit all the panels and record a movie

resize_to_layout!(fig)

@info "Animating..."
record(fig, filename * ".mp4", 1:length(times), framerate=10) do i
       n[] = i
end

# ![](kelvin_helmholtz.mp4)
#
# Similarly to the kinetic energy dissipation rate (see the [Two-dimensional turbulence example](@ref two_d_turbulence_example)),
# `TracerVarianceDissipationRate` and `TracerVarianceDiffusiveTerm` are implemented
# with a energy-conserving formulation, which means that (for `NoFlux` boundary conditions) their
# volume-integral should be exactly (up to machine precision) the same.
