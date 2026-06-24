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
# pkg"add Oceananigans, Oceanostics, CairoMakie"
# ```

# ## Model and simulation setup

using Oceananigans

# We work with nondimensional quantities, following the standard nondimensionalization of the
# stratified shear layer ([Kaminski and Smyth, 2019](https://doi.org/10.1017/jfm.2018.973)). We
# nondimensionalize the Boussinesq equations using the shear-layer half-width `h` as the length
# scale and the velocity scale `U` (half the velocity difference across the layer), so that time is
# measured in units of `h / U`. The flow is then governed by three nondimensional numbers — the
# Richardson number `Ri`, the Reynolds number `Re = U h / ν`, and the Prandtl number `Pr = ν / κ` —
# from which the viscosity `ν` and the buoyancy diffusivity `κ` follow:

U  = 1     # velocity scale (half the velocity difference across the shear layer)
h  = 1     # length scale (shear-layer half-width)
Ri = 0.1   # Richardson number
Re = 5e4   # Reynolds number
Pr = 1     # Prandtl number

ν = U * h / Re   # viscosity
κ = ν / Pr       # buoyancy diffusivity

# We begin by creating a model with this isotropic diffusivity and fifth-order advection on a `xz`
# grid, using a buoyancy `b` as the active scalar. We make the box one wavelength of the most
# unstable Kelvin-Helmholtz mode wide (`k_max = 0.4446 / h`; [Michalke,
# 1964](https://doi.org/10.1017/S0022112064000908)), so that the perturbation we seed below fits
# periodically:

N = 128
k_max = 0.4446 / h   # most unstable KH wavenumber (Michalke, 1964)
Lx = 2π / k_max      # one most-unstable wavelength
Lz = 10
grid = RectilinearGrid(size=(N, N), x=(-Lx/2, +Lx/2), z=(-Lz/2, +Lz/2), topology=(Periodic, Flat, Bounded))

model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = UpwindBiased(order=5),
                            closure = ScalarDiffusivity(; ν, κ),
                            buoyancy = BuoyancyTracer(), tracers = :b)

# We use hyperbolic tangent profiles with the *same* length scale `h` for both the shear flow and
# the stratification. The buoyancy jump `B₀ = U² Ri / h` is chosen so that the gradient Richardson
# number `N² / (∂u/∂z)²` reaches its minimum value `Ri = 0.1` — below the classical stability
# threshold of 1/4 — at the center of the shear layer (`z = 0`), where the flow is most unstable. To
# kick off the instability we perturb the vertical velocity `w` with the most unstable mode
# `sin(k_max x)`, localized to the shear layer by a Gaussian envelope `exp(-z²)` and given a random
# amplitude:

B₀ = U^2 * Ri / h
perturbation_amplitude = 5e-2

shear_flow(x, z) = U * tanh(z / h)
stratification(x, z) = B₀ * tanh(z / h)
perturbation(x, z) = perturbation_amplitude * abs(randn()) * exp(-z^2) * sin(x * k_max - π)

set!(model, u=shear_flow, b=stratification, w=perturbation)

#
# Next create an adaptive-time-step simulation using the model above:

simulation = Simulation(model, Δt=0.1, stop_time=200)

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

∫χᴰ = Integral(TracerVarianceEquation.DissipationRate(model, :b))
∫χ = Integral(TracerVarianceEquation.Diffusion(model, :b))


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

# Now we'll read the results using `FieldTimeSeries`

filepath = simulation.output_writers[:nc].filepath
Ri_t = FieldTimeSeries(filepath, "Ri")
Q_t  = FieldTimeSeries(filepath, "Q")
b_t  = FieldTimeSeries(filepath, "b")

# Volume-integrated quantities are scalar time series, so we read them directly with NCDatasets:

ds = NCDataset(filepath)
∫χ  = ds["∫χ"][:]
∫χᴰ = ds["∫χᴰ"][:]
close(ds)

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

Riₙ = @lift Ri_t[$n]
hm1 = heatmap!(ax1, Riₙ; colormap = :bwr, colorrange = (-1, +1))
Colorbar(fig[3, 1], hm1, vertical=false, height=8)

Qₙ = @lift Q_t[$n]
hm2 = heatmap!(ax2, Qₙ; colormap = :inferno, colorrange = (0, 0.2))
Colorbar(fig[3, 2], hm2, vertical=false, height=8)

bₙ = @lift b_t[$n]
hm3 = heatmap!(ax3, bₙ; colormap = :balance, colorrange = (-B₀, +B₀))
Colorbar(fig[3, 3], hm3, vertical=false, height=8);

# We now plot the time evolution of our integrated quantities

axb = Axis(fig[4, 1:3]; xlabel="Time", height=100)
times = b_t.times
lines!(axb, times, ∫χ,  label = "∫χdV")
lines!(axb, times, ∫χᴰ, label = "∫χᴰdV", linestyle=:dash)
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
# `TracerVarianceDissipationRate` and `TracerVarianceDiffusion` are implemented
# with a energy-conserving formulation, which means that (for `NoFlux` boundary conditions) their
# volume-integral should be exactly (up to machine precision) the same.
